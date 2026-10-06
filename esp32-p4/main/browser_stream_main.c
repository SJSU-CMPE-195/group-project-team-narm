/*
 * Direct browser stream for the Waveshare ESP32-P4-WIFI6 + OV5647.
 *
 * The camera produces an RGB565 frame through the ISP. Frames are resized
 * to the configured stream dimensions before P4 hardware JPEG encoding,
 * and the HTTP server exposes multipart MJPEG:
 *
 *   http://<board-ip>/
 *   http://<board-ip>/stream
 *
 * MJPEG is intentionally used here because browsers can display it in an
 * <img> element without a native H.264/WebRTC player.
 */

#include <stdbool.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "sdkconfig.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "esp_err.h"
#include "esp_event.h"
#include "esp_check.h"
#include "esp_http_server.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"

#include "protocol_examples_common.h"
#include "example_video_common.h"
#include "esp_video_device.h"
#include "linux/videodev2.h"
#include "rgb565_resize.h"

#define CAMERA_BUFFER_COUNT CONFIG_EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER
#define HTTP_PORT            CONFIG_EXAMPLE_HTTP_PORT
#define JPEG_QUALITY         CONFIG_EXAMPLE_JPEG_COMPRESSION_QUALITY
#define HTTP_BOUNDARY        CONFIG_EXAMPLE_HTTP_PART_BOUNDARY

static const char *TAG = "browser_stream";

static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" HTTP_BOUNDARY;
static const char *STREAM_PART = "\r\n--" HTTP_BOUNDARY "\r\n"
                               "Content-Type: image/jpeg\r\nContent-Length: %" PRIu32 "\r\n\r\n";

typedef struct {
    int fd;
    uint32_t buffer_count;
    uint8_t *buffer[CAMERA_BUFFER_COUNT];
    size_t buffer_length[CAMERA_BUFFER_COUNT];
    uint32_t width;
    uint32_t height;
    size_t stride;
    uint32_t pixel_format;
    uint32_t frame_rate;

    uint32_t stream_width;
    uint32_t stream_height;
    uint16_t *resize_buffer;
    uint32_t resize_buffer_size;
    uint32_t *resize_columns;
    size_t *resize_row_offsets;

    example_encoder_handle_t jpeg_encoder;
    uint8_t *jpeg_buffer;
    uint32_t jpeg_buffer_size;
    SemaphoreHandle_t frame_lock;
} browser_camera_t;

static browser_camera_t s_camera = {
    .fd = -1,
};

static esp_err_t camera_cleanup(browser_camera_t *camera)
{
    if (camera->fd >= 0) {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(camera->fd, VIDIOC_STREAMOFF, &type);
    }

    for (uint32_t i = 0; i < camera->buffer_count; ++i) {
        if (camera->buffer[i] != NULL) {
            munmap(camera->buffer[i], camera->buffer_length[i]);
            camera->buffer[i] = NULL;
        }
    }

    if (camera->jpeg_encoder != NULL) {
        if (camera->jpeg_buffer != NULL) {
            example_encoder_free_output_buffer(camera->jpeg_encoder, camera->jpeg_buffer);
            camera->jpeg_buffer = NULL;
        }
        example_encoder_deinit(camera->jpeg_encoder);
        camera->jpeg_encoder = NULL;
    }

    if (camera->frame_lock != NULL) {
        vSemaphoreDelete(camera->frame_lock);
        camera->frame_lock = NULL;
    }

    heap_caps_free(camera->resize_buffer);
    heap_caps_free(camera->resize_columns);
    heap_caps_free(camera->resize_row_offsets);
    camera->resize_buffer = NULL;
    camera->resize_columns = NULL;
    camera->resize_row_offsets = NULL;

    if (camera->fd >= 0) {
        close(camera->fd);
        camera->fd = -1;
    }

    return ESP_OK;
}

static esp_err_t camera_init(browser_camera_t *camera)
{
    esp_err_t ret = ESP_OK;
    struct v4l2_format format = { 0 };
    struct v4l2_streamparm streamparm = { 0 };
    struct v4l2_requestbuffers request = { 0 };
    const char *device = EXAMPLE_CAM_DEV_PATH;

    camera->fd = open(device, O_RDWR);
    if (camera->fd < 0) {
        ESP_LOGE(TAG, "failed to open camera device %s", device);
        return ESP_FAIL;
    }

    /* S_FMT only accepts the sensor's active dimensions. Read those first;
     * CONFIG_EXAMPLE_FRAME_WIDTH/HEIGHT describe the JPEG output size. */
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->fd, VIDIOC_G_FMT, &format) != 0) {
        ret = ESP_FAIL;
        ESP_LOGE(TAG, "failed to get camera format");
        goto fail;
    }

    /* RGB565 is supported by the P4 JPEG encoder on all supported P4 revisions. */
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(camera->fd, VIDIOC_S_FMT, &format) != 0) {
        ESP_LOGW(TAG, "camera rejected RGB565 format; using its negotiated format");
    }
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->fd, VIDIOC_G_FMT, &format) != 0) {
        ret = ESP_FAIL;
        ESP_LOGE(TAG, "failed to get negotiated camera format");
        goto fail;
    }

    camera->width = format.fmt.pix.width;
    camera->height = format.fmt.pix.height;
    camera->pixel_format = format.fmt.pix.pixelformat;
    camera->stride = format.fmt.pix.bytesperline;
    camera->stream_width = CONFIG_EXAMPLE_FRAME_WIDTH;
    camera->stream_height = CONFIG_EXAMPLE_FRAME_HEIGHT;

    ESP_GOTO_ON_FALSE(camera->width && camera->height &&
                      camera->stream_width <= camera->width &&
                      camera->stream_height <= camera->height,
                      ESP_ERR_INVALID_ARG, fail, TAG,
                      "stream dimensions must fit within the camera frame");

    bool resize = camera->stream_width != camera->width ||
                  camera->stream_height != camera->height;
    if (resize) {
        ESP_GOTO_ON_FALSE(camera->pixel_format == V4L2_PIX_FMT_RGB565,
                          ESP_ERR_NOT_SUPPORTED, fail, TAG,
                          "resizing requires RGB565 camera output");
        if (camera->stride == 0) {
            camera->stride = camera->width * sizeof(uint16_t);
        }
        camera->resize_buffer_size = camera->stream_width * camera->stream_height * sizeof(uint16_t);
        camera->resize_buffer = heap_caps_malloc(camera->resize_buffer_size,
                                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        camera->resize_columns = heap_caps_malloc(camera->stream_width * sizeof(uint32_t),
                                                  MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        camera->resize_row_offsets = heap_caps_malloc(camera->stream_height * sizeof(size_t),
                                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        ESP_GOTO_ON_FALSE(camera->resize_buffer && camera->resize_columns &&
                          camera->resize_row_offsets, ESP_ERR_NO_MEM, fail, TAG,
                          "failed to allocate RGB565 resize buffers");
        ESP_GOTO_ON_FALSE(rgb565_resize_prepare(camera->width, camera->height,
                                                camera->stride, camera->stream_width,
                                                camera->stream_height, camera->resize_columns,
                                                camera->resize_row_offsets),
                          ESP_ERR_INVALID_ARG, fail, TAG, "invalid RGB565 resize dimensions/stride");
    }

    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    streamparm.parm.capture.capability = V4L2_CAP_TIMEPERFRAME;
    streamparm.parm.capture.timeperframe.numerator = 1;
    streamparm.parm.capture.timeperframe.denominator = CONFIG_EXAMPLE_FRAME_FPS;
    if (ioctl(camera->fd, VIDIOC_S_PARM, &streamparm) != 0) {
        ESP_LOGW(TAG, "camera rejected requested FPS; using its default");
    }

    memset(&streamparm, 0, sizeof(streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->fd, VIDIOC_G_PARM, &streamparm) == 0 &&
        streamparm.parm.capture.timeperframe.numerator != 0) {
        camera->frame_rate = streamparm.parm.capture.timeperframe.denominator /
                             streamparm.parm.capture.timeperframe.numerator;
    }
    if (camera->frame_rate == 0) {
        camera->frame_rate = CONFIG_EXAMPLE_FRAME_FPS;
    }

    request.count = CAMERA_BUFFER_COUNT;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (ioctl(camera->fd, VIDIOC_REQBUFS, &request) != 0 || request.count < 2 ||
        request.count > CAMERA_BUFFER_COUNT) {
        ESP_LOGE(TAG, "failed to request camera buffers; count=%" PRIu32, request.count);
        ret = ESP_FAIL;
        goto fail;
    }
    camera->buffer_count = request.count;

    for (uint32_t i = 0; i < camera->buffer_count; ++i) {
        struct v4l2_buffer buffer = { 0 };
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;

        if (ioctl(camera->fd, VIDIOC_QUERYBUF, &buffer) != 0) {
            ESP_LOGE(TAG, "failed to query camera buffer %" PRIu32, i);
            ret = ESP_FAIL;
            goto fail;
        }

        if (resize) {
            size_t required_size = (camera->height - 1) * camera->stride +
                                   camera->width * sizeof(uint16_t);
            ESP_GOTO_ON_FALSE(buffer.length >= required_size, ESP_ERR_INVALID_SIZE,
                              fail, TAG, "camera buffer is too small for RGB565 resize");
        }

        camera->buffer_length[i] = buffer.length;
        camera->buffer[i] = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, camera->fd, buffer.m.offset);
        if (camera->buffer[i] == MAP_FAILED) {
            camera->buffer[i] = NULL;
            ESP_LOGE(TAG, "failed to map camera buffer %" PRIu32, i);
            ret = ESP_FAIL;
            goto fail;
        }

        if (ioctl(camera->fd, VIDIOC_QBUF, &buffer) != 0) {
            ESP_LOGE(TAG, "failed to queue camera buffer %" PRIu32, i);
            ret = ESP_FAIL;
            goto fail;
        }
    }

    if (camera->pixel_format != V4L2_PIX_FMT_JPEG) {
        example_encoder_config_t encoder_config = {
            .width = camera->stream_width,
            .height = camera->stream_height,
            .pixel_format = camera->pixel_format,
            .quality = JPEG_QUALITY,
        };
        ESP_GOTO_ON_ERROR(example_encoder_init(&encoder_config, &camera->jpeg_encoder),
                          fail, TAG, "failed to initialize JPEG encoder");
        ESP_GOTO_ON_ERROR(example_encoder_alloc_output_buffer(camera->jpeg_encoder,
                                                              &camera->jpeg_buffer,
                                                              &camera->jpeg_buffer_size),
                          fail, TAG, "failed to allocate JPEG output buffer");
    }

    camera->frame_lock = xSemaphoreCreateMutex();
    if (camera->frame_lock == NULL) {
        ESP_LOGE(TAG, "failed to create frame mutex");
        goto fail;
    }

    {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(camera->fd, VIDIOC_STREAMON, &type) != 0) {
            ESP_LOGE(TAG, "failed to start camera stream");
            goto fail;
        }
    }

    ESP_LOGI(TAG, "camera ready: %" PRIu32 "x%" PRIu32 " @ %" PRIu32
             " fps, pixel format=0x%08" PRIx32, camera->width, camera->height,
             camera->frame_rate, camera->pixel_format);
    ESP_LOGI(TAG, "MJPEG output: %" PRIu32 "x%" PRIu32 ", JPEG quality=%d, resize=%s",
             camera->stream_width, camera->stream_height, JPEG_QUALITY,
             resize ? "RGB565 nearest-neighbor" : "none");
    return ESP_OK;

fail:
    camera_cleanup(camera);
    return ret == ESP_OK ? ESP_FAIL : ret;
}

static esp_err_t root_handler(httpd_req_t *request)
{
    static const char html[] =
        "<!doctype html><html><head>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>ESP32-P4 Camera</title></head>"
        "<body style='margin:0;background:#111;color:#eee;font-family:sans-serif'>"
        "<h2 style='margin:12px'>ESP32-P4 Camera</h2>"
        "<img src='/stream' style='display:block;max-width:100%;height:auto'>"
        "</body></html>";

    httpd_resp_set_type(request, "text/html; charset=utf-8");
    return httpd_resp_send(request, html, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t stream_handler(httpd_req_t *request)
{
    char part_header[128];
    browser_camera_t *camera = &s_camera;
    esp_err_t ret = httpd_resp_set_type(request, STREAM_CONTENT_TYPE);
    if (ret != ESP_OK) {
        return ret;
    }
    httpd_resp_set_hdr(request, "Cache-Control", "no-cache, private");
    httpd_resp_set_hdr(request, "Pragma", "no-cache");
    httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", "*");

    /* MJPEG/HTTP chunk framing contains small writes. Avoid Nagle waiting
     * for acknowledgements before sending the next header or JPEG. */
    int no_delay = 1;
    int socket_fd = httpd_req_to_sockfd(request);
    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay)) != 0) {
        ESP_LOGW(TAG, "could not enable TCP_NODELAY: errno=%d", errno);
    } else {
        ESP_LOGI(TAG, "MJPEG TCP_NODELAY enabled, send buffer=%d bytes",
                 CONFIG_LWIP_TCP_SND_BUF_DEFAULT);
    }

    int64_t stats_started = esp_timer_get_time();
    uint32_t sent_frames = 0;
    uint64_t sent_bytes = 0;
    int64_t wait_us = 0, resize_us = 0, encode_us = 0, send_us = 0;

    while (true) {
        struct v4l2_buffer buffer = { 0 };
        uint8_t *jpeg_data = NULL;
        uint32_t jpeg_size = 0;
        bool locked = false;
        int64_t frame_started = esp_timer_get_time();

        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (ioctl(camera->fd, VIDIOC_DQBUF, &buffer) != 0) {
            ESP_LOGW(TAG, "camera dequeue failed; closing browser stream");
            return ESP_FAIL;
        }

        if (xSemaphoreTake(camera->frame_lock, portMAX_DELAY) != pdTRUE) {
            ioctl(camera->fd, VIDIOC_QBUF, &buffer);
            return ESP_FAIL;
        }
        locked = true;

        int64_t frame_ready = esp_timer_get_time();
        int64_t resized_at = frame_ready;
        int64_t encoded_at = frame_ready;

        if (camera->pixel_format == V4L2_PIX_FMT_JPEG) {
            jpeg_data = camera->buffer[buffer.index];
            jpeg_size = buffer.bytesused;
        } else {
            uint8_t *encoder_input = camera->buffer[buffer.index];
            uint32_t encoder_input_size = camera->buffer_length[buffer.index];
            if (camera->resize_buffer != NULL) {
                rgb565_resize_frame(encoder_input, camera->resize_buffer,
                                     camera->width, camera->stream_width, camera->stream_height,
                                     camera->resize_columns, camera->resize_row_offsets);
                encoder_input = (uint8_t *)camera->resize_buffer;
                encoder_input_size = camera->resize_buffer_size;
            }
            resized_at = esp_timer_get_time();
            ret = example_encoder_process(camera->jpeg_encoder,
                                          encoder_input, encoder_input_size,
                                          camera->jpeg_buffer,
                                          camera->jpeg_buffer_size,
                                          &jpeg_size);
            if (ret != ESP_OK) {
                ESP_LOGW(TAG, "JPEG encoding failed: %s", esp_err_to_name(ret));
                goto frame_done;
            }
            encoded_at = esp_timer_get_time();
            jpeg_data = camera->jpeg_buffer;
        }

        /* Send the boundary and part header together, reducing tiny writes. */
        int header_len = snprintf(part_header, sizeof(part_header), STREAM_PART, jpeg_size);
        if (header_len <= 0 || header_len >= (int)sizeof(part_header)) {
            ret = ESP_FAIL;
            goto frame_done;
        }
        ret = httpd_resp_send_chunk(request, part_header, header_len);
        if (ret != ESP_OK) {
            goto frame_done;
        }
        ret = httpd_resp_send_chunk(request, (const char *)jpeg_data, jpeg_size);
        if (ret != ESP_OK) {
            goto frame_done;
        }
        ret = httpd_resp_send_chunk(request, "\r\n", 2);
        if (ret == ESP_OK) {
            int64_t sent_at = esp_timer_get_time();
            ++sent_frames;
            sent_bytes += jpeg_size;
            wait_us += frame_ready - frame_started;
            resize_us += resized_at - frame_ready;
            encode_us += encoded_at - resized_at;
            send_us += sent_at - encoded_at;

            if (sent_at - stats_started >= CONFIG_EXAMPLE_STATS_INTERVAL_SEC * 1000000LL) {
                ESP_LOGI(TAG, "MJPEG stats: %" PRIu32 "x%" PRIu32 " sent=%" PRIu32
                         " fps=%.1f avg_jpeg=%.1f KiB wait=%.1f resize=%.1f encode=%.1f send=%.1f ms",
                         camera->stream_width, camera->stream_height, sent_frames,
                         sent_frames * 1000000.0 / (sent_at - stats_started),
                         (double)sent_bytes / sent_frames / 1024.0,
                         wait_us / (sent_frames * 1000.0), resize_us / (sent_frames * 1000.0),
                         encode_us / (sent_frames * 1000.0), send_us / (sent_frames * 1000.0));
                stats_started = sent_at;
                sent_frames = 0;
                sent_bytes = 0;
                wait_us = resize_us = encode_us = send_us = 0;
            }
        }

frame_done:
        if (locked) {
            xSemaphoreGive(camera->frame_lock);
        }
        if (ioctl(camera->fd, VIDIOC_QBUF, &buffer) != 0) {
            ESP_LOGW(TAG, "camera queue failed; closing browser stream");
            return ESP_FAIL;
        }
        if (ret != ESP_OK) {
            return ret;
        }
    }
}

static esp_err_t http_server_start(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = HTTP_PORT;
    config.stack_size = 8192;

    httpd_handle_t server = NULL;
    ESP_RETURN_ON_ERROR(httpd_start(&server, &config), TAG, "failed to start HTTP server");

    httpd_uri_t root = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = root_handler,
    };
    httpd_uri_t stream = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = stream_handler,
    };

    esp_err_t ret = httpd_register_uri_handler(server, &root);
    if (ret == ESP_OK) {
        ret = httpd_register_uri_handler(server, &stream);
    }
    if (ret != ESP_OK) {
        httpd_stop(server);
        return ret;
    }
    return ESP_OK;
}

static void configure_stream_wifi(void)
{
#if CONFIG_EXAMPLE_CONNECT_WIFI
    /* Low-latency streaming favors keeping the radio awake over power saving. */
    esp_err_t ret = esp_wifi_set_ps(WIFI_PS_NONE);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "could not disable Wi-Fi power saving: %s", esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "Wi-Fi power saving disabled for streaming");
    }

    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
        ESP_LOGI(TAG, "Wi-Fi link: RSSI=%d dBm, channel=%u", ap_info.rssi, ap_info.primary);
    }
#endif
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    } else {
        ESP_ERROR_CHECK(ret);
    }

    ESP_ERROR_CHECK(example_video_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(example_connect());
    configure_stream_wifi();
    ESP_ERROR_CHECK(camera_init(&s_camera));
    ESP_ERROR_CHECK(http_server_start());

    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    esp_netif_ip_info_t ip_info;
    if (netif != NULL && esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
        ESP_LOGI(TAG, "open http://" IPSTR ":%d/ in a browser", IP2STR(&ip_info.ip), HTTP_PORT);
    } else {
        ESP_LOGI(TAG, "HTTP server ready on port %d; use the IP shown by Wi-Fi logs", HTTP_PORT);
    }
    ESP_LOGI(TAG, "MJPEG stream endpoint: /stream");
}
