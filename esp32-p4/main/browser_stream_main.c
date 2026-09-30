/*
 * Direct browser stream for the Waveshare ESP32-P4-WIFI6 + OV5647.
 *
 * The camera produces an RGB565 frame through the ISP. The P4 JPEG encoder
 * converts each frame to JPEG, and the HTTP server exposes multipart MJPEG:
 *
 *   http://<board-ip>/
 *   http://<board-ip>/stream
 *
 * MJPEG is intentionally used here because browsers can display it in an
 * <img> element without a native H.264/WebRTC player.
 */

#include <stdbool.h>
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
#include "esp_log.h"
#include "esp_netif.h"
#include "nvs_flash.h"

#include "protocol_examples_common.h"
#include "example_video_common.h"
#include "esp_video_device.h"
#include "linux/videodev2.h"

#define CAMERA_BUFFER_COUNT CONFIG_EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER
#define HTTP_PORT            CONFIG_EXAMPLE_HTTP_PORT
#define JPEG_QUALITY         CONFIG_EXAMPLE_JPEG_COMPRESSION_QUALITY
#define HTTP_BOUNDARY        CONFIG_EXAMPLE_HTTP_PART_BOUNDARY

static const char *TAG = "browser_stream";

static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" HTTP_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" HTTP_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %" PRIu32 "\r\n\r\n";

typedef struct {
    int fd;
    uint32_t buffer_count;
    uint8_t *buffer[CAMERA_BUFFER_COUNT];
    size_t buffer_length[CAMERA_BUFFER_COUNT];
    uint32_t buffer_size;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    uint32_t frame_rate;

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

    /* RGB565 is supported by the P4 JPEG encoder on all supported P4 revisions. */
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = CONFIG_EXAMPLE_FRAME_WIDTH;
    format.fmt.pix.height = CONFIG_EXAMPLE_FRAME_HEIGHT;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(camera->fd, VIDIOC_S_FMT, &format) != 0) {
        ESP_LOGW(TAG, "camera rejected RGB565 format; using its negotiated format");
        memset(&format, 0, sizeof(format));
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(camera->fd, VIDIOC_G_FMT, &format) != 0) {
            ESP_LOGE(TAG, "failed to get camera format");
            return camera_cleanup(camera);
        }
    }

    camera->width = format.fmt.pix.width;
    camera->height = format.fmt.pix.height;
    camera->pixel_format = format.fmt.pix.pixelformat;

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
        return camera_cleanup(camera);
    }
    camera->buffer_count = request.count;

    for (uint32_t i = 0; i < camera->buffer_count; ++i) {
        struct v4l2_buffer buffer = { 0 };
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;

        if (ioctl(camera->fd, VIDIOC_QUERYBUF, &buffer) != 0) {
            ESP_LOGE(TAG, "failed to query camera buffer %" PRIu32, i);
            return camera_cleanup(camera);
        }

        camera->buffer_length[i] = buffer.length;
        camera->buffer[i] = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, camera->fd, buffer.m.offset);
        if (camera->buffer[i] == MAP_FAILED) {
            camera->buffer[i] = NULL;
            ESP_LOGE(TAG, "failed to map camera buffer %" PRIu32, i);
            return camera_cleanup(camera);
        }
        camera->buffer_size = buffer.length;

        if (ioctl(camera->fd, VIDIOC_QBUF, &buffer) != 0) {
            ESP_LOGE(TAG, "failed to queue camera buffer %" PRIu32, i);
            return camera_cleanup(camera);
        }
    }

    if (camera->pixel_format != V4L2_PIX_FMT_JPEG) {
        example_encoder_config_t encoder_config = {
            .width = camera->width,
            .height = camera->height,
            .pixel_format = camera->pixel_format,
            .quality = JPEG_QUALITY,
        };
        if (example_encoder_init(&encoder_config, &camera->jpeg_encoder) != ESP_OK) {
            ESP_LOGE(TAG, "failed to initialize JPEG encoder");
            goto fail;
        }
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

    while (true) {
        struct v4l2_buffer buffer = { 0 };
        uint8_t *jpeg_data = NULL;
        uint32_t jpeg_size = 0;
        bool locked = false;

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

        if (camera->pixel_format == V4L2_PIX_FMT_JPEG) {
            jpeg_data = camera->buffer[buffer.index];
            jpeg_size = buffer.bytesused;
        } else {
            ret = example_encoder_process(camera->jpeg_encoder,
                                          camera->buffer[buffer.index],
                                          camera->buffer_size,
                                          camera->jpeg_buffer,
                                          camera->jpeg_buffer_size,
                                          &jpeg_size);
            if (ret != ESP_OK) {
                ESP_LOGW(TAG, "JPEG encoding failed: %s", esp_err_to_name(ret));
                goto frame_done;
            }
            jpeg_data = camera->jpeg_buffer;
        }

        ret = httpd_resp_send_chunk(request, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (ret != ESP_OK) {
            goto frame_done;
        }

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
