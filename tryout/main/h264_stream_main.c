/*
 * H.264 stream over Wi-Fi for glasses (button start/stop).
 * ESP32-P4-WIFI6 + OV5647: capture -> H.264 encode -> TCP client to Jetson.
 * Assumes Wi-Fi stability; use protocol_examples_common (and esp_wifi_remote for P4).
 */

#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/errno.h>
#include <unistd.h>

#include "lwip/sockets.h"
#include "lwip/netdb.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "driver/gpio.h"

#include "protocol_examples_common.h"
#include "example_video_common.h"

#include "linux/videodev2.h"

#if CONFIG_EXAMPLE_H264_MAX_QP <= CONFIG_EXAMPLE_H264_MIN_QP
#error "CONFIG_EXAMPLE_H264_MAX_QP must be larger than CONFIG_EXAMPLE_H264_MIN_QP"
#endif

#define ENCODE_DEV_PATH                 ESP_VIDEO_H264_DEVICE_NAME
#define VIDEO_BUFFER_COUNT              2
#define VIDEO_ENCODER_BUFFER_COUNT      1
#define SKIP_STARTUP_FRAME_COUNT        2
#define BUTTON_DEBOUNCE_MS              200
#define STREAM_FRAME_HEADER_SIZE        4   /* length (big-endian) before each H.264 frame */

static const char *TAG = "h264_stream";

typedef struct {
    int cap_fd;
    uint32_t format;
    uint8_t *cap_buffer[VIDEO_BUFFER_COUNT];
    int m2m_fd;
    uint8_t *m2m_cap_buffer;
    uint32_t width;
    uint32_t height;
} stream_ctx_t;

static stream_ctx_t s_ctx;
static volatile bool s_streaming = false;
static volatile bool s_stop_requested = false;
static int s_sock = -1;

static void print_cap(const struct v4l2_capability *cap)
{
    ESP_LOGI(TAG, "driver: %s card: %s", cap->driver, cap->card);
}

static esp_err_t init_capture(stream_ctx_t *ctx)
{
    int fd = open(EXAMPLE_CAM_DEV_PATH, O_RDONLY);
    if (fd < 0) {
        ESP_LOGE(TAG, "open cam failed");
        return ESP_FAIL;
    }
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
        close(fd);
        return ESP_FAIL;
    }
    print_cap(&cap);
    ctx->cap_fd = fd;
    return ESP_OK;
}

static esp_err_t set_codec_control(int fd, uint32_t ctrl_class, uint32_t id, int32_t value)
{
    struct v4l2_ext_controls ctrls = { 0 };
    struct v4l2_ext_control ctrl = { .id = id, .value = value };
    ctrls.ctrl_class = ctrl_class;
    ctrls.count = 1;
    ctrls.controls = &ctrl;
    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls) != 0) {
        ESP_LOGW(TAG, "set ctrl %" PRIu32 " failed", id);
        return ESP_FAIL;
    }
    return ESP_OK;
}

static esp_err_t init_codec(stream_ctx_t *ctx)
{
    int fd = open(ENCODE_DEV_PATH, O_RDONLY);
    if (fd < 0) {
        ESP_LOGE(TAG, "open H.264 device failed");
        return ESP_FAIL;
    }
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) != 0) {
        close(fd);
        return ESP_FAIL;
    }
    print_cap(&cap);

    set_codec_control(fd, V4L2_CID_CODEC_CLASS, V4L2_CID_MPEG_VIDEO_H264_I_PERIOD, CONFIG_EXAMPLE_H264_I_PERIOD);
    set_codec_control(fd, V4L2_CID_CODEC_CLASS, V4L2_CID_MPEG_VIDEO_BITRATE, CONFIG_EXAMPLE_H264_BITRATE);
    set_codec_control(fd, V4L2_CID_CODEC_CLASS, V4L2_CID_MPEG_VIDEO_H264_MIN_QP, CONFIG_EXAMPLE_H264_MIN_QP);
    set_codec_control(fd, V4L2_CID_CODEC_CLASS, V4L2_CID_MPEG_VIDEO_H264_MAX_QP, CONFIG_EXAMPLE_H264_MAX_QP);

    ctx->format = V4L2_PIX_FMT_H264;
    ctx->m2m_fd = fd;
    return ESP_OK;
}

static esp_err_t video_start(stream_ctx_t *ctx)
{
    struct v4l2_format fmt = { 0 };
    struct v4l2_requestbuffers req = { 0 };
    struct v4l2_buffer buf = { 0 };
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    uint32_t width, height;
    uint32_t capture_fmt = V4L2_PIX_FMT_YUV420;

    fmt.type = type;
    if (ioctl(ctx->cap_fd, VIDIOC_G_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "get cam format failed");
        return ESP_FAIL;
    }
    width = fmt.fmt.pix.width;
    height = fmt.fmt.pix.height;
    ctx->width = width;
    ctx->height = height;

    /* Camera capture */
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = capture_fmt;
    if (ioctl(ctx->cap_fd, VIDIOC_S_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "set cam format failed");
        return ESP_FAIL;
    }
    req.count = VIDEO_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->cap_fd, VIDIOC_REQBUFS, &req) != 0) {
        ESP_LOGE(TAG, "reqbufs cam failed");
        return ESP_FAIL;
    }
    for (int i = 0; i < VIDEO_BUFFER_COUNT; i++) {
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(ctx->cap_fd, VIDIOC_QUERYBUF, &buf) != 0) {
            return ESP_FAIL;
        }
        ctx->cap_buffer[i] = (uint8_t *)mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->cap_fd, buf.m.offset);
        if (ctx->cap_buffer[i] == MAP_FAILED) {
            return ESP_FAIL;
        }
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &buf);
    }

    /* Encoder input */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = capture_fmt;
    if (ioctl(ctx->m2m_fd, VIDIOC_S_FMT, &fmt) != 0) {
        return ESP_FAIL;
    }
    req.count = VIDEO_ENCODER_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    req.memory = V4L2_MEMORY_USERPTR;
    if (ioctl(ctx->m2m_fd, VIDIOC_REQBUFS, &req) != 0) {
        return ESP_FAIL;
    }

    /* Encoder output */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = ctx->format;
    if (ioctl(ctx->m2m_fd, VIDIOC_S_FMT, &fmt) != 0) {
        return ESP_FAIL;
    }
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->m2m_fd, VIDIOC_REQBUFS, &req) != 0) {
        return ESP_FAIL;
    }
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (ioctl(ctx->m2m_fd, VIDIOC_QUERYBUF, &buf) != 0) {
        return ESP_FAIL;
    }
    ctx->m2m_cap_buffer = (uint8_t *)mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->m2m_fd, buf.m.offset);
    if (ctx->m2m_cap_buffer == MAP_FAILED) {
        return ESP_FAIL;
    }
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    ioctl(ctx->m2m_fd, VIDIOC_QBUF, &buf);

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->m2m_fd, VIDIOC_STREAMON, &type);
    type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    ioctl(ctx->m2m_fd, VIDIOC_STREAMON, &type);
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->cap_fd, VIDIOC_STREAMON, &type);

    for (int i = 0; i < SKIP_STARTUP_FRAME_COUNT; i++) {
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        ioctl(ctx->cap_fd, VIDIOC_DQBUF, &buf);
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &buf);
    }

    ESP_LOGI(TAG, "Video started %" PRIu32 "x%" PRIu32, width, height);
    return ESP_OK;
}

static void video_stop(stream_ctx_t *ctx)
{
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->cap_fd, VIDIOC_STREAMOFF, &type);
    type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    ioctl(ctx->m2m_fd, VIDIOC_STREAMOFF, &type);
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->m2m_fd, VIDIOC_STREAMOFF, &type);
    ESP_LOGI(TAG, "Video stopped");
}

/* Returns encoded size in *out_len. Caller must call video_fb_return after using data. */
static esp_err_t video_fb_get(stream_ctx_t *ctx, uint8_t **out_buf, uint32_t *out_len)
{
    struct v4l2_buffer cap_buf = { 0 };
    struct v4l2_buffer m2m_out = { 0 };
    struct v4l2_buffer m2m_cap = { 0 };

    cap_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    cap_buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->cap_fd, VIDIOC_DQBUF, &cap_buf) != 0) {
        return ESP_FAIL;
    }

    m2m_out.index = 0;
    m2m_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    m2m_out.memory = V4L2_MEMORY_USERPTR;
    m2m_out.m.userptr = (unsigned long)ctx->cap_buffer[cap_buf.index];
    m2m_out.length = cap_buf.bytesused;
    if (ioctl(ctx->m2m_fd, VIDIOC_QBUF, &m2m_out) != 0) {
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &cap_buf);
        return ESP_FAIL;
    }

    m2m_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    m2m_cap.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->m2m_fd, VIDIOC_DQBUF, &m2m_cap) != 0) {
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &cap_buf);
        return ESP_FAIL;
    }

    ioctl(ctx->cap_fd, VIDIOC_QBUF, &cap_buf);
    ioctl(ctx->m2m_fd, VIDIOC_DQBUF, &m2m_out);

    *out_buf = ctx->m2m_cap_buffer;
    *out_len = m2m_cap.bytesused;
    return ESP_OK;
}

static void video_fb_return(stream_ctx_t *ctx)
{
    struct v4l2_buffer m2m_cap = { 0 };
    m2m_cap.index = 0;
    m2m_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    m2m_cap.memory = V4L2_MEMORY_MMAP;
    ioctl(ctx->m2m_fd, VIDIOC_QBUF, &m2m_cap);
}

/* Send one H.264 frame: 4-byte big-endian length + payload. */
static esp_err_t send_frame(int sock, const uint8_t *data, uint32_t len)
{
    uint8_t header[STREAM_FRAME_HEADER_SIZE];
    header[0] = (len >> 24) & 0xff;
    header[1] = (len >> 16) & 0xff;
    header[2] = (len >> 8) & 0xff;
    header[3] = len & 0xff;
    int sent = send(sock, header, sizeof(header), 0);
    if (sent != sizeof(header)) {
        return ESP_FAIL;
    }
    while (len > 0) {
        sent = send(sock, data, len, 0);
        if (sent <= 0) {
            return ESP_FAIL;
        }
        data += sent;
        len -= sent;
    }
    return ESP_OK;
}

static void stream_task(void *arg)
{
    stream_ctx_t *ctx = (stream_ctx_t *)arg;
    struct sockaddr_in dest = { 0 };
    int sock;

    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (s_stop_requested) {
            continue;
        }

        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            ESP_LOGE(TAG, "socket failed");
            s_streaming = false;
            continue;
        }
        dest.sin_family = AF_INET;
        dest.sin_port = htons(CONFIG_EXAMPLE_JETSON_PORT);
        if (inet_pton(AF_INET, CONFIG_EXAMPLE_JETSON_IP, &dest.sin_addr) != 1) {
            ESP_LOGE(TAG, "invalid IP");
            close(sock);
            s_streaming = false;
            continue;
        }
        if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
            ESP_LOGE(TAG, "connect to %s:%d failed", CONFIG_EXAMPLE_JETSON_IP, CONFIG_EXAMPLE_JETSON_PORT);
            close(sock);
            s_streaming = false;
            continue;
        }
        s_sock = sock;
        ESP_LOGI(TAG, "Connected to Jetson, streaming");

        if (video_start(ctx) != ESP_OK) {
            close(sock);
            s_sock = -1;
            s_streaming = false;
            continue;
        }

        while (!s_stop_requested) {
            uint8_t *buf;
            uint32_t len;
            if (video_fb_get(ctx, &buf, &len) != ESP_OK) {
                break;
            }
            if (send_frame(sock, buf, len) != ESP_OK) {
                video_fb_return(ctx);
                break;
            }
            video_fb_return(ctx);
        }

        video_stop(ctx);
        close(sock);
        s_sock = -1;
        s_streaming = false;
        ESP_LOGI(TAG, "Stream ended");
    }
}

static void button_task(void *arg)
{
    const int gpio = CONFIG_EXAMPLE_STREAM_BUTTON_GPIO;
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << gpio),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io);

    uint32_t last_toggle = 0;
    int last_level = gpio_get_level(gpio);

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(50));
        int level = gpio_get_level(gpio);
        if (level == last_level) {
            continue;
        }
        last_level = level;
        uint32_t now = xTaskGetTickCount() * portTICK_PERIOD_MS;
        if (now - last_toggle < BUTTON_DEBOUNCE_MS) {
            continue;
        }
        last_toggle = now;

        /* Button pressed (assuming active-low): toggle stream. */
        if (level == 0) {
            if (!s_streaming) {
                s_stop_requested = false;
                s_streaming = true;
                xTaskNotifyGive((TaskHandle_t)arg);
            } else {
                s_stop_requested = true;
            }
        }
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(example_connect());

    ESP_ERROR_CHECK(example_video_init());
    ESP_ERROR_CHECK(init_capture(&s_ctx));
    ESP_ERROR_CHECK(init_codec(&s_ctx));

    TaskHandle_t stream_task_handle = NULL;
    xTaskCreate(stream_task, "stream", 4096, &s_ctx, 5, &stream_task_handle);
    xTaskCreate(button_task, "btn", 2048, stream_task_handle, 5, NULL);

    ESP_LOGI(TAG, "Ready. Press button to start H.264 stream to %s:%d. Press again to stop.",
             CONFIG_EXAMPLE_JETSON_IP, CONFIG_EXAMPLE_JETSON_PORT);
}
