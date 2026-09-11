/*
 * Low-latency H.264 stream for the Waveshare ESP32-P4-WIFI6 + OV5647.
 *
 * Pipeline:
 *   MIPI-CSI capture -> P4 hardware H.264 encoder -> bounded encoded-buffer
 *   queue -> TCP client -> Jetson
 *
 * Capture/encode is separate from TCP transmission. Two MMAP'd encoder
 * output buffers absorb short Wi-Fi stalls without copying compressed frames.
 * Encoded frames are never discarded because later P-frames may depend on them.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include "lwip/sockets.h"
#include "lwip/netdb.h"

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"

#include "protocol_examples_common.h"
#include "example_video_common.h"
#include "linux/videodev2.h"

#if CONFIG_EXAMPLE_H264_MAX_QP <= CONFIG_EXAMPLE_H264_MIN_QP
#error "CONFIG_EXAMPLE_H264_MAX_QP must be larger than CONFIG_EXAMPLE_H264_MIN_QP"
#endif

#define ENCODE_DEV_PATH                  ESP_VIDEO_H264_DEVICE_NAME
#define CAMERA_BUFFER_COUNT              4
#define ENCODER_INPUT_BUFFER_COUNT       1
#define ENCODED_BUFFER_COUNT             2
#define SKIP_STARTUP_FRAME_COUNT         2
#define BUTTON_DEBOUNCE_MS               200
#define STREAM_FRAME_HEADER_SIZE         4

static const char *TAG = "p4_h264_stream";

typedef struct {
    uint32_t index;
    uint32_t len;
    int64_t encoded_us;
} encoded_frame_t;

typedef struct {
    int cap_fd;
    int m2m_fd;
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t cap_buffer_count;
    uint32_t encoded_buffer_count;
    uint8_t *cap_buffer[CAMERA_BUFFER_COUNT];
    size_t cap_buffer_len[CAMERA_BUFFER_COUNT];
    uint8_t *encoded_buffer[ENCODED_BUFFER_COUNT];
    size_t encoded_buffer_len[ENCODED_BUFFER_COUNT];
    bool configured;
} stream_ctx_t;

static stream_ctx_t s_ctx = {
    .cap_fd = -1,
    .m2m_fd = -1,
};
static QueueHandle_t s_encoded_queue;
static TaskHandle_t s_stream_task_handle;
static volatile bool s_streaming;
static volatile bool s_stop_requested;
static volatile bool s_capture_running;
static volatile int s_sock = -1;
static volatile uint32_t s_encoded_frames;

static void print_cap(const struct v4l2_capability *cap)
{
    ESP_LOGI(TAG, "driver: %s card: %s", cap->driver, cap->card);
}

static esp_err_t init_capture(stream_ctx_t *ctx)
{
    ctx->cap_fd = open(EXAMPLE_CAM_DEV_PATH, O_RDONLY);
    if (ctx->cap_fd < 0) {
        ESP_LOGE(TAG, "open camera failed");
        return ESP_FAIL;
    }

    struct v4l2_capability cap = { 0 };
    if (ioctl(ctx->cap_fd, VIDIOC_QUERYCAP, &cap) != 0) {
        ESP_LOGE(TAG, "query camera capabilities failed");
        close(ctx->cap_fd);
        ctx->cap_fd = -1;
        return ESP_FAIL;
    }
    print_cap(&cap);
    return ESP_OK;
}

static esp_err_t set_codec_control(int fd, uint32_t ctrl_class, uint32_t id, int32_t value)
{
    struct v4l2_ext_control ctrl = {
        .id = id,
        .value = value,
    };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = ctrl_class,
        .count = 1,
        .controls = &ctrl,
    };

    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls) != 0) {
        ESP_LOGE(TAG, "set codec control %" PRIu32 "=%" PRId32 " failed", id, value);
        return ESP_FAIL;
    }
    return ESP_OK;
}

static esp_err_t init_codec(stream_ctx_t *ctx)
{
    ctx->m2m_fd = open(ENCODE_DEV_PATH, O_RDONLY);
    if (ctx->m2m_fd < 0) {
        ESP_LOGE(TAG, "open H.264 device failed");
        return ESP_FAIL;
    }

    struct v4l2_capability cap = { 0 };
    if (ioctl(ctx->m2m_fd, VIDIOC_QUERYCAP, &cap) != 0) {
        ESP_LOGE(TAG, "query H.264 capabilities failed");
        close(ctx->m2m_fd);
        ctx->m2m_fd = -1;
        return ESP_FAIL;
    }
    print_cap(&cap);

    ESP_RETURN_ON_ERROR(set_codec_control(ctx->m2m_fd, V4L2_CID_CODEC_CLASS,
                                         V4L2_CID_MPEG_VIDEO_H264_I_PERIOD,
                                         CONFIG_EXAMPLE_H264_I_PERIOD),
                        TAG, "set I-frame period failed");
    ESP_RETURN_ON_ERROR(set_codec_control(ctx->m2m_fd, V4L2_CID_CODEC_CLASS,
                                         V4L2_CID_MPEG_VIDEO_BITRATE,
                                         CONFIG_EXAMPLE_H264_BITRATE),
                        TAG, "set bitrate failed");
    ESP_RETURN_ON_ERROR(set_codec_control(ctx->m2m_fd, V4L2_CID_CODEC_CLASS,
                                         V4L2_CID_MPEG_VIDEO_H264_MIN_QP,
                                         CONFIG_EXAMPLE_H264_MIN_QP),
                        TAG, "set minimum QP failed");
    ESP_RETURN_ON_ERROR(set_codec_control(ctx->m2m_fd, V4L2_CID_CODEC_CLASS,
                                         V4L2_CID_MPEG_VIDEO_H264_MAX_QP,
                                         CONFIG_EXAMPLE_H264_MAX_QP),
                        TAG, "set maximum QP failed");

    ctx->format = V4L2_PIX_FMT_H264;
    return ESP_OK;
}

static esp_err_t configure_video(stream_ctx_t *ctx)
{
    if (ctx->configured) {
        return ESP_OK;
    }

    const uint32_t capture_fmt = V4L2_PIX_FMT_YUV420;
    struct v4l2_format fmt = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .fmt.pix = {
            .width = CONFIG_EXAMPLE_FRAME_WIDTH,
            .height = CONFIG_EXAMPLE_FRAME_HEIGHT,
            .pixelformat = capture_fmt,
            .field = V4L2_FIELD_ANY,
        },
    };
    if (ioctl(ctx->cap_fd, VIDIOC_S_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "set camera format failed");
        return ESP_FAIL;
    }
    ctx->width = fmt.fmt.pix.width;
    ctx->height = fmt.fmt.pix.height;

    struct v4l2_streamparm parm = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .parm.capture = {
            .capability = V4L2_CAP_TIMEPERFRAME,
            .timeperframe = {
                .numerator = 1,
                .denominator = CONFIG_EXAMPLE_FRAME_FPS,
            },
        },
    };
    if (ioctl(ctx->cap_fd, VIDIOC_S_PARM, &parm) != 0) {
        ESP_LOGW(TAG, "camera rejected explicit FPS; using sensor default");
    }

    struct v4l2_requestbuffers req = {
        .count = CAMERA_BUFFER_COUNT,
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .memory = V4L2_MEMORY_MMAP,
    };
    if (ioctl(ctx->cap_fd, VIDIOC_REQBUFS, &req) != 0 || req.count < 2) {
        ESP_LOGE(TAG, "request camera buffers failed (allocated=%" PRIu32 ")", req.count);
        return ESP_FAIL;
    }
    ctx->cap_buffer_count = req.count < CAMERA_BUFFER_COUNT ? req.count : CAMERA_BUFFER_COUNT;

    for (uint32_t i = 0; i < ctx->cap_buffer_count; ++i) {
        struct v4l2_buffer buf = {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            .memory = V4L2_MEMORY_MMAP,
            .index = i,
        };
        if (ioctl(ctx->cap_fd, VIDIOC_QUERYBUF, &buf) != 0) {
            ESP_LOGE(TAG, "query camera buffer %" PRIu32 " failed", i);
            return ESP_FAIL;
        }
        ctx->cap_buffer_len[i] = buf.length;
        ctx->cap_buffer[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, ctx->cap_fd, buf.m.offset);
        if (ctx->cap_buffer[i] == MAP_FAILED) {
            ctx->cap_buffer[i] = NULL;
            ESP_LOGE(TAG, "map camera buffer %" PRIu32 " failed", i);
            return ESP_FAIL;
        }
    }

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = ctx->width;
    fmt.fmt.pix.height = ctx->height;
    fmt.fmt.pix.pixelformat = capture_fmt;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(ctx->m2m_fd, VIDIOC_S_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "set encoder input format failed");
        return ESP_FAIL;
    }

    memset(&req, 0, sizeof(req));
    req.count = ENCODER_INPUT_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    req.memory = V4L2_MEMORY_USERPTR;
    if (ioctl(ctx->m2m_fd, VIDIOC_REQBUFS, &req) != 0 || req.count < 1) {
        ESP_LOGE(TAG, "request encoder input buffer failed");
        return ESP_FAIL;
    }

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = ctx->width;
    fmt.fmt.pix.height = ctx->height;
    fmt.fmt.pix.pixelformat = ctx->format;
    if (ioctl(ctx->m2m_fd, VIDIOC_S_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "set encoder output format failed");
        return ESP_FAIL;
    }

    memset(&req, 0, sizeof(req));
    req.count = ENCODED_BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->m2m_fd, VIDIOC_REQBUFS, &req) != 0 || req.count < 2) {
        ESP_LOGE(TAG, "request encoded buffers failed (allocated=%" PRIu32 ")", req.count);
        return ESP_FAIL;
    }
    ctx->encoded_buffer_count = req.count < ENCODED_BUFFER_COUNT ? req.count : ENCODED_BUFFER_COUNT;

    for (uint32_t i = 0; i < ctx->encoded_buffer_count; ++i) {
        struct v4l2_buffer buf = {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            .memory = V4L2_MEMORY_MMAP,
            .index = i,
        };
        if (ioctl(ctx->m2m_fd, VIDIOC_QUERYBUF, &buf) != 0) {
            ESP_LOGE(TAG, "query encoded buffer %" PRIu32 " failed", i);
            return ESP_FAIL;
        }
        ctx->encoded_buffer_len[i] = buf.length;
        ctx->encoded_buffer[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, ctx->m2m_fd, buf.m.offset);
        if (ctx->encoded_buffer[i] == MAP_FAILED) {
            ctx->encoded_buffer[i] = NULL;
            ESP_LOGE(TAG, "map encoded buffer %" PRIu32 " failed", i);
            return ESP_FAIL;
        }
    }

    ctx->configured = true;
    ESP_LOGI(TAG, "configured %" PRIu32 "x%" PRIu32 "@%d, bitrate=%d, camera_bufs=%" PRIu32
             ", encoded_bufs=%" PRIu32,
             ctx->width, ctx->height, CONFIG_EXAMPLE_FRAME_FPS,
             CONFIG_EXAMPLE_H264_BITRATE, ctx->cap_buffer_count,
             ctx->encoded_buffer_count);
    return ESP_OK;
}

static esp_err_t queue_mmap_buffer(int fd, enum v4l2_buf_type type, uint32_t index)
{
    struct v4l2_buffer buf = {
        .type = type,
        .memory = V4L2_MEMORY_MMAP,
        .index = index,
    };
    return ioctl(fd, VIDIOC_QBUF, &buf) == 0 ? ESP_OK : ESP_FAIL;
}

static esp_err_t video_start(stream_ctx_t *ctx)
{
    ESP_RETURN_ON_ERROR(configure_video(ctx), TAG, "video configuration failed");

    for (uint32_t i = 0; i < ctx->cap_buffer_count; ++i) {
        ESP_RETURN_ON_ERROR(queue_mmap_buffer(ctx->cap_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, i),
                            TAG, "queue camera buffer failed");
    }
    for (uint32_t i = 0; i < ctx->encoded_buffer_count; ++i) {
        ESP_RETURN_ON_ERROR(queue_mmap_buffer(ctx->m2m_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, i),
                            TAG, "queue encoded buffer failed");
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->m2m_fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "start encoder output stream failed");
        return ESP_FAIL;
    }
    type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(ctx->m2m_fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "start encoder input stream failed");
        return ESP_FAIL;
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cap_fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "start camera stream failed");
        return ESP_FAIL;
    }

    for (int i = 0; i < SKIP_STARTUP_FRAME_COUNT; ++i) {
        struct v4l2_buffer buf = {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            .memory = V4L2_MEMORY_MMAP,
        };
        if (ioctl(ctx->cap_fd, VIDIOC_DQBUF, &buf) != 0 ||
            ioctl(ctx->cap_fd, VIDIOC_QBUF, &buf) != 0) {
            ESP_LOGW(TAG, "could not skip startup frame %d", i);
            break;
        }
    }

    ESP_LOGI(TAG, "video started");
    return ESP_OK;
}

static void video_stop(stream_ctx_t *ctx)
{
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cap_fd, VIDIOC_STREAMOFF, &type) != 0) {
        ESP_LOGW(TAG, "stop camera stream failed");
    }
    type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(ctx->m2m_fd, VIDIOC_STREAMOFF, &type) != 0) {
        ESP_LOGW(TAG, "stop encoder input stream failed");
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->m2m_fd, VIDIOC_STREAMOFF, &type) != 0) {
        ESP_LOGW(TAG, "stop encoder output stream failed");
    }
    ESP_LOGI(TAG, "video stopped");
}

static esp_err_t video_fb_get(stream_ctx_t *ctx, encoded_frame_t *frame)
{
    struct v4l2_buffer camera = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .memory = V4L2_MEMORY_MMAP,
    };
    if (ioctl(ctx->cap_fd, VIDIOC_DQBUF, &camera) != 0) {
        return ESP_FAIL;
    }
    if (camera.index >= ctx->cap_buffer_count || (camera.flags & V4L2_BUF_FLAG_ERROR)) {
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &camera);
        return ESP_ERR_INVALID_RESPONSE;
    }

    struct v4l2_buffer encoder_input = {
        .index = 0,
        .type = V4L2_BUF_TYPE_VIDEO_OUTPUT,
        .memory = V4L2_MEMORY_USERPTR,
        .m.userptr = (unsigned long)ctx->cap_buffer[camera.index],
        .length = camera.bytesused,
        .bytesused = camera.bytesused,
    };
    if (ioctl(ctx->m2m_fd, VIDIOC_QBUF, &encoder_input) != 0) {
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &camera);
        return ESP_FAIL;
    }

    struct v4l2_buffer encoded = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .memory = V4L2_MEMORY_MMAP,
    };
    if (ioctl(ctx->m2m_fd, VIDIOC_DQBUF, &encoded) != 0) {
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &camera);
        return ESP_FAIL;
    }

    if (ioctl(ctx->m2m_fd, VIDIOC_DQBUF, &encoder_input) != 0) {
        queue_mmap_buffer(ctx->m2m_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, encoded.index);
        ioctl(ctx->cap_fd, VIDIOC_QBUF, &camera);
        return ESP_FAIL;
    }
    if (ioctl(ctx->cap_fd, VIDIOC_QBUF, &camera) != 0) {
        queue_mmap_buffer(ctx->m2m_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, encoded.index);
        return ESP_FAIL;
    }

    if (encoded.index >= ctx->encoded_buffer_count || encoded.bytesused == 0 ||
        (encoded.flags & V4L2_BUF_FLAG_ERROR)) {
        queue_mmap_buffer(ctx->m2m_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, encoded.index);
        return ESP_ERR_INVALID_RESPONSE;
    }

    frame->index = encoded.index;
    frame->len = encoded.bytesused;
    frame->encoded_us = esp_timer_get_time();
    return ESP_OK;
}

static void video_fb_return(stream_ctx_t *ctx, uint32_t index)
{
    if (index < ctx->encoded_buffer_count &&
        queue_mmap_buffer(ctx->m2m_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, index) != ESP_OK) {
        ESP_LOGE(TAG, "requeue encoded buffer %" PRIu32 " failed", index);
        s_stop_requested = true;
    }
}

static esp_err_t send_all(int sock, const uint8_t *data, size_t len)
{
    while (len > 0) {
        int sent = send(sock, data, len, 0);
        if (sent <= 0) {
            return ESP_FAIL;
        }
        data += sent;
        len -= (size_t)sent;
    }
    return ESP_OK;
}

static esp_err_t send_frame(int sock, const uint8_t *data, uint32_t len)
{
    const uint8_t header[STREAM_FRAME_HEADER_SIZE] = {
        (uint8_t)((len >> 24) & 0xff),
        (uint8_t)((len >> 16) & 0xff),
        (uint8_t)((len >> 8) & 0xff),
        (uint8_t)(len & 0xff),
    };
    ESP_RETURN_ON_ERROR(send_all(sock, header, sizeof(header)), TAG, "send header failed");
    return send_all(sock, data, len);
}

static int connect_jetson(void)
{
    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "socket creation failed");
        return -1;
    }

    int yes = 1;
    int send_buffer = CONFIG_EXAMPLE_TCP_SNDBUF_BYTES;
    struct timeval timeout = {
        .tv_sec = CONFIG_EXAMPLE_TCP_SEND_TIMEOUT_MS / 1000,
        .tv_usec = (CONFIG_EXAMPLE_TCP_SEND_TIMEOUT_MS % 1000) * 1000,
    };
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes));
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    struct sockaddr_in dest = {
        .sin_family = AF_INET,
        .sin_port = htons(CONFIG_EXAMPLE_JETSON_PORT),
    };
    if (inet_pton(AF_INET, CONFIG_EXAMPLE_JETSON_IP, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "invalid Jetson IP: %s", CONFIG_EXAMPLE_JETSON_IP);
        close(sock);
        return -1;
    }
    if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
        close(sock);
        return -1;
    }
    return sock;
}

static void capture_task(void *arg)
{
    stream_ctx_t *ctx = (stream_ctx_t *)arg;
    s_capture_running = true;

    while (!s_stop_requested) {
        encoded_frame_t frame;
        esp_err_t ret = video_fb_get(ctx, &frame);
        if (ret == ESP_ERR_INVALID_RESPONSE) {
            continue;
        }
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "capture/encode failed");
            s_stop_requested = true;
            break;
        }

        ++s_encoded_frames;
        while (xQueueSend(s_encoded_queue, &frame, pdMS_TO_TICKS(50)) != pdTRUE) {
            if (s_stop_requested) {
                video_fb_return(ctx, frame.index);
                goto done;
            }
        }
    }

done:
    s_capture_running = false;
    xTaskNotifyGive(s_stream_task_handle);
    vTaskDelete(NULL);
}

static void drain_encoded_queue(stream_ctx_t *ctx)
{
    encoded_frame_t frame;
    while (xQueueReceive(s_encoded_queue, &frame, 0) == pdTRUE) {
        video_fb_return(ctx, frame.index);
    }
}

static void run_connection(stream_ctx_t *ctx, int sock)
{
    xQueueReset(s_encoded_queue);
    s_encoded_frames = 0;
    if (video_start(ctx) != ESP_OK) {
        s_stop_requested = true;
        return;
    }

    s_capture_running = true;
    if (xTaskCreate(capture_task, "p4_capture", 4096, ctx, 6, NULL) != pdPASS) {
        ESP_LOGE(TAG, "create capture task failed");
        s_capture_running = false;
        video_stop(ctx);
        s_stop_requested = true;
        return;
    }

    uint32_t sent_frames = 0;
    uint64_t sent_bytes = 0;
    int64_t stats_start_us = esp_timer_get_time();

    while (!s_stop_requested) {
        encoded_frame_t frame;
        if (xQueueReceive(s_encoded_queue, &frame, pdMS_TO_TICKS(100)) != pdTRUE) {
            continue;
        }

        esp_err_t ret = send_frame(sock, ctx->encoded_buffer[frame.index], frame.len);
        int64_t sent_us = esp_timer_get_time();
        video_fb_return(ctx, frame.index);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "TCP send failed; restarting stream");
            s_stop_requested = true;
            break;
        }

        ++sent_frames;
        sent_bytes += frame.len;
        int64_t elapsed_us = sent_us - stats_start_us;
        if (elapsed_us >= (int64_t)CONFIG_EXAMPLE_STATS_INTERVAL_SEC * 1000000) {
            double seconds = (double)elapsed_us / 1000000.0;
            double fps = (double)sent_frames / seconds;
            double mbps = ((double)sent_bytes * 8.0) / seconds / 1000000.0;
            double age_ms = (double)(sent_us - frame.encoded_us) / 1000.0;
            ESP_LOGI(TAG, "stats: encoded=%" PRIu32 " sent=%" PRIu32
                     " fps=%.1f bitrate=%.2fMbps queue=%u age=%.1fms",
                     s_encoded_frames, sent_frames, fps, mbps,
                     (unsigned)uxQueueMessagesWaiting(s_encoded_queue), age_ms);
            sent_frames = 0;
            sent_bytes = 0;
            s_encoded_frames = 0;
            stats_start_us = sent_us;
        }
    }

    shutdown(sock, SHUT_RDWR);
    while (s_capture_running) {
        /* Return queued encoder buffers so a DQBUF blocked in capture can finish. */
        drain_encoded_queue(ctx);
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(50));
    }
    drain_encoded_queue(ctx);
    video_stop(ctx);
}

static void stream_task(void *arg)
{
    stream_ctx_t *ctx = (stream_ctx_t *)arg;
    s_stream_task_handle = xTaskGetCurrentTaskHandle();

    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        while (s_streaming && !s_stop_requested) {
            int sock = connect_jetson();
            if (sock < 0) {
                ESP_LOGW(TAG, "connect to %s:%d failed; retrying",
                         CONFIG_EXAMPLE_JETSON_IP, CONFIG_EXAMPLE_JETSON_PORT);
                vTaskDelay(pdMS_TO_TICKS(CONFIG_EXAMPLE_RECONNECT_DELAY_MS));
                continue;
            }

            s_sock = sock;
            ESP_LOGI(TAG, "connected to Jetson at %s:%d",
                     CONFIG_EXAMPLE_JETSON_IP, CONFIG_EXAMPLE_JETSON_PORT);
            run_connection(ctx, sock);
            close(sock);
            s_sock = -1;

            if (s_streaming && s_stop_requested) {
                /* A clean restart ensures the receiver gets fresh SPS/PPS/IDR. */
                s_stop_requested = false;
                vTaskDelay(pdMS_TO_TICKS(CONFIG_EXAMPLE_RECONNECT_DELAY_MS));
            }
        }

        s_streaming = false;
        s_stop_requested = false;
        ESP_LOGI(TAG, "stream stopped");
    }
}

static void button_task(void *arg)
{
    TaskHandle_t stream_handle = (TaskHandle_t)arg;
    const int gpio = CONFIG_EXAMPLE_STREAM_BUTTON_GPIO;
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << gpio),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io));

    TickType_t last_toggle = 0;
    int last_level = gpio_get_level(gpio);
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(25));
        int level = gpio_get_level(gpio);
        if (level == last_level) {
            continue;
        }
        last_level = level;

        TickType_t now = xTaskGetTickCount();
        if ((now - last_toggle) < pdMS_TO_TICKS(BUTTON_DEBOUNCE_MS)) {
            continue;
        }
        last_toggle = now;

        if (level == 0) {
            if (!s_streaming) {
                s_stop_requested = false;
                s_streaming = true;
                xTaskNotifyGive(stream_handle);
            } else {
                s_streaming = false;
                s_stop_requested = true;
                int sock = s_sock;
                if (sock >= 0) {
                    shutdown(sock, SHUT_RDWR);
                }
            }
        }
    }
}

void app_main(void)
{
    esp_err_t nvs_ret = nvs_flash_init();
    if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES || nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    } else {
        ESP_ERROR_CHECK(nvs_ret);
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* Camera clock/init should happen early for MIPI sensors. */
    ESP_ERROR_CHECK(example_video_init());
    ESP_ERROR_CHECK(init_capture(&s_ctx));
    ESP_ERROR_CHECK(init_codec(&s_ctx));
    ESP_ERROR_CHECK(configure_video(&s_ctx));

    ESP_ERROR_CHECK(example_connect());

    s_encoded_queue = xQueueCreate(ENCODED_BUFFER_COUNT, sizeof(encoded_frame_t));
    if (s_encoded_queue == NULL) {
        ESP_LOGE(TAG, "create encoded queue failed");
        abort();
    }

    TaskHandle_t stream_handle = NULL;
    if (xTaskCreate(stream_task, "p4_stream", 6144, &s_ctx, 5, &stream_handle) != pdPASS ||
        xTaskCreate(button_task, "stream_button", 2048, stream_handle, 5, NULL) != pdPASS) {
        ESP_LOGE(TAG, "create control tasks failed");
        abort();
    }

    ESP_LOGI(TAG, "ready: press GPIO %d button to stream %dx%d@%d H.264 to %s:%d",
             CONFIG_EXAMPLE_STREAM_BUTTON_GPIO,
             CONFIG_EXAMPLE_FRAME_WIDTH, CONFIG_EXAMPLE_FRAME_HEIGHT,
             CONFIG_EXAMPLE_FRAME_FPS, CONFIG_EXAMPLE_JETSON_IP,
             CONFIG_EXAMPLE_JETSON_PORT);
}
