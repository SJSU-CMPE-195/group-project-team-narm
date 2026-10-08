/* Native YUV420 capture -> ESP32-P4 hardware H.264, without CPU resizing.
 * Encoded buffers stay owned by the caller until h264_camera_return(). */
#include "h264_camera.h"

#include <fcntl.h>
#include <inttypes.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include "sdkconfig.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "example_video_common.h"
#include "esp_video_device.h"
#include "esp_video_ioctl.h"
#include "linux/videodev2.h"

#if CONFIG_EXAMPLE_H264_MAX_QP <= CONFIG_EXAMPLE_H264_MIN_QP
#error "H.264 maximum QP must be greater than minimum QP"
#endif

static const char *TAG = "browser_h264_camera";

static esp_err_t queue_buffer(int fd, uint32_t type, uint32_t index)
{
    struct v4l2_buffer buffer = {
        .type = type, .memory = V4L2_MEMORY_MMAP, .index = index,
    };
    return ioctl(fd, VIDIOC_QBUF, &buffer) == 0 ? ESP_OK : ESP_FAIL;
}

static esp_err_t set_control(int fd, uint32_t id, int32_t value)
{
    struct v4l2_ext_control control = { .id = id, .value = value };
    struct v4l2_ext_controls controls = {
        .ctrl_class = V4L2_CID_CODEC_CLASS, .count = 1, .controls = &control,
    };
    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &controls) != 0) {
        ESP_LOGE(TAG, "H.264 control %" PRIu32 "=%" PRId32 " failed", id, value);
        return ESP_FAIL;
    }
    return ESP_OK;
}

static esp_err_t map_buffers(int fd, uint32_t count, uint8_t **data, size_t *sizes)
{
    for (uint32_t i = 0; i < count; ++i) {
        struct v4l2_buffer buffer = {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP, .index = i,
        };
        if (ioctl(fd, VIDIOC_QUERYBUF, &buffer) != 0 || buffer.length == 0) {
            return ESP_FAIL;
        }
        sizes[i] = buffer.length;
        void *mapped = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, buffer.m.offset);
        if (mapped == MAP_FAILED || mapped == NULL) {
            return ESP_FAIL;
        }
        data[i] = mapped;
    }
    return ESP_OK;
}

void h264_camera_stop(h264_camera_t *camera)
{
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (camera->camera_started) {
        ioctl(camera->camera_fd, VIDIOC_STREAMOFF, &type);
        camera->camera_started = false;
    }
    type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (camera->encoder_input_started) {
        ioctl(camera->encoder_fd, VIDIOC_STREAMOFF, &type);
        camera->encoder_input_started = false;
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (camera->encoder_output_started) {
        ioctl(camera->encoder_fd, VIDIOC_STREAMOFF, &type);
        camera->encoder_output_started = false;
    }
}

void h264_camera_close(h264_camera_t *camera)
{
    h264_camera_stop(camera);
    for (uint32_t i = 0; i < camera->camera_count; ++i) {
        if (camera->camera_data[i]) {
            munmap(camera->camera_data[i], camera->camera_size[i]);
            camera->camera_data[i] = NULL;
        }
    }
    for (uint32_t i = 0; i < camera->encoded_count; ++i) {
        if (camera->encoded_data[i]) {
            munmap(camera->encoded_data[i], camera->encoded_size[i]);
            camera->encoded_data[i] = NULL;
        }
    }
    if (camera->encoder_fd >= 0) {
        close(camera->encoder_fd);
        camera->encoder_fd = -1;
    }
    if (camera->camera_fd >= 0) {
        close(camera->camera_fd);
        camera->camera_fd = -1;
    }
}

esp_err_t h264_camera_open(h264_camera_t *camera)
{
    memset(camera, 0, sizeof(*camera));
    camera->camera_fd = camera->encoder_fd = -1;
    camera->camera_fd = open(EXAMPLE_CAM_DEV_PATH, O_RDWR);
    camera->encoder_fd = open(ESP_VIDEO_H264_DEVICE_NAME, O_RDWR);
    if (camera->camera_fd < 0 || camera->encoder_fd < 0) {
        ESP_LOGE(TAG, "could not open camera/H.264 hardware devices");
        goto fail;
    }

    /* Read the sensor's active mode instead of requesting unsupported 720p.
     * MJPEG output size settings deliberately do not apply to this mode. */
    struct v4l2_format format = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };
    if (ioctl(camera->camera_fd, VIDIOC_G_FMT, &format) != 0) {
        goto fail;
    }
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (ioctl(camera->camera_fd, VIDIOC_S_FMT, &format) != 0 ||
        ioctl(camera->camera_fd, VIDIOC_G_FMT, &format) != 0 ||
        format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUV420 ||
        !format.fmt.pix.width || !format.fmt.pix.height) {
        ESP_LOGE(TAG, "native YUV420 camera format is unavailable");
        goto fail;
    }
    camera->width = format.fmt.pix.width;
    camera->height = format.fmt.pix.height;
    if (camera->width > UINT16_MAX || camera->height > UINT16_MAX) {
        goto fail;
    }

    struct v4l2_streamparm parameters = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .parm.capture = {
            .capability = V4L2_CAP_TIMEPERFRAME,
            .timeperframe = { .numerator = 1, .denominator = CONFIG_EXAMPLE_FRAME_FPS },
        },
    };
    if (ioctl(camera->camera_fd, VIDIOC_S_PARM, &parameters) != 0) {
        ESP_LOGW(TAG, "using the sensor's default frame rate");
    }
    memset(&parameters, 0, sizeof(parameters));
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    camera->fps = CONFIG_EXAMPLE_FRAME_FPS;
    if (ioctl(camera->camera_fd, VIDIOC_G_PARM, &parameters) == 0 &&
        parameters.parm.capture.timeperframe.numerator) {
        uint32_t fps = parameters.parm.capture.timeperframe.denominator /
                       parameters.parm.capture.timeperframe.numerator;
        if (fps) {
            camera->fps = fps;
        }
    }

    if (set_control(camera->encoder_fd, V4L2_CID_MPEG_VIDEO_H264_I_PERIOD,
                    CONFIG_EXAMPLE_H264_I_PERIOD) != ESP_OK ||
        set_control(camera->encoder_fd, V4L2_CID_MPEG_VIDEO_BITRATE,
                    CONFIG_EXAMPLE_H264_BITRATE) != ESP_OK ||
        set_control(camera->encoder_fd, V4L2_CID_MPEG_VIDEO_H264_MIN_QP,
                    CONFIG_EXAMPLE_H264_MIN_QP) != ESP_OK ||
        set_control(camera->encoder_fd, V4L2_CID_MPEG_VIDEO_H264_MAX_QP,
                    CONFIG_EXAMPLE_H264_MAX_QP) != ESP_OK) {
        goto fail;
    }

    struct v4l2_requestbuffers buffers = {
        .count = CONFIG_EXAMPLE_CAMERA_VIDEO_BUFFER_NUMBER,
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP,
    };
    if (ioctl(camera->camera_fd, VIDIOC_REQBUFS, &buffers) != 0 ||
        buffers.count < 2 || buffers.count > H264_CAMERA_BUFFERS) {
        goto fail;
    }
    camera->camera_count = buffers.count;
    if (map_buffers(camera->camera_fd, camera->camera_count,
                    camera->camera_data, camera->camera_size) != ESP_OK) {
        goto fail;
    }

    format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(camera->encoder_fd, VIDIOC_S_FMT, &format) != 0) {
        goto fail;
    }
    buffers = (struct v4l2_requestbuffers) {
        .count = 1, .type = V4L2_BUF_TYPE_VIDEO_OUTPUT, .memory = V4L2_MEMORY_USERPTR,
    };
    if (ioctl(camera->encoder_fd, VIDIOC_REQBUFS, &buffers) != 0 || buffers.count < 1) {
        goto fail;
    }
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_H264;
    if (ioctl(camera->encoder_fd, VIDIOC_S_FMT, &format) != 0) {
        goto fail;
    }
    buffers = (struct v4l2_requestbuffers) {
        .count = H264_ENCODED_BUFFERS,
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP,
    };
    if (ioctl(camera->encoder_fd, VIDIOC_REQBUFS, &buffers) != 0 ||
        buffers.count < 2 || buffers.count > H264_ENCODED_BUFFERS) {
        goto fail;
    }
    camera->encoded_count = buffers.count;
    if (map_buffers(camera->encoder_fd, camera->encoded_count,
                    camera->encoded_data, camera->encoded_size) != ESP_OK) {
        goto fail;
    }

    /* Keep a failed/stopped session from leaving the producer in an
     * unbounded camera/encoder dequeue. */
    struct timeval timeout = { .tv_sec = 1, .tv_usec = 0 };
    if (ioctl(camera->camera_fd, VIDIOC_S_DQBUF_TIMEOUT, &timeout) != 0 ||
        ioctl(camera->encoder_fd, VIDIOC_S_DQBUF_TIMEOUT, &timeout) != 0) {
        goto fail;
    }
    ESP_LOGI(TAG, "native capture: %" PRIu32 "x%" PRIu32 " @ %" PRIu32
             " fps, hardware H.264 target=%d bps, resize=none",
             camera->width, camera->height, camera->fps, CONFIG_EXAMPLE_H264_BITRATE);
    return ESP_OK;

fail:
    ESP_LOGE(TAG, "camera/H.264 configuration failed");
    h264_camera_close(camera);
    return ESP_FAIL;
}

esp_err_t h264_camera_start(h264_camera_t *camera)
{
    for (uint32_t i = 0; i < camera->camera_count; ++i) {
        if (queue_buffer(camera->camera_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, i) != ESP_OK) {
            goto fail;
        }
    }
    for (uint32_t i = 0; i < camera->encoded_count; ++i) {
        if (queue_buffer(camera->encoder_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, i) != ESP_OK) {
            goto fail;
        }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->encoder_fd, VIDIOC_STREAMON, &type) != 0) {
        goto fail;
    }
    camera->encoder_output_started = true;
    type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    if (ioctl(camera->encoder_fd, VIDIOC_STREAMON, &type) != 0) {
        goto fail;
    }
    camera->encoder_input_started = true;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(camera->camera_fd, VIDIOC_STREAMON, &type) != 0) {
        goto fail;
    }
    camera->camera_started = true;
    return ESP_OK;

fail:
    h264_camera_stop(camera);
    return ESP_FAIL;
}

esp_err_t h264_camera_next(h264_camera_t *camera, h264_camera_frame_t *frame)
{
    struct v4l2_buffer captured = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP,
    };
    if (ioctl(camera->camera_fd, VIDIOC_DQBUF, &captured) != 0) {
        return ESP_FAIL;
    }
    frame->captured_us = esp_timer_get_time();
    if (captured.index >= camera->camera_count || !captured.bytesused ||
        captured.bytesused > camera->camera_size[captured.index] ||
        (captured.flags & V4L2_BUF_FLAG_ERROR)) {
        ioctl(camera->camera_fd, VIDIOC_QBUF, &captured);
        return ESP_FAIL;
    }
    struct v4l2_buffer input = {
        .index = 0, .type = V4L2_BUF_TYPE_VIDEO_OUTPUT,
        .memory = V4L2_MEMORY_USERPTR,
        .m.userptr = (unsigned long)camera->camera_data[captured.index],
        .length = captured.bytesused, .bytesused = captured.bytesused,
    };
    if (ioctl(camera->encoder_fd, VIDIOC_QBUF, &input) != 0) {
        ioctl(camera->camera_fd, VIDIOC_QBUF, &captured);
        return ESP_FAIL;
    }
    struct v4l2_buffer encoded = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP,
    };
    if (ioctl(camera->encoder_fd, VIDIOC_DQBUF, &encoded) != 0) {
        /* Stop the whole pipeline before releasing camera storage still
         * potentially owned by the encoder. The caller ends the session. */
        return ESP_FAIL;
    }
    if (ioctl(camera->encoder_fd, VIDIOC_DQBUF, &input) != 0) {
        return ESP_FAIL;
    }
    if (ioctl(camera->camera_fd, VIDIOC_QBUF, &captured) != 0 ||
        encoded.index >= camera->encoded_count || !encoded.bytesused ||
        encoded.bytesused > camera->encoded_size[encoded.index] ||
        (encoded.flags & V4L2_BUF_FLAG_ERROR)) {
        return ESP_FAIL;
    }
    frame->index = encoded.index;
    frame->size = encoded.bytesused;
    frame->encoded_us = esp_timer_get_time();
    return ESP_OK;
}

esp_err_t h264_camera_return(h264_camera_t *camera, uint32_t index)
{
    if (index >= camera->encoded_count) {
        return ESP_ERR_INVALID_ARG;
    }
    return queue_buffer(camera->encoder_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE, index);
}
