#pragma once

#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define H264_CAMERA_BUFFERS 4
#define H264_ENCODED_BUFFERS 2

typedef struct {
  int camera_fd;
  int encoder_fd;
  uint32_t width, height, fps;
  uint32_t camera_count, encoded_count;
  uint8_t *camera_data[H264_CAMERA_BUFFERS];
  size_t camera_size[H264_CAMERA_BUFFERS];
  uint8_t *encoded_data[H264_ENCODED_BUFFERS];
  size_t encoded_size[H264_ENCODED_BUFFERS];
  bool camera_started, encoder_input_started, encoder_output_started;
} h264_camera_t;

typedef struct {
  uint32_t index, size;
  int64_t captured_us, encoded_us;
} h264_camera_frame_t;

esp_err_t h264_camera_open(h264_camera_t *camera);
esp_err_t h264_camera_start(h264_camera_t *camera);
void h264_camera_stop(h264_camera_t *camera);
void h264_camera_close(h264_camera_t *camera);
esp_err_t h264_camera_next(h264_camera_t *camera, h264_camera_frame_t *frame);
esp_err_t h264_camera_return(h264_camera_t *camera, uint32_t index);
