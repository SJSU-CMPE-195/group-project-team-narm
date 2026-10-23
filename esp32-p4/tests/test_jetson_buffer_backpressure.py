"""Exercise actual producer/requeue functions against a stalled TCP sender.

GCC is required. The mock encoder rejects a third encode immediately when both
MMAP output buffers are held, matching esp_video's M2M driver behavior.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_jetson_stream_control import firmware_function


HARNESS = r"""
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

typedef int esp_err_t;
typedef struct { uint32_t index, len; int64_t encoded_us; } encoded_frame_t;
typedef struct { int m2m_fd; uint32_t encoded_buffer_count; } stream_ctx_t;
#define ESP_OK 0
#define ESP_FAIL -1
#define ESP_ERR_INVALID_RESPONSE -2
#define V4L2_BUF_TYPE_VIDEO_CAPTURE 1
#define pdTRUE 1
#define pdMS_TO_TICKS(ms) (ms)
#define ESP_LOGE(...) ((void)0)

static volatile bool s_stop_requested, s_capture_running;
static volatile uint32_t s_encoded_frames;
static int s_encoded_queue, s_available_buffers, s_stream_task_handle;
static unsigned credits, queued, encoded, waits, notifications, deletions;
static bool driver_queued[2], fail_requeue;
static encoded_frame_t frames[2];
static stream_ctx_t context = { .encoded_buffer_count = 2 };

static void video_fb_return(stream_ctx_t *ctx, uint32_t index);
static int queue_mmap_buffer(int fd, int type, uint32_t index) {
  (void)fd; (void)type;
  if (fail_requeue) return ESP_FAIL;
  assert(index < 2 && !driver_queued[index]);
  driver_queued[index] = true;
  return ESP_OK;
}
static int xSemaphoreGive(int semaphore) {
  (void)semaphore;
  assert(credits < 2);
  ++credits;
  return pdTRUE;
}
static int xSemaphoreTake(int semaphore, unsigned timeout) {
  (void)semaphore;
  if (credits == 0 && timeout) {
    assert(queued == 2);
    assert(!driver_queued[0] && !driver_queued[1]);
    ++waits;
    if (waits == 1) {
      /* TCP finally finishes the oldest frame after both buffers filled. */
      assert(encoded == 2);
      video_fb_return(&context, frames[0].index);
      frames[0] = frames[1];
      --queued;
    } else {
      /* Stop while the sender is stalled again: producer must exit. */
      s_stop_requested = true;
      return 0;
    }
  }
  if (credits == 0) return 0;
  --credits;
  return pdTRUE;
}
static esp_err_t video_fb_get(stream_ctx_t *ctx, encoded_frame_t *frame) {
  (void)ctx;
  assert((driver_queued[0] || driver_queued[1]) &&
         "M2M encoding attempted with no queued output buffer");
  frame->index = driver_queued[0] ? 0 : 1;
  driver_queued[frame->index] = false;
  frame->len = 64;
  frame->encoded_us = 0;
  ++encoded;
  return ESP_OK;
}
static int xQueueSend(int queue, const encoded_frame_t *frame, unsigned ticks) {
  (void)queue; (void)ticks;
  assert(queued < 2);
  frames[queued++] = *frame;
  return pdTRUE;
}
static void xTaskNotifyGive(int task) { (void)task; ++notifications; }
static void vTaskDelete(void *task) { (void)task; ++deletions; }
"""

CHECKS = r"""
int main(void) {
  /* Keep the same harness compilable against the producer before credits. */
  (void)xSemaphoreTake; (void)xSemaphoreGive; (void)s_available_buffers;
  /* Repeat a session to catch leaked credits across a stop/restart. */
  for (unsigned session = 0; session < 2; ++session) {
    s_stop_requested = false;
    s_capture_running = false;
    s_encoded_frames = queued = encoded = waits = 0;
    credits = 2;
    driver_queued[0] = driver_queued[1] = true;
    capture_task(&context);
    assert(!s_capture_running && s_stop_requested);
    assert(waits == 2 && encoded == 3 && s_encoded_frames == 3);
    assert(queued == 2 && credits == 0);
    for (unsigned i = 0; i < queued; ++i)
      video_fb_return(&context, frames[i].index);
    assert(credits == 2 && driver_queued[0] && driver_queued[1]);
  }
  assert(notifications == 2 && deletions == 2);

  /* A failed driver requeue must never advertise a free encoder buffer. */
  credits = 0;
  driver_queued[0] = driver_queued[1] = false;
  fail_requeue = true;
  s_stop_requested = false;
  video_fb_return(&context, 0);
  assert(credits == 0 && s_stop_requested && !driver_queued[0]);
  return 0;
}
"""


def run_harness(source):
    actual_functions = "\n".join(
        firmware_function(source, name) for name in ("video_fb_return", "capture_task")
    )
    with tempfile.TemporaryDirectory() as directory:
        harness = Path(directory) / "backpressure.c"
        executable = Path(directory) / ("backpressure.exe" if os.name == "nt" else "backpressure")
        harness.write_text(HARNESS + actual_functions + CHECKS)
        compilation = subprocess.run(
            [shutil.which("gcc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
             str(harness), "-o", str(executable)],
            capture_output=True, text=True, timeout=30,
        )
        if compilation.returncode:
            raise AssertionError(compilation.stderr)
        return subprocess.run(
            [str(executable)], capture_output=True, text=True, timeout=5
        )


class JetsonBufferBackpressureTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gcc"), "requires GCC on PATH")
    def test_slow_sender_stop_restart_and_failed_requeue(self):
        source = (Path(__file__).parents[1] / "main/h264_stream_main.c").read_text()
        result = run_harness(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
