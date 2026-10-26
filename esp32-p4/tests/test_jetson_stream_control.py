"""Run the firmware's actual control functions with deterministic host events.

Requires GCC on PATH; no ESP-IDF installation or connected board is needed.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def firmware_function(source, name):
    start = source.index(f"static void {name}(")
    opening = source.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


HARNESS = r"""
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <setjmp.h>
#include <stdlib.h>

typedef void *TaskHandle_t;
typedef unsigned TickType_t;
typedef struct { int unused; } stream_ctx_t;
typedef struct {
  uint64_t pin_bit_mask;
  int mode, pull_up_en, pull_down_en, intr_type;
} gpio_config_t;

#define CONFIG_EXAMPLE_RECONNECT_DELAY_MS 1000
#define CONFIG_EXAMPLE_JETSON_IP "10.42.0.1"
#define CONFIG_EXAMPLE_JETSON_PORT 5000
#define CONFIG_EXAMPLE_STREAM_BUTTON_GPIO 35
#define BUTTON_DEBOUNCE_MS 200
#define GPIO_IS_VALID_GPIO(pin) ((pin) >= 0 && (pin) < 55)
#define GPIO_MODE_INPUT 0
#define GPIO_PULLUP_ENABLE 1
#define GPIO_PULLDOWN_DISABLE 0
#define GPIO_INTR_DISABLE 0
#define SHUT_RDWR 2
#define pdTRUE 1
#define portMAX_DELAY 0xffffffffU
#define pdMS_TO_TICKS(ms) (ms)
#define ESP_ERROR_CHECK(result) assert((result) == 0)
#define ESP_LOGI(...) ((void)0)
#define ESP_LOGE(...) ((void)0)
#define ESP_LOGW(...) (++warnings)

enum { BUTTON_EVENTS, RESTART_DURING_CLEANUP, CANCEL_THEN_RESTART,
       CANCEL_AND_STOP };
static int scenario, connects, sessions, waits, warnings, notifications;
static int shutdowns;
static TickType_t ticks;
static jmp_buf finished;
static volatile bool s_streaming, s_stop_requested;
static volatile int s_sock = -1;
static TaskHandle_t s_stream_task_handle;

static TaskHandle_t xTaskGetCurrentTaskHandle(void) { return (void *)1; }
static void xTaskNotifyGive(TaskHandle_t task) { (void)task; ++notifications; }
static unsigned ulTaskNotifyTake(int clear, unsigned timeout) {
  (void)clear; (void)timeout;
  if (++waits > 1) longjmp(finished, 1);
  return 1;
}
static void vTaskDelay(unsigned delay) {
  if (scenario == BUTTON_EVENTS) {
    ticks += delay;
    if (ticks >= 625) longjmp(finished, 1);
  }
}
static TickType_t xTaskGetTickCount(void) { return ticks; }
static int gpio_config(const gpio_config_t *config) {
  assert(config->pin_bit_mask == (1ULL << 35));
  assert(config->pull_up_en == GPIO_PULLUP_ENABLE);
  return 0;
}
static int gpio_get_level(int gpio) {
  (void)gpio;
  /* Two active-low presses, with enough release time for debouncing. */
  return (ticks >= 250 && ticks < 300) ||
         (ticks >= 550 && ticks < 600) ? 0 : 1;
}
static int shutdown(int sock, int how) {
  (void)sock; (void)how; ++shutdowns; return 0;
}
static int close(int sock) { (void)sock; return 0; }
static int connect_jetson(void) {
  assert(s_streaming && !s_stop_requested);
  ++connects;
  if (connects == 1 &&
      (scenario == CANCEL_THEN_RESTART || scenario == CANCEL_AND_STOP)) {
    s_streaming = false;
    s_stop_requested = true;
    if (scenario == CANCEL_THEN_RESTART) s_streaming = true;
    return -1;
  }
  return 42;
}
static void run_connection(stream_ctx_t *ctx, int sock) {
  (void)ctx; (void)sock;
  assert(s_streaming && !s_stop_requested);
  ++sessions;
  s_streaming = false;
  s_stop_requested = true;
  if (scenario == RESTART_DURING_CLEANUP && sessions == 1) {
    /* Third press arrives before the old capture task finishes stopping. */
    s_streaming = true;
    assert(s_stop_requested);
  }
}
"""

CHECKS = r"""
int main(void) {
  scenario = BUTTON_EVENTS;
  s_streaming = true;
  s_stop_requested = false;
  s_sock = 42;
  if (setjmp(finished) == 0) button_task((void *)1);
  assert(s_streaming && s_stop_requested);
  assert(shutdowns == 1 && notifications == 1);

  for (scenario = RESTART_DURING_CLEANUP;
       scenario <= CANCEL_AND_STOP; ++scenario) {
    connects = sessions = waits = warnings = 0;
    s_streaming = true;
    s_stop_requested = false;
    if (setjmp(finished) == 0) stream_task(0);
    assert(!s_streaming && s_stop_requested);
    assert(warnings == 0);
    assert(connects == (scenario == CANCEL_AND_STOP ? 1 : 2));
    assert(sessions == (scenario == RESTART_DURING_CLEANUP ? 2 :
                        scenario == CANCEL_THEN_RESTART ? 1 : 0));
  }
  return 0;
}
"""


class JetsonStreamControlTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gcc"), "requires GCC on PATH")
    def test_button_restart_and_cancelled_connect(self):
        source = (Path(__file__).parents[1] / "main/h264_stream_main.c").read_text()
        control = "\n".join(
            firmware_function(source, name) for name in ("stream_task", "button_task")
        )
        with tempfile.TemporaryDirectory() as directory:
            harness = Path(directory) / "control.c"
            executable = Path(directory) / ("control.exe" if os.name == "nt" else "control")
            harness.write_text(HARNESS + control + CHECKS)
            compilation = subprocess.run(
                [shutil.which("gcc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                 str(harness), "-o", str(executable)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compilation.returncode, 0, compilation.stderr)
            subprocess.run([str(executable)], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
