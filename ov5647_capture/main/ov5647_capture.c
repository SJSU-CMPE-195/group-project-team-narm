/*
 * Camera capture for Seeed Studio XIAO ESP32-S3 Sense with OV3660 (DVP +
 * SCCB). esp32-camera probes the sensor and loads the OV3660 driver.
 *
 * Other Sense modules (OV2640, OV5640) use the same Seeed pinout; swap only
 * the hardware. This replaces the old ESP32-P4 + MIPI CSI-2 OV5647 stack.
 */
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include <stdio.h>
#include <string.h>

// Default: STA Wi‑Fi + TCP JPEG to Jetson ([len:u32be][jpeg] — see jetson-code/).
#define ENABLE_TCP_STREAM 1
// Optional: SoftAP browser MJPEG at http://192.168.4.1/stream (mutually exclusive with TCP).
#define ENABLE_WEB_PREVIEW 0

#if ENABLE_TCP_STREAM && ENABLE_WEB_PREVIEW
#error "Enable only one: ENABLE_TCP_STREAM or ENABLE_WEB_PREVIEW"
#endif

#if ENABLE_TCP_STREAM || ENABLE_WEB_PREVIEW
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#endif
#if ENABLE_TCP_STREAM
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include <unistd.h>
#endif
#if ENABLE_WEB_PREVIEW
#include "esp_http_server.h"
#endif

static const char *TAG = "ov5647_capture";

#if ENABLE_WEB_PREVIEW
// Phone/laptop preview: keep bitrate low and FPS stable.
#define CAM_WIDTH 320
#define CAM_HEIGHT 240
#define CAM_FRAME_SIZE FRAMESIZE_QVGA
#define CAM_JPEG_QUALITY 25
#define CAM_FPS 15
#else
// Jetson ingest: higher FPS and resolution are okay.
#define CAM_WIDTH 640
#define CAM_HEIGHT 480
#define CAM_FRAME_SIZE FRAMESIZE_VGA
#define CAM_JPEG_QUALITY MJPEG_ESP_CAM_JPEG_QUALITY
#define CAM_FPS MJPEG_FPS
#endif

/* Seeed XIAO ESP32-S3 Sense — OV3660 on the expansion FPC (DVP + SCCB). */
#define XIAO_PIN_PWDN -1
#define XIAO_PIN_RESET -1
#define XIAO_PIN_XCLK 10
#define XIAO_PIN_SIOD 40
#define XIAO_PIN_SIOC 39
#define XIAO_PIN_D7 48
#define XIAO_PIN_D6 11
#define XIAO_PIN_D5 12
#define XIAO_PIN_D4 14
#define XIAO_PIN_D3 16
#define XIAO_PIN_D2 18
#define XIAO_PIN_D1 17
#define XIAO_PIN_D0 15
#define XIAO_PIN_VSYNC 38
#define XIAO_PIN_HREF 47
#define XIAO_PIN_PCLK 13

/* Jetson tcp_ingest_server.py / main.py (listen on 0.0.0.0:5000 by default).
 * Hotspot gateway is often 10.42.0.1 — confirm with `ip -4 addr` on the Jetson. */
#define JETSON_TCP_IP "10.42.0.1"
#define JETSON_TCP_PORT 5000

/* Target capture/send FPS cap (~12–15 is stable over Wi‑Fi for VGA JPEG). */
#define MJPEG_FPS 15
/* esp32-camera: 0–63, lower value = higher quality / larger JPEG (more Mbps). */
#define MJPEG_ESP_CAM_JPEG_QUALITY 26

#if ENABLE_TCP_STREAM
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1
#define WIFI_CONNECTED_BITS (WIFI_CONNECTED_BIT | WIFI_FAIL_BIT)

static EventGroupHandle_t s_wifi_event_group;
static int s_wifi_retry_num = 0;
static int s_tcp_sock = -1;

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  (void)arg;
  (void)event_data;

  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    if (s_wifi_retry_num < CONFIG_ESP_WIFI_REMOTE_MAX_RETRIES) {
      esp_wifi_connect();
      s_wifi_retry_num++;
      ESP_LOGI(TAG, "Wi-Fi retry connect (attempt %d)", s_wifi_retry_num);
    } else {
      xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
    }
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    s_wifi_retry_num = 0;
    xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

static void wifi_init_sta(void) {
  s_wifi_event_group = xEventGroupCreate();

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  esp_event_handler_instance_t instance_any_id;
  esp_event_handler_instance_t instance_got_ip;
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL,
      &instance_any_id));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL,
      &instance_got_ip));

  wifi_config_t wifi_config = {
      .sta =
          {
              .ssid = CONFIG_ESP_WIFI_REMOTE_SSID,
              .password = CONFIG_ESP_WIFI_REMOTE_PASSWORD,
          },
  };

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());

  ESP_LOGI(TAG, "Waiting for Wi-Fi STA connection...");
  EventBits_t bits =
      xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BITS, pdFALSE,
                          pdFALSE, portMAX_DELAY);

  if (bits & WIFI_CONNECTED_BIT) {
    ESP_LOGI(TAG, "Wi-Fi connected (IP acquired).");
    /* Disable modem sleep while streaming — reduces latency/jitter vs default MIN_MODEM. */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
  } else if (bits & WIFI_FAIL_BIT) {
    ESP_LOGW(TAG, "Wi-Fi failed to connect after retries.");
  } else {
    ESP_LOGE(TAG, "Wi-Fi: unexpected event bits=0x%lx", (unsigned long)bits);
  }
}

static int tcp_send_all(int sock, const void *data, size_t len) {
  const uint8_t *p = (const uint8_t *)data;
  size_t sent_total = 0;
  while (sent_total < len) {
    int s = send(sock, p + sent_total, len - sent_total, 0);
    if (s <= 0) {
      return -1;
    }
    sent_total += (size_t)s;
  }
  return 0;
}

static int tcp_connect(const char *ip, uint16_t port) {
  int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
  if (sock < 0) {
    return -1;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &addr.sin_addr) <= 0) {
    close(sock);
    return -1;
  }

  if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
    close(sock);
    return -1;
  }

  // Reduce latency for small writes (length header + payload).
  int one = 1;
  (void)setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  int sndbuf = 128 * 1024;
  (void)setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

  return sock;
}
#endif // ENABLE_TCP_STREAM

#if ENABLE_WEB_PREVIEW
/* SoftAP: join from phone/PC, then open URL in a browser (Chrome / Edge work well). */
#define WEB_AP_SSID "XIAO-CAM"
#define WEB_AP_PASS "xiao3660" /* WPA2: min 8 characters */

static esp_err_t http_root_get(httpd_req_t *req) {
  static const char html[] =
      "<!DOCTYPE html><html><head><meta charset=utf-8><title>XIAO OV3660</title></head>"
      "<body><h1>XIAO ESP32-S3 + OV3660</h1>"
      "<p><b>Live view:</b> open <a href=\"/stream\" target=\"_blank\">/stream</a> in a "
      "new tab (works best), or use the iframe below.</p>"
      "<iframe src=\"/stream\" title=cam style=\"width:100%%;max-width:640px;height:480px;"
      "border:1px solid #444;background:#000\"></iframe>"
      "</body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

/* Raw HTTP write for multipart MJPEG. Do NOT use httpd_resp_send_chunk — it adds
 * Transfer-Encoding: chunked and many browsers show only the first frame. */
static int http_raw_send_all(httpd_req_t *req, const void *data, size_t len) {
  const char *p = (const char *)data;
  size_t left = len;
  while (left > 0) {
    int n = httpd_send(req, p, left);
    if (n <= 0) {
      return -1;
    }
    p += (size_t)n;
    left -= (size_t)n;
  }
  return 0;
}

static esp_err_t http_stream_get(httpd_req_t *req) {
  static const char resp_hdr[] =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Cache-Control: no-cache, no-store, must-revalidate\r\n"
      "Pragma: no-cache\r\n"
      "\r\n";
  if (http_raw_send_all(req, resp_hdr, sizeof(resp_hdr) - 1) != 0) {
    return ESP_FAIL;
  }

  char hdr[96];
  while (1) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    if (fb->len == 0) {
      esp_camera_fb_return(fb);
      continue;
    }
    int pl = snprintf(hdr, sizeof(hdr),
                      "\r\n--frame\r\nContent-Type: image/jpeg\r\n"
                      "Content-Length: %u\r\n\r\n",
                      (unsigned)fb->len);
    if (pl <= 0 || pl >= (int)sizeof(hdr)) {
      esp_camera_fb_return(fb);
      break;
    }
    if (http_raw_send_all(req, hdr, (size_t)pl) != 0) {
      esp_camera_fb_return(fb);
      break;
    }
    if (http_raw_send_all(req, fb->buf, fb->len) != 0) {
      esp_camera_fb_return(fb);
      break;
    }
    esp_camera_fb_return(fb);
    // Throttle the preview so the camera pipeline doesn't outpace the client.
    // This reduces cam_hal: FB-OVF when the browser stalls or Wi-Fi is weak.
    if (CAM_FPS > 0) {
      vTaskDelay(pdMS_TO_TICKS(1000 / CAM_FPS));
    }
  }
  return ESP_OK;
}

static void web_preview_start(void) {
  esp_err_t nvs_ret = nvs_flash_init();
  if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
  }

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_ap();

  wifi_init_config_t wcfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&wcfg));

  wifi_config_t ap = {0};
  strncpy((char *)ap.ap.ssid, WEB_AP_SSID, sizeof(ap.ap.ssid) - 1);
  strncpy((char *)ap.ap.password, WEB_AP_PASS, sizeof(ap.ap.password) - 1);
  ap.ap.ssid_len = (uint8_t)strlen(WEB_AP_SSID);
  ap.ap.channel = 1;
  ap.ap.authmode = WIFI_AUTH_WPA_WPA2_PSK;
  ap.ap.max_connection = 3;

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
  ESP_ERROR_CHECK(esp_wifi_start());

  httpd_config_t hcfg = HTTPD_DEFAULT_CONFIG();
  hcfg.server_port = 80;
  hcfg.ctrl_port = 32768;
  hcfg.stack_size = 8192;
  hcfg.max_open_sockets = 3;
  hcfg.send_wait_timeout = 30; /* seconds; long-lived MJPEG stream */

  httpd_handle_t server = NULL;
  ESP_ERROR_CHECK(httpd_start(&server, &hcfg));

  httpd_uri_t u_root = {.uri = "/",
                        .method = HTTP_GET,
                        .handler = http_root_get,
                        .user_ctx = NULL};
  httpd_uri_t u_stream = {.uri = "/stream",
                          .method = HTTP_GET,
                          .handler = http_stream_get,
                          .user_ctx = NULL};
  ESP_ERROR_CHECK(httpd_register_uri_handler(server, &u_root));
  ESP_ERROR_CHECK(httpd_register_uri_handler(server, &u_stream));

  ESP_LOGI(TAG,
           "Web preview: join Wi-Fi AP \"%s\" (password \"%s\"), open "
           "http://192.168.4.1/ or http://192.168.4.1/stream",
           WEB_AP_SSID, WEB_AP_PASS);
}
#endif // ENABLE_WEB_PREVIEW

void app_main(void) {
  camera_config_t config = {
      .pin_pwdn = XIAO_PIN_PWDN,
      .pin_reset = XIAO_PIN_RESET,
      .pin_xclk = XIAO_PIN_XCLK,
      .pin_sccb_sda = XIAO_PIN_SIOD,
      .pin_sccb_scl = XIAO_PIN_SIOC,
      .pin_d7 = XIAO_PIN_D7,
      .pin_d6 = XIAO_PIN_D6,
      .pin_d5 = XIAO_PIN_D5,
      .pin_d4 = XIAO_PIN_D4,
      .pin_d3 = XIAO_PIN_D3,
      .pin_d2 = XIAO_PIN_D2,
      .pin_d1 = XIAO_PIN_D1,
      .pin_d0 = XIAO_PIN_D0,
      .pin_vsync = XIAO_PIN_VSYNC,
      .pin_href = XIAO_PIN_HREF,
      .pin_pclk = XIAO_PIN_PCLK,

      .xclk_freq_hz = 20000000,
      .ledc_timer = LEDC_TIMER_0,
      .ledc_channel = LEDC_CHANNEL_0,

      .pixel_format = PIXFORMAT_JPEG,
      .frame_size = CAM_FRAME_SIZE,
      .jpeg_quality = CAM_JPEG_QUALITY,
      /* 3 buffers: reduces cam_hal FB-OVF when the app throttles with vTaskDelay
       * (sensor keeps filling DMA while we wait between esp_camera_fb_get calls). */
      .fb_count = 4,
      .fb_location = CAMERA_FB_IN_PSRAM,
      .grab_mode = CAMERA_GRAB_LATEST,
  };

  ESP_LOGI(TAG,
           "Camera: OV3660 @ %dx%d JPEG (XIAO ESP32-S3 Sense DVP pinout)",
           CAM_WIDTH, CAM_HEIGHT);

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG,
             "esp_camera_init failed (0x%x). Check OV3660 FPC, PSRAM, and pins.",
             (unsigned)err);
    while (1) {
      vTaskDelay(pdMS_TO_TICKS(3000));
    }
  }

  if (esp_camera_sensor_get()) {
    ESP_LOGI(TAG, "OV3660 sensor driver ready (esp32-camera)");
  }

#if ENABLE_TCP_STREAM
  esp_err_t nvs_ret = nvs_flash_init();
  if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
  }
  wifi_init_sta();

  ESP_LOGI(TAG, "Connecting to Jetson TCP...");
  s_tcp_sock = -1;
  while (s_tcp_sock < 0) {
    s_tcp_sock = tcp_connect(JETSON_TCP_IP, JETSON_TCP_PORT);
    if (s_tcp_sock < 0) {
      ESP_LOGW(TAG, "TCP connect failed, retrying...");
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }
  ESP_LOGI(TAG, "Camera running (Jetson TCP JPEG streaming)");

  const int frame_period_ms =
      (MJPEG_FPS > 0) ? (1000 / MJPEG_FPS) : 0;
  uint32_t report_frames = 0;
  uint64_t report_bytes = 0;
  int64_t report_t0_us = esp_timer_get_time();

  while (1) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGW(TAG, "esp_camera_fb_get failed");
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }
    report_frames++;
    report_bytes += fb->len;

    uint32_t jpeg_len = (uint32_t)fb->len;
    uint8_t len_be[4];
    len_be[0] = (uint8_t)((jpeg_len >> 24) & 0xFF);
    len_be[1] = (uint8_t)((jpeg_len >> 16) & 0xFF);
    len_be[2] = (uint8_t)((jpeg_len >> 8) & 0xFF);
    len_be[3] = (uint8_t)(jpeg_len & 0xFF);

    if (tcp_send_all(s_tcp_sock, len_be, sizeof(len_be)) != 0 ||
        tcp_send_all(s_tcp_sock, fb->buf, fb->len) != 0) {
      ESP_LOGW(TAG, "TCP send failed, reconnecting...");
      close(s_tcp_sock);
      s_tcp_sock = -1;
      while (s_tcp_sock < 0) {
        s_tcp_sock = tcp_connect(JETSON_TCP_IP, JETSON_TCP_PORT);
        if (s_tcp_sock < 0) {
          ESP_LOGW(TAG, "TCP reconnect failed, retrying...");
          vTaskDelay(pdMS_TO_TICKS(1000));
        }
      }
    }

    esp_camera_fb_return(fb);
    if (frame_period_ms > 0) {
      vTaskDelay(pdMS_TO_TICKS(frame_period_ms));
    }

    int64_t now_us = esp_timer_get_time();
    int64_t dt_us = now_us - report_t0_us;
    if (dt_us >= 1000000) {
      float fps = (dt_us > 0)
                      ? ((float)report_frames * 1000000.0f / (float)dt_us)
                      : 0.0f;
      float kbps =
          (dt_us > 0)
              ? ((float)report_bytes * 8.0f / 1000.0f) /
                    ((float)dt_us / 1000000.0f)
              : 0.0f;
      ESP_LOGI(TAG, "camera ok: fps=%.1f avg_frame=%u bytes bitrate=%.0f kbps",
               fps,
               (unsigned)(report_frames ? (report_bytes / report_frames) : 0),
               kbps);
      report_frames = 0;
      report_bytes = 0;
      report_t0_us = now_us;
    }
  }

#elif ENABLE_WEB_PREVIEW
  web_preview_start();
  ESP_LOGI(TAG, "Camera running (browser web preview; HTTP server owns frames)");
  while (1) {
    vTaskDelay(pdMS_TO_TICKS(60000));
  }

#else
  ESP_LOGI(TAG, "Camera running (local test mode)");

  const int frame_period_ms =
      (MJPEG_FPS > 0) ? (1000 / MJPEG_FPS) : 0;
  uint32_t report_frames = 0;
  uint64_t report_bytes = 0;
  int64_t report_t0_us = esp_timer_get_time();

  while (1) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGW(TAG, "esp_camera_fb_get failed");
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    report_frames++;
    report_bytes += fb->len;
    esp_camera_fb_return(fb);

    if (frame_period_ms > 0) {
      vTaskDelay(pdMS_TO_TICKS(frame_period_ms));
    }

    int64_t now_us = esp_timer_get_time();
    int64_t dt_us = now_us - report_t0_us;
    if (dt_us >= 1000000) {
      float fps = (dt_us > 0)
                      ? ((float)report_frames * 1000000.0f / (float)dt_us)
                      : 0.0f;
      float kbps =
          (dt_us > 0)
              ? ((float)report_bytes * 8.0f / 1000.0f) /
                    ((float)dt_us / 1000000.0f)
              : 0.0f;
      ESP_LOGI(TAG, "camera ok: fps=%.1f avg_frame=%u bytes bitrate=%.0f kbps",
               fps,
               (unsigned)(report_frames ? (report_bytes / report_frames) : 0),
               kbps);
      report_frames = 0;
      report_bytes = 0;
      report_t0_us = now_us;
    }
  }
#endif
}
