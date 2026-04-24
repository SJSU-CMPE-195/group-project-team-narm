#include "driver/i2c_master.h"
#include "driver/isp.h"
#include "esp_cam_ctlr.h"
#include "esp_cam_ctlr_csi.h"
#include "esp_err.h"
#include "esp_ldo_regulator.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "freertos/event_groups.h"

#include "driver/jpeg_encode.h"

#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"

#include "esp_wifi.h"
#include "esp_wifi_remote.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"

#include <unistd.h>

static const char *TAG = "ov5647_capture";

#define CAM_WIDTH 800
#define CAM_HEIGHT 640
#define CAM_CLK_MHZ 200

/* Jetson TCP server settings (change to match your Jetson). */
#define JETSON_TCP_IP "192.168.1.10"
#define JETSON_TCP_PORT 5000

/* Tune these later if bandwidth/quality needs adjustment. */
#define MJPEG_FPS 10
#define MJPEG_QUALITY 70 /* 1-100 (higher = better quality + larger frames) */

// Wi-Fi Remote STA connection
#define WIFI_REMOTE_CONNECTED_BIT BIT0
#define WIFI_REMOTE_FAIL_BIT BIT1
#define WIFI_REMOTE_BITS (WIFI_REMOTE_CONNECTED_BIT | WIFI_REMOTE_FAIL_BIT)

static EventGroupHandle_t s_wifi_event_group;
static int s_wifi_remote_retry_num = 0;

static void event_handler_remote(void *arg, esp_event_base_t event_base,
                                 int32_t event_id, void *event_data) {
  (void)arg;
  (void)event_data;

  if (event_base == WIFI_REMOTE_EVENT && event_id == WIFI_EVENT_STA_START) {
    // Kick off connection attempts when STA starts.
    esp_wifi_remote_connect();
  } else if (event_base == WIFI_REMOTE_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    if (s_wifi_remote_retry_num < CONFIG_ESP_WIFI_REMOTE_MAX_RETRIES) {
      esp_wifi_remote_connect();
      s_wifi_remote_retry_num++;
      ESP_LOGI(TAG, "Remote Wi-Fi: retry to connect (attempt %d)",
               s_wifi_remote_retry_num);
    } else {
      xEventGroupSetBits(s_wifi_event_group, WIFI_REMOTE_FAIL_BIT);
    }
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    s_wifi_remote_retry_num = 0;
    xEventGroupSetBits(s_wifi_event_group, WIFI_REMOTE_CONNECTED_BIT);
  }
}

static void wifi_init_remote_sta(void) {
  s_wifi_event_group = xEventGroupCreate();

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());

  esp_wifi_remote_create_default_sta();

  wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_remote_init(&wifi_cfg));

  ESP_ERROR_CHECK(esp_event_handler_register(
      WIFI_REMOTE_EVENT, ESP_EVENT_ANY_ID, event_handler_remote, NULL));
  ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                             event_handler_remote, NULL));

  wifi_config_t wifi_config = {
      .sta =
          {
              .ssid = CONFIG_ESP_WIFI_REMOTE_SSID,
              .password = CONFIG_ESP_WIFI_REMOTE_PASSWORD,
          },
  };

  ESP_ERROR_CHECK(esp_wifi_remote_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_remote_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_remote_start());

  ESP_LOGI(TAG, "Waiting for remote Wi-Fi STA connection...");
  EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group, WIFI_REMOTE_BITS,
                                         pdFALSE, pdFALSE, portMAX_DELAY);

  if (bits & WIFI_REMOTE_CONNECTED_BIT) {
    ESP_LOGI(TAG, "Remote Wi-Fi connected (IP acquired).");
  } else if (bits & WIFI_REMOTE_FAIL_BIT) {
    ESP_LOGW(TAG, "Remote Wi-Fi failed to connect after retries.");
  } else {
    ESP_LOGE(TAG, "Remote Wi-Fi: unexpected event bits=0x%lx",
             (unsigned long)bits);
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
  return sock;
}

// OV5647 I2C address
#define OV5647_ADDR 0x36
#define I2C_SCL_PIN 8
#define I2C_SDA_PIN 7

// LDO init for MIPI CSI 2.5V
static void init_ldo(void) {
  esp_ldo_channel_handle_t ldo_mipi_phy = NULL;
  esp_ldo_channel_config_t ldo_cfg = {
      .chan_id = 3,
      .voltage_mv = 2500,
  };
  ESP_ERROR_CHECK(esp_ldo_acquire_channel(&ldo_cfg, &ldo_mipi_phy));
  ESP_LOGI(TAG, "MIPI CSI LDO initialized at 2500mV");
}

// Write one register to OV5647 over I2C
static esp_err_t ov5647_write_reg(i2c_master_dev_handle_t dev, uint16_t reg,
                                  uint8_t val) {
  uint8_t buf[3] = {reg >> 8, reg & 0xFF, val};
  return i2c_master_transmit(dev, buf, sizeof(buf), pdMS_TO_TICKS(100));
}

// Minimal OV5647 init sequence for MIPI CSI 800x640 RAW8
static void init_ov5647(i2c_master_dev_handle_t dev) {
  ESP_LOGI(TAG, "Initializing OV5647...");

  // Software reset
  ov5647_write_reg(dev, 0x0103, 0x01);
  vTaskDelay(pdMS_TO_TICKS(10));

  // MIPI enable, 2-lane
  ov5647_write_reg(dev, 0x3018, 0x72);
  ov5647_write_reg(dev, 0x3019, 0x00);
  ov5647_write_reg(dev, 0x3034, 0x1A); // 10-bit RAW
  ov5647_write_reg(dev, 0x3035, 0x21);
  ov5647_write_reg(dev, 0x3036, 0x46);
  ov5647_write_reg(dev, 0x303C, 0x11);

  // Resolution 800x640
  ov5647_write_reg(dev, 0x3820, 0x41);
  ov5647_write_reg(dev, 0x3821, 0x07);
  ov5647_write_reg(dev, 0x380C, 0x07);
  ov5647_write_reg(dev, 0x380D, 0x3C);
  ov5647_write_reg(dev, 0x380E, 0x03);
  ov5647_write_reg(dev, 0x380F, 0xE8);

  // Stream on
  ov5647_write_reg(dev, 0x4800, 0x04);
  ov5647_write_reg(dev, 0x0100, 0x01);

  ESP_LOGI(TAG, "OV5647 init done");
}

// Frame callback
static bool on_frame_ready(esp_cam_ctlr_handle_t handle,
                           esp_cam_ctlr_trans_t *trans, void *user_data) {
  (void)handle;
  (void)user_data;
  ESP_LOGD(TAG, "Frame received: %d bytes", (int)trans->received_size);
  return false; // false = do not free buffer
}

void app_main(void) {
  // 1. Init LDO
  init_ldo();

  // 2. Init I2C for OV5647
  i2c_master_bus_handle_t i2c_bus = NULL;
  i2c_master_bus_config_t i2c_bus_cfg = {
      .i2c_port = I2C_NUM_0,
      .sda_io_num = I2C_SDA_PIN,
      .scl_io_num = I2C_SCL_PIN,
      .clk_source = I2C_CLK_SRC_DEFAULT,
      .glitch_ignore_cnt = 7,
  };
  ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_bus_cfg, &i2c_bus));

  i2c_master_dev_handle_t ov5647_dev = NULL;
  i2c_device_config_t dev_cfg = {
      .dev_addr_length = I2C_ADDR_BIT_LEN_7,
      .device_address = OV5647_ADDR,
      .scl_speed_hz = 100000,
  };
  ESP_ERROR_CHECK(i2c_master_bus_add_device(i2c_bus, &dev_cfg, &ov5647_dev));

  // 3. Init OV5647 sensor
  init_ov5647(ov5647_dev);

  // 4. Remote Wi-Fi STA (ESP32-P4-WIFI6) — before CSI streaming so TCP can use
  // the link
  esp_err_t nvs_ret = nvs_flash_init();
  if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ESP_ERROR_CHECK(nvs_flash_init());
  }
  wifi_init_remote_sta();

  // 5. Allocate camera output buffer in PSRAM (YUV422 = 2 bytes/pixel)
  const size_t yuv_buf_size = (size_t)CAM_WIDTH * (size_t)CAM_HEIGHT * 2;
  uint8_t *frame_buf = heap_caps_malloc(yuv_buf_size, MALLOC_CAP_SPIRAM);
  assert(frame_buf != NULL);

  // 6. Initialize JPEG encoder engine (ESP32-P4 JPEG HW)
  jpeg_encoder_handle_t jpeg_enc = NULL;
  jpeg_encode_engine_cfg_t jpeg_eng_cfg = {
      .intr_priority = 0,
      // If encoding ever stalls, we want to drop frames rather than block capture forever.
      .timeout_ms = 200,
  };
  ESP_ERROR_CHECK(jpeg_new_encoder_engine(&jpeg_eng_cfg, &jpeg_enc));

  jpeg_encode_cfg_t jpeg_cfg = {
      .height = CAM_HEIGHT,
      .width = CAM_WIDTH,
      .src_type = JPEG_ENC_SRC_YUV422,
      .sub_sample = JPEG_DOWN_SAMPLING_YUV422,
      .image_quality = MJPEG_QUALITY,
  };

  // JPEG size varies; allocate a generous PSRAM output buffer.
  // Rule of thumb: allow up to ~1 byte/pixel at moderate quality.
  // (Keep headroom: encoder may not always error cleanly when outbuf too small.)
  const size_t jpeg_out_cap = (size_t)CAM_WIDTH * (size_t)CAM_HEIGHT * 2;
  uint8_t *jpeg_out_buf = heap_caps_malloc(jpeg_out_cap, MALLOC_CAP_SPIRAM);
  assert(jpeg_out_buf != NULL);

  // 7. Configure CSI controller (RAW8 in, YUV422 out for JPEG encoder)
  esp_cam_ctlr_csi_config_t csi_cfg = {
      .ctlr_id = 0,
      .h_res = CAM_WIDTH,
      .v_res = CAM_HEIGHT,
      .lane_bit_rate_mbps = 200,
      .input_data_color_type = CAM_CTLR_COLOR_RAW8,
      .output_data_color_type = CAM_CTLR_COLOR_YUV422,
      .data_lane_num = 2,
      .byte_swap_en = false,
      // More buffering reduces "queue full" during encode/send spikes.
      .queue_items = 4,
  };

  esp_cam_ctlr_handle_t cam_handle = NULL;
  ESP_ERROR_CHECK(esp_cam_new_csi_ctlr(&csi_cfg, &cam_handle));

  // 8. Register frame callback
  esp_cam_ctlr_evt_cbs_t cbs = {
      .on_trans_finished = on_frame_ready,
  };
  ESP_ERROR_CHECK(
      esp_cam_ctlr_register_event_callbacks(cam_handle, &cbs, NULL));

  // 9. Enable and start
  ESP_ERROR_CHECK(esp_cam_ctlr_enable(cam_handle));
  ESP_ERROR_CHECK(esp_cam_ctlr_start(cam_handle));

  ESP_LOGI(TAG, "Camera + MJPEG started, connecting to Jetson TCP...");

  // 10. Connect to Jetson (single persistent TCP connection)
  int sock = -1;
  while (sock < 0) {
    sock = tcp_connect(JETSON_TCP_IP, JETSON_TCP_PORT);
    if (sock < 0) {
      ESP_LOGW(TAG, "TCP connect failed, retrying...");
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }

  // 10. Receive loop: CSI frame -> JPEG -> TCP send
  uint32_t frame_idx = 0;
  while (1) {
    esp_cam_ctlr_trans_t trans = {
        .buffer = frame_buf,
        .buflen = yuv_buf_size,
    };

    esp_err_t cam_ret =
        esp_cam_ctlr_receive(cam_handle, &trans, pdMS_TO_TICKS(2000));
    if (cam_ret == ESP_OK && trans.received_size > 0) {
      if (trans.received_size != yuv_buf_size) {
        ESP_LOGW(TAG, "Unexpected YUV size: got %d expected %d",
                 (int)trans.received_size, (int)yuv_buf_size);
        continue;
      }

      uint32_t jpeg_len = 0;
      esp_err_t jpeg_ret =
          jpeg_encoder_process(jpeg_enc, &jpeg_cfg, frame_buf,
                               (uint32_t)yuv_buf_size, jpeg_out_buf,
                               (uint32_t)jpeg_out_cap, &jpeg_len);
      if (jpeg_ret != ESP_OK || jpeg_len == 0) {
        ESP_LOGW(TAG, "JPEG encode failed: 0x%x len=%u", (unsigned)jpeg_ret,
                 (unsigned)jpeg_len);
        continue;
      }

      // Frame format: [4-byte BE length][exact JPEG payload]
      uint8_t len_be[4];
      len_be[0] = (uint8_t)((jpeg_len >> 24) & 0xFF);
      len_be[1] = (uint8_t)((jpeg_len >> 16) & 0xFF);
      len_be[2] = (uint8_t)((jpeg_len >> 8) & 0xFF);
      len_be[3] = (uint8_t)(jpeg_len & 0xFF);

      if (tcp_send_all(sock, len_be, sizeof(len_be)) != 0 ||
          tcp_send_all(sock, jpeg_out_buf, jpeg_len) != 0) {
        ESP_LOGW(TAG, "TCP send failed, reconnecting...");
        close(sock);
        sock = -1;
        while (sock < 0) {
          sock = tcp_connect(JETSON_TCP_IP, JETSON_TCP_PORT);
          if (sock < 0) {
            ESP_LOGW(TAG, "TCP reconnect failed, retrying...");
            vTaskDelay(pdMS_TO_TICKS(1000));
          }
        }
      }

      frame_idx++;

      // Simple pacing to avoid saturating the link/receiver.
      vTaskDelay(pdMS_TO_TICKS(1000 / MJPEG_FPS));
    } else {
      ESP_LOGW(TAG, "Frame receive timeout or error: 0x%x", cam_ret);
    }
  }
}
