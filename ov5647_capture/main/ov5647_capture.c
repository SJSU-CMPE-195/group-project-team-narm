#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "esp_ldo_regulator.h"
#include "esp_cam_ctlr_csi.h"
#include "esp_cam_ctlr.h"
#include "driver/isp.h"

static const char *TAG = "ov5647_capture";

#define CAM_WIDTH       800
#define CAM_HEIGHT      640
#define CAM_CLK_MHZ     200

// OV5647 I2C address
#define OV5647_ADDR     0x36
#define I2C_SCL_PIN     31
#define I2C_SDA_PIN     34

// LDO init for MIPI CSI 2.5V
static void init_ldo(void)
{
    esp_ldo_channel_handle_t ldo_mipi_phy = NULL;
    esp_ldo_channel_config_t ldo_cfg = {
        .chan_id    = 3,
        .voltage_mv = 2500,
    };
    ESP_ERROR_CHECK(esp_ldo_acquire_channel(&ldo_cfg, &ldo_mipi_phy));
    ESP_LOGI(TAG, "MIPI CSI LDO initialized at 2500mV");
}

// Write one register to OV5647 over I2C
static esp_err_t ov5647_write_reg(i2c_master_dev_handle_t dev,
                                   uint16_t reg, uint8_t val)
{
    uint8_t buf[3] = { reg >> 8, reg & 0xFF, val };
    return i2c_master_transmit(dev, buf, sizeof(buf), pdMS_TO_TICKS(100));
}

// Minimal OV5647 init sequence for MIPI CSI 800x640 RAW8
static void init_ov5647(i2c_master_dev_handle_t dev)
{
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
                            esp_cam_ctlr_trans_t *trans, void *user_data)
{
    ESP_LOGI(TAG, "Frame received: %d bytes", trans->received_size);
    return false; // false = do not free buffer
}

void app_main(void)
{
    // 1. Init LDO
    init_ldo();

    // 2. Init I2C for OV5647
    i2c_master_bus_handle_t i2c_bus = NULL;
    i2c_master_bus_config_t i2c_bus_cfg = {
        .i2c_port       = I2C_NUM_0,
        .sda_io_num     = I2C_SDA_PIN,
        .scl_io_num     = I2C_SCL_PIN,
        .clk_source     = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_bus_cfg, &i2c_bus));

    i2c_master_dev_handle_t ov5647_dev = NULL;
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = OV5647_ADDR,
        .scl_speed_hz    = 100000,
    };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(i2c_bus, &dev_cfg, &ov5647_dev));

    // 3. Init OV5647 sensor
    init_ov5647(ov5647_dev);

    // 4. Allocate frame buffer in PSRAM
    size_t buf_size = CAM_WIDTH * CAM_HEIGHT; // RAW8 = 1 byte/pixel
    uint8_t *frame_buf = heap_caps_malloc(buf_size, MALLOC_CAP_SPIRAM);
    assert(frame_buf != NULL);

    // 5. Configure CSI controller
    esp_cam_ctlr_csi_config_t csi_cfg = {
        .ctlr_id                = 0,
        .h_res                  = CAM_WIDTH,
        .v_res                  = CAM_HEIGHT,
        .lane_bit_rate_mbps     = 200,
        .input_data_color_type  = CAM_CTLR_COLOR_RAW8,
        .output_data_color_type = CAM_CTLR_COLOR_RAW8,
        .data_lane_num          = 2,
        .byte_swap_en           = false,
        .queue_items            = 1,
    };

    esp_cam_ctlr_handle_t cam_handle = NULL;
    ESP_ERROR_CHECK(esp_cam_new_csi_ctlr(&csi_cfg, &cam_handle));

    // 6. Register frame callback
    esp_cam_ctlr_evt_cbs_t cbs = {
        .on_trans_finished = on_frame_ready,
    };
    ESP_ERROR_CHECK(esp_cam_ctlr_register_event_callbacks(cam_handle, &cbs, NULL));

    // 7. Enable and start
    ESP_ERROR_CHECK(esp_cam_ctlr_enable(cam_handle));
    ESP_ERROR_CHECK(esp_cam_ctlr_start(cam_handle));

    ESP_LOGI(TAG, "Camera started, receiving frames...");

    // 8. Receive loop
    while (1) {
        esp_cam_ctlr_trans_t trans = {
            .buffer = frame_buf,
            .buflen = buf_size,
        };
        esp_err_t ret = esp_cam_ctlr_receive(cam_handle, &trans, pdMS_TO_TICKS(1000));
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "Frame captured: %dx%d, %d bytes",
                     CAM_WIDTH, CAM_HEIGHT, trans.received_size);
        } else {
            ESP_LOGW(TAG, "Frame receive timeout or error: 0x%x", ret);
        }
    }
}