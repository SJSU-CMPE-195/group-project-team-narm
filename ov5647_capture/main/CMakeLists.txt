idf_component_register(SRCS "ov5647_capture.c"
                    INCLUDE_DIRS "."
                    REQUIRES esp_driver_cam esp_driver_isp esp_driver_i2c
                             esp_hw_support esp_psram freertos)