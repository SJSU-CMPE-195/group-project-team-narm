idf_component_register(SRCS "ov5647_capture.c"
                    INCLUDE_DIRS "."
                    REQUIRES esp32_camera esp_psram esp_timer freertos
                             esp_wifi esp_netif esp_event nvs_flash esp_http_server)
