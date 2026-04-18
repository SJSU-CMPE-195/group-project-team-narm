idf_component_register(SRCS "h264_stream_main.c"
                       REQUIRES esp_video example_video_common protocol_examples_common driver nvs_flash esp_netif esp_event lwip)
