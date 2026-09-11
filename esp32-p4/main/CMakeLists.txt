idf_component_register(
    SRCS "h264_stream_main.c"
    REQUIRES
        driver
        esp_event
        esp_netif
        esp_timer
        esp_video
        example_video_common
        lwip
        nvs_flash
        protocol_examples_common
)
