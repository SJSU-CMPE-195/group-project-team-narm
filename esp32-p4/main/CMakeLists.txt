if(CONFIG_EXAMPLE_BROWSER_MJPEG)
    set(STREAM_SOURCE "browser_stream_main.c")
else()
    set(STREAM_SOURCE "h264_stream_main.c")
endif()

idf_component_register(
    SRCS "${STREAM_SOURCE}"
    REQUIRES
        driver
        esp_event
        esp_http_server
        esp_netif
        esp_timer
        esp_video
        example_video_common
        lwip
        nvs_flash
        protocol_examples_common
)
