if(CONFIG_EXAMPLE_BROWSER_H264)
    set(STREAM_SOURCE "browser_h264_main.c" "h264_camera.c" "h264_mp4.c")
    set(BROWSER_ASSETS "h264_player.html" "h264_player.js")
elseif(CONFIG_EXAMPLE_BROWSER_MJPEG)
    set(STREAM_SOURCE "browser_stream_main.c" "rgb565_resize.c")
else()
    set(STREAM_SOURCE "h264_stream_main.c")
endif()

idf_component_register(
    SRCS "${STREAM_SOURCE}"
    EMBED_FILES ${BROWSER_ASSETS}
    REQUIRES
        driver
        esp_event
        esp_http_server
        esp_netif
        esp_timer
        esp_video
        esp_wifi
        example_video_common
        heap
        lwip
        nvs_flash
        protocol_examples_common
)
