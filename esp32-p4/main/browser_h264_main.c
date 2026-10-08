/* Direct ESP32-P4 -> browser H.264 in fragmented MP4 over HTTP.
 * The browser uses MediaSource, not WebCodecs or an external relay. */
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_check.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "protocol_examples_common.h"
#include "example_video_common.h"
#include "h264_camera.h"
#include "h264_mp4.h"

static const char *TAG = "browser_h264";
static h264_camera_t s_camera = { .camera_fd = -1, .encoder_fd = -1 };

extern const uint8_t player_html_start[] asm("_binary_h264_player_html_start");
extern const uint8_t player_html_end[] asm("_binary_h264_player_html_end");
extern const uint8_t player_js_start[] asm("_binary_h264_player_js_start");
extern const uint8_t player_js_end[] asm("_binary_h264_player_js_end");

typedef struct {
    QueueHandle_t frames;
    SemaphoreHandle_t available_buffers;
    SemaphoreHandle_t producer_done;
    volatile bool stopping;
    volatile bool failed;
    volatile uint32_t encoded_frames;
} stream_session_t;

static void release_frame(stream_session_t *session, const h264_camera_frame_t *frame)
{
    if (h264_camera_return(&s_camera, frame->index) != ESP_OK) {
        session->failed = true;
        session->stopping = true;
    }
    xSemaphoreGive(session->available_buffers);
}

static void capture_task(void *argument)
{
    stream_session_t *session = argument;
    while (!session->stopping) {
        if (xSemaphoreTake(session->available_buffers, pdMS_TO_TICKS(50)) != pdTRUE) {
            continue;
        }
        if (session->stopping) {
            xSemaphoreGive(session->available_buffers);
            break;
        }
        h264_camera_frame_t frame;
        if (h264_camera_next(&s_camera, &frame) != ESP_OK) {
            session->failed = true;
            xSemaphoreGive(session->available_buffers);
            break;
        }
        ++session->encoded_frames;
        bool queued = false;
        while (!session->stopping && !queued) {
            queued = xQueueSend(session->frames, &frame, pdMS_TO_TICKS(50)) == pdTRUE;
        }
        if (!queued) {
            release_frame(session, &frame);
        }
    }
    xSemaphoreGive(session->producer_done);
    vTaskDelete(NULL);
}

static void drain_frames(stream_session_t *session)
{
    h264_camera_frame_t frame;
    while (xQueueReceive(session->frames, &frame, 0) == pdTRUE) {
        release_frame(session, &frame);
    }
}

static void stop_session(stream_session_t *session, bool producer_started)
{
    session->stopping = true;
    if (producer_started) {
        /* Release queued MMAP buffers while waiting, so the producer cannot
         * deadlock behind a disconnected browser. Dequeues have a timeout. */
        while (xSemaphoreTake(session->producer_done, pdMS_TO_TICKS(50)) != pdTRUE) {
            drain_frames(session);
        }
        drain_frames(session);
    }
    h264_camera_stop(&s_camera);
    if (session->frames) {
        vQueueDelete(session->frames);
    }
    if (session->available_buffers) {
        vSemaphoreDelete(session->available_buffers);
    }
    if (session->producer_done) {
        vSemaphoreDelete(session->producer_done);
    }
}

static esp_err_t root_handler(httpd_req_t *request)
{
    httpd_resp_set_type(request, "text/html; charset=utf-8");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_send(request, (const char *)player_html_start,
                           player_html_end - player_html_start);
}

static esp_err_t javascript_handler(httpd_req_t *request)
{
    httpd_resp_set_type(request, "application/javascript; charset=utf-8");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");
    return httpd_resp_send(request, (const char *)player_js_start,
                           player_js_end - player_js_start);
}

static esp_err_t stream_handler(httpd_req_t *request)
{
    /* Retry configuration after a failed start released the devices. */
    if ((s_camera.camera_fd < 0 || s_camera.encoder_fd < 0) &&
        h264_camera_open(&s_camera) != ESP_OK) {
        return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                                   "Camera/H.264 hardware is unavailable. Check serial logs.");
    }
    stream_session_t session = { 0 };
    bool producer_started = false;
    bool response_started = false;
    bool holding_frame = false;
    h264_camera_frame_t frame = { 0 };
    esp_err_t result = ESP_FAIL;
    uint8_t *mp4 = NULL;

    session.frames = xQueueCreate(s_camera.encoded_count, sizeof(frame));
    session.available_buffers = xSemaphoreCreateCounting(s_camera.encoded_count,
                                                         s_camera.encoded_count);
    session.producer_done = xSemaphoreCreateBinary();
    if (!session.frames || !session.available_buffers || !session.producer_done) {
        goto done;
    }
    size_t capacity = 2048;
    for (uint32_t i = 0; i < s_camera.encoded_count; ++i) {
        if (s_camera.encoded_size[i] > SIZE_MAX - 2048) {
            goto done;
        }
        size_t needed = s_camera.encoded_size[i] + 2048;
        if (needed > capacity) {
            capacity = needed;
        }
    }
    mp4 = heap_caps_malloc(capacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!mp4) {
        goto done;
    }
    if (h264_camera_start(&s_camera) != ESP_OK) {
        /* A partial start can leave buffers queued even when STREAMON
         * failed. Close the devices so the next request starts cleanly. */
        h264_camera_close(&s_camera);
        goto done;
    }
    if (xTaskCreate(capture_task, "h264_capture", 4096, &session, 6, NULL) != pdPASS) {
        goto done;
    }
    producer_started = true;

    int no_delay = 1;
    if (setsockopt(httpd_req_to_sockfd(request), IPPROTO_TCP, TCP_NODELAY,
                   &no_delay, sizeof(no_delay)) != 0) {
        ESP_LOGW(TAG, "TCP_NODELAY failed: errno=%d", errno);
    }

    h264_mp4_config_t config = { 0 };
    bool initialized = false;
    uint32_t skipped = 0, sequence = 1;
    uint64_t decode_time = 0;
    uint32_t duration = 90000 / s_camera.fps;
    int64_t stats_start = esp_timer_get_time();
    uint32_t sent_frames = 0, previous_encoded = 0;
    uint64_t sent_bytes = 0;
    int64_t encode_us = 0, mux_us = 0, send_us = 0, age_us = 0;
    /* These header strings must live until the first HTTP chunk is sent. */
    char mime[64], width[16], height[16], fps[16];

    while (!session.failed) {
        if (xQueueReceive(session.frames, &frame, pdMS_TO_TICKS(200)) != pdTRUE) {
            continue;
        }
        holding_frame = true;
        const uint8_t *encoded = s_camera.encoded_data[frame.index];
        if (!initialized) {
            if (!h264_mp4_is_keyframe(encoded, frame.size) ||
                !h264_mp4_configure(&config, encoded, frame.size,
                                     s_camera.width, s_camera.height)) {
                release_frame(&session, &frame);
                holding_frame = false;
                if (++skipped > 2 * CONFIG_EXAMPLE_H264_I_PERIOD) {
                    ESP_LOGE(TAG, "no IDR frame with SPS/PPS from the encoder");
                    break;
                }
                continue;
            }
            size_t init_size = h264_mp4_init_segment(&config, mp4, capacity);
            if (!init_size) {
                ESP_LOGE(TAG, "could not build the MP4 initialization segment");
                break;
            }
            snprintf(mime, sizeof(mime), "video/mp4; codecs=\"%s\"", config.codec);
            snprintf(width, sizeof(width), "%" PRIu32, s_camera.width);
            snprintf(height, sizeof(height), "%" PRIu32, s_camera.height);
            snprintf(fps, sizeof(fps), "%" PRIu32, s_camera.fps);
            httpd_resp_set_type(request, mime);
            httpd_resp_set_hdr(request, "Cache-Control", "no-store");
            httpd_resp_set_hdr(request, "X-Video-Width", width);
            httpd_resp_set_hdr(request, "X-Video-Height", height);
            httpd_resp_set_hdr(request, "X-Video-FPS", fps);
            result = httpd_resp_send_chunk(request, (const char *)mp4, init_size);
            response_started = true;
            if (result != ESP_OK) {
                break;
            }
            initialized = true;
            ESP_LOGI(TAG, "browser H.264 started: %" PRIu32 "x%" PRIu32
                     " codec=%s, target=%d bps, TCP send buffer=%d",
                     s_camera.width, s_camera.height, config.codec,
                     CONFIG_EXAMPLE_H264_BITRATE, CONFIG_LWIP_TCP_SND_BUF_DEFAULT);
        }

        int64_t mux_start = esp_timer_get_time();
        size_t fragment_size = h264_mp4_fragment(encoded, frame.size, sequence,
                                                 decode_time, duration, mp4, capacity);
        int64_t mux_end = esp_timer_get_time();
        uint32_t encoded_size = frame.size;
        int64_t encoded_at = frame.encoded_us, captured_at = frame.captured_us;
        /* The MP4 buffer owns a copy now. Return the MMAP buffer before
         * potentially blocking on TCP; encoding and sending can overlap. */
        release_frame(&session, &frame);
        holding_frame = false;
        if (!fragment_size || session.failed) {
            ESP_LOGE(TAG, "invalid H.264 access unit or MP4 buffer too small");
            result = ESP_FAIL;
            break;
        }
        result = httpd_resp_send_chunk(request, (const char *)mp4, fragment_size);
        int64_t sent_at = esp_timer_get_time();
        if (result != ESP_OK) {
            break;
        }
        ++sequence;
        decode_time += duration;
        ++sent_frames;
        sent_bytes += encoded_size;
        encode_us += encoded_at - captured_at;
        mux_us += mux_end - mux_start;
        send_us += sent_at - mux_end;
        age_us += sent_at - captured_at;
        if (sent_at - stats_start >= CONFIG_EXAMPLE_STATS_INTERVAL_SEC * 1000000LL) {
            uint32_t encoded_count = session.encoded_frames;
            double seconds = (sent_at - stats_start) / 1000000.0;
            ESP_LOGI(TAG, "H264 stats: %" PRIu32 "x%" PRIu32 " encoded=%" PRIu32
                     " sent=%" PRIu32 " fps=%.1f bitrate=%.2f Mbps encode=%.1f"
                     " mux=%.1f send=%.1f age=%.1f ms queue=%u",
                     s_camera.width, s_camera.height, encoded_count - previous_encoded,
                     sent_frames, sent_frames / seconds, sent_bytes * 8.0 / seconds / 1000000.0,
                     encode_us / (sent_frames * 1000.0), mux_us / (sent_frames * 1000.0),
                     send_us / (sent_frames * 1000.0), age_us / (sent_frames * 1000.0),
                     (unsigned)uxQueueMessagesWaiting(session.frames));
            stats_start = sent_at;
            previous_encoded = encoded_count;
            sent_frames = 0;
            sent_bytes = 0;
            encode_us = mux_us = send_us = age_us = 0;
        }
    }

done:
    if (holding_frame) {
        release_frame(&session, &frame);
    }
    stop_session(&session, producer_started);
    heap_caps_free(mp4);
    ESP_LOGI(TAG, "browser H.264 stopped; ready for a new viewer");
    if (!response_started) {
        return httpd_resp_send_err(request, HTTPD_500_INTERNAL_SERVER_ERROR,
                                   "Could not start the camera/H.264 stream. Check serial logs.");
    }
    /* A failed response must close this socket; a new request creates a
     * fresh encoder and a new IDR/SPS/PPS sequence. */
    return result == ESP_OK ? ESP_FAIL : result;
}

static esp_err_t start_http_server(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = CONFIG_EXAMPLE_HTTP_PORT;
    config.stack_size = 8192;
    config.send_wait_timeout = CONFIG_EXAMPLE_TCP_SEND_TIMEOUT_MS / 1000;
    if (config.send_wait_timeout < 1) {
        config.send_wait_timeout = 1;
    }
    httpd_handle_t server;
    ESP_RETURN_ON_ERROR(httpd_start(&server, &config), TAG, "HTTP server failed");
    const httpd_uri_t handlers[] = {
        { .uri = "/", .method = HTTP_GET, .handler = root_handler },
        { .uri = "/player.js", .method = HTTP_GET, .handler = javascript_handler },
        { .uri = "/stream.mp4", .method = HTTP_GET, .handler = stream_handler },
    };
    for (size_t i = 0; i < sizeof(handlers) / sizeof(handlers[0]); ++i) {
        esp_err_t result = httpd_register_uri_handler(server, &handlers[i]);
        if (result != ESP_OK) {
            httpd_stop(server);
            return result;
        }
    }
    return ESP_OK;
}

void app_main(void)
{
    esp_err_t result = nvs_flash_init();
    if (result == ESP_ERR_NVS_NO_FREE_PAGES || result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        result = nvs_flash_init();
    }
    ESP_ERROR_CHECK(result);
    ESP_ERROR_CHECK(example_video_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(example_connect());
#if CONFIG_EXAMPLE_CONNECT_WIFI
    result = esp_wifi_set_ps(WIFI_PS_NONE);
    if (result != ESP_OK) {
        ESP_LOGW(TAG, "could not disable Wi-Fi power saving: %s", esp_err_to_name(result));
    }
    wifi_ap_record_t ap;
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
        ESP_LOGI(TAG, "Wi-Fi link: RSSI=%d dBm, channel=%u", ap.rssi, ap.primary);
    }
#endif
    ESP_ERROR_CHECK(h264_camera_open(&s_camera));
    ESP_ERROR_CHECK(start_http_server());
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    esp_netif_ip_info_t ip;
    if (netif && esp_netif_get_ip_info(netif, &ip) == ESP_OK) {
        ESP_LOGI(TAG, "open http://" IPSTR ":%d/ for direct H.264 playback",
                 IP2STR(&ip.ip), CONFIG_EXAMPLE_HTTP_PORT);
    }
    ESP_LOGI(TAG, "one browser viewer at a time; H.264 MP4 endpoint: /stream.mp4");
}
