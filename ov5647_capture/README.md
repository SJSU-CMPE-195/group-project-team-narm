# ov5647_capture (XIAO ESP32-S3 Sense -> Jetson TCP JPEG)

This ESP-IDF app streams camera JPEG frames to the Jetson using a length-prefixed TCP protocol:

- `[len:u32be][jpeg_payload]...`

## Credential/config source of truth

Tracked defaults are intentionally non-secret:

- `sdkconfig.defaults`
- `main/Kconfig.projbuild`

Set real Wi-Fi credentials locally with:

```bash
idf.py menuconfig
```

Then open:

- `TCP streaming Wi-Fi (ESP32-S3 STA)`

and set:

- `Remote Wi-Fi SSID`
- `Remote Wi-Fi Password`
- `Remote Wi-Fi maximum retry count`

After saving and rebuilding, confirm resolved values in:

- `build/config/sdkconfig.json`

(`ESP_WIFI_REMOTE_SSID`, `ESP_WIFI_REMOTE_PASSWORD`, `ESP_WIFI_REMOTE_MAX_RETRIES`)

## Build/flash

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

