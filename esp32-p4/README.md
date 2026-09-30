# ESP32-P4 OV5647 Camera Firmware

This firmware targets the **Waveshare ESP32-P4-WIFI6** with an **OV5647
MIPI-CSI camera**. Browser MJPEG streaming is enabled by default. The older
Jetson H.264 sender remains available as an alternate build mode.

## Browser mode

The default firmware captures RGB565 frames, JPEG-encodes them with the P4
hardware JPEG encoder, and serves an HTTP MJPEG stream:

    http://<esp32-ip>/
    http://<esp32-ip>/stream

Connect the ESP32-P4 and your laptop/phone to the same Wi-Fi network. The IP
address and browser URL are printed by `idf.py monitor` after boot.

## Requirements

- ESP-IDF 5.4 or newer
- Waveshare ESP32-P4-WIFI6
- OV5647 connected to the board's MIPI-CSI connector
- Network access during the first build so the ESP-IDF component manager can
  fetch the pinned esp_video package and its example initialization component

## Configure

    cd esp32-p4
    idf.py set-target esp32p4
    idf.py menuconfig

Set these menu items:

1. **Example Connection Configuration**
   - Wi-Fi SSID and password (2.4 GHz).
2. **Camera Streaming Configuration**
   - Keep **Serve browser MJPEG stream** enabled.
   - Adjust HTTP port, JPEG quality, resolution, and FPS if needed.

The checked-in defaults already select the Waveshare board's ESP32-C6 over
4-bit SDIO, its GPIO wiring, 32 MB flash, 32 MB PSRAM at 200 MHz, the OV5647
MIPI-CSI sensor, the ISP pipeline, and the ESP32-P4 hardware video support.

The performance defaults are:

- 1920x1080 capture
- 30 FPS
- JPEG quality 70
- Three camera buffers

The driver prints the negotiated resolution at startup because the sensor may
select the nearest supported mode.

## Build and flash

    idf.py build
    idf.py -p <PORT> flash monitor

After the ESP32 joins Wi-Fi, open the printed URL in a browser.

## Jetson H.264 mode

To restore the original Jetson sender, open `idf.py menuconfig`, disable
**Serve browser MJPEG stream**, and configure the Jetson IP, TCP port, H.264
settings, and stream button under **Camera Streaming Configuration**.

That mode sends each access unit as:

    [4-byte big-endian payload length][H.264 access unit]

## Jetson command

From jetson-code:

    PAYLOAD_FORMAT=h264 \
    MODEL_PATH=/absolute/path/to/models/action.h5 \
    python3 main.py

## H.264 performance output

Every five seconds the ESP prints:

    stats: encoded=150 sent=150 fps=30.0 bitrate=3.92Mbps queue=0 age=2.1ms

The Jetson prints receive, decode, inference, queue, drop, and end-to-end
latency measurements separately. Use both lines when tuning:

- If ESP encoded FPS is low, investigate camera/ISP/encoder configuration.
- If ESP queue depth or buffer age rises, investigate Wi-Fi or TCP throughput.
- If Jetson decode FPS matches receive FPS but inference FPS is low, the
  MediaPipe/model stage is the bottleneck.

## Important H.264 rule

Do not discard encoded P-frames to reduce latency. Decode every H.264 access
unit in order and drop only fully decoded RGB frames. The Jetson implementation
in this repository follows that rule.
