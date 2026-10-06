# ESP32-P4 OV5647 Camera Firmware

This firmware targets the **Waveshare ESP32-P4-WIFI6** with an **OV5647
MIPI-CSI camera**. Browser MJPEG streaming is enabled by default. The older
Jetson H.264 sender remains available as an alternate build mode.

## Browser mode

The default firmware captures RGB565 frames at the sensor's native 1920x1080,
resizes them to 1280x720 on the P4, JPEG-encodes them with the hardware JPEG
encoder, and serves an HTTP MJPEG stream:

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
   - Set **Frame width (browser output / H.264 capture)** to **1280** and
     **Frame height (browser output / H.264 capture)** to **720** for 720p.
   - Adjust HTTP port, JPEG quality, resolution, and FPS if needed.

The checked-in defaults already select the Waveshare board's ESP32-C6 over
4-bit SDIO, its GPIO wiring, 32 MB flash, 32 MB PSRAM at 200 MHz, the OV5647
MIPI-CSI sensor, the ISP pipeline, and the ESP32-P4 hardware video support.

The performance defaults are:

- 1920x1080 capture, resized to a 1280x720 MJPEG stream
- 30 FPS camera configuration (measured stream FPS depends on throughput)
- JPEG quality 70
- Three camera buffers
- 32 KiB TCP send buffer (uses more RAM while sending)

Browser streaming disables Wi-Fi modem sleep, enables TCP_NODELAY, and sends
the MJPEG boundary and part header together to reduce transport delays.
Keeping Wi-Fi awake increases power use. At boot, the firmware logs the access
point's RSSI/channel; stream startup logs the configured TCP send buffer size.
The common 1920x1080-to-1280x720 resize uses a specialized 3:2 loop that keeps
the same center-sampled pixels and full field of view as the general resizer.

Existing `sdkconfig` settings take precedence over defaults. There is no need
to run `set-target` again when this project is already targeting `esp32p4`;
that command resets project configuration, including Wi-Fi credentials.

The camera's capture dimensions and the JPEG output dimensions are logged
separately. For 720p, expect `camera ready: 1920x1080 @ 30 fps` followed by
`MJPEG output: 1280x720`. The nearest-neighbor resize samples the entire image
without cropping, preserving the field of view when the aspect ratio matches.
Matching the output dimensions to capture dimensions bypasses resizing.

While a browser is streaming, `MJPEG stats` reports measured sent FPS,
average JPEG size, and average wait/resize/encode/send times every five
seconds. These are server throughput measurements, not browser display FPS.

The resize helper can also be checked on a computer with GCC, from `esp32-p4`:

    gcc -std=c11 -O2 -Wall -Wextra -Werror -I main main/rgb565_resize.c tests/test_rgb565_resize.c -o build/test_rgb565_resize.exe
    ./build/test_rgb565_resize.exe

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
