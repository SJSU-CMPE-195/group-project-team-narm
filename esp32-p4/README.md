# ESP32-P4 OV5647 Camera Firmware

This firmware targets the **Waveshare ESP32-P4-WIFI6** with an **OV5647
MIPI-CSI camera**. Direct browser H.264 streaming is enabled by default.
Browser MJPEG and the original Jetson H.264 sender remain alternate build modes.

## Browser H.264 mode (default)

The camera's native YUV420 frames go directly to the P4 hardware H.264 encoder.
The ESP packages the resulting H.264 access units as fragmented MP4 and serves
them over HTTP. Its embedded browser player uses MediaSource and a video element:

    http://<esp32-ip>/

No Jetson, cloud service, browser extension, or WebCodecs/HTTPS setup is needed.
Use a browser with H.264-in-MP4 MediaSource support, such as a supported desktop
Chrome/Edge build. The player checks the exact codec from the encoder's SPS and
shows an error if the browser cannot decode it. It uses no CDN assets.

The defaults are native **1920x1080**, a **30 FPS** capture target, **4 Mbps**
H.264 target bitrate, three camera buffers, two encoded buffers, and a 32 KiB
TCP send buffer. Actual sent/displayed FPS still depends on Wi-Fi and decoding.
There is no CPU resize or JPEG re-encoding. The MJPEG width/height settings are
ignored in this mode; changing the sensor's native mode changes H.264 resolution.

Click **Start**/**Stop** to control playback. One browser viewer is supported at
a time; stop/close the current player before opening another. The HTTP server
handles the long-lived stream synchronously, so other requests can wait while
it is active. Stopping/reconnecting starts a fresh encoder and IDR sequence.
Encoded dependent frames are not dropped during a session. The browser bounds
its playback buffer and catches up to the live edge if it falls behind.

While streaming, serial logs print `H264 stats` every five seconds: encoded/sent
frame counts, measured sent FPS and bitrate, encode/mux/send times, frame age,
and queue depth. The page also shows browser playback statistics when available.
These measurements are separate from the configured camera FPS. `/stream.mp4`
is the continuous fragmented-MP4 endpoint consumed by the player.

Both browser modes keep Wi-Fi modem sleep disabled (higher power consumption)
and enable TCP_NODELAY. Keep the board and browser on the same trusted LAN.
The HTTP endpoint has no authentication or encryption; do not expose it publicly.

## Browser MJPEG fallback

The MJPEG firmware captures RGB565 frames at the sensor's native 1920x1080,
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
   - Under **Streaming mode**, select **Serve browser H.264 stream (native
     capture)** for direct H.264 playback (the default).
   - Adjust the HTTP port, H.264 target bitrate, QP bounds, and frame rate if needed.
   - To restore MJPEG, select **Serve browser MJPEG stream**. Set **Frame width
     (MJPEG output / Jetson capture)** to **1280** and **Frame height (MJPEG
     output / Jetson capture)** to **720** for its resized 720p output.

The checked-in defaults already select the Waveshare board's ESP32-C6 over
4-bit SDIO, its GPIO wiring, 32 MB flash, 32 MB PSRAM at 200 MHz, the OV5647
MIPI-CSI sensor, the ISP pipeline, and the ESP32-P4 hardware video support.

The MJPEG fallback performance defaults are:

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

Open logs without flashing with `idf.py -p <PORT> monitor`; exit with Ctrl+].

## Host checks

From `esp32-p4`, with GCC and Node.js available:

    gcc -std=c11 -O2 -Wall -Wextra -Werror -I main main/h264_mp4.c tests/test_h264_mp4.c -o build/test_h264_mp4.exe
    ./build/test_h264_mp4.exe
    node --test tests/test_h264_player.mjs

The MP4 checks cover Annex-B framing, SPS/PPS, init/fragment boxes, keyframe
flags, timestamps, and output-buffer bounds. The player checks cover its live
buffer policy and asynchronous append/cancellation behavior. A camera and a
flashed board are still required to measure end-to-end FPS.

## Jetson H.264 mode

To restore the original Jetson sender, select **Send H.264 over TCP to Jetson**
under **Camera Streaming Configuration -> Streaming mode**. Configure the Jetson
IP, TCP port, H.264 settings, and stream button. Use the sensor's native capture
dimensions (normally 1920x1080); this sender has no 720p resize stage.

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
