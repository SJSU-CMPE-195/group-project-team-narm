# ESP32-P4 OV5647 H.264 Sender

This is the canonical glasses-camera firmware. It targets the
**Waveshare ESP32-P4-WIFI6** with an **OV5647 MIPI-CSI camera**.

The firmware captures YUV420 frames, uses the ESP32-P4 hardware H.264 encoder,
and sends each complete encoded access unit to the Jetson using:

    [4-byte big-endian payload length][H.264 access unit]

Capture/encoding and TCP transmission run in separate FreeRTOS tasks. Two
MMAP'd encoder-output buffers absorb brief Wi-Fi stalls without copying or
dropping dependent H.264 frames.

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
   - Wi-Fi SSID and password for the Jetson hotspot/LAN (2.4 GHz).
2. **H.264 Stream Example Configuration**
   - Jetson IPv4 and TCP port.
   - Stream button GPIO.
   - Resolution, FPS, bitrate, GOP, and QP range.

The checked-in defaults already select the Waveshare board's ESP32-C6 over
4-bit SDIO, its GPIO wiring, 32 MB flash, 32 MB PSRAM at 200 MHz, the OV5647
MIPI-CSI sensor, the ISP pipeline, and the ESP32-P4 hardware H.264 device.

The performance defaults are:

- 1280x720
- 30 FPS
- 4 Mbps H.264
- One I-frame every 30 frames
- Four camera buffers and two encoded-output buffers

The driver prints the negotiated resolution at startup because the sensor may
select the nearest supported mode.

## Build and flash

Start the Jetson listener first, then:

    idf.py build
    idf.py -p <PORT> flash monitor

Press the configured active-low button once to start streaming and again to
stop. If TCP disconnects unexpectedly, the sender reconnects and starts a clean
encoder session so the Jetson receives fresh SPS/PPS/IDR data.

## Jetson command

From jetson-code:

    PAYLOAD_FORMAT=h264 \
    MODEL_PATH=/absolute/path/to/models/action.h5 \
    python3 main.py

## Performance output

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
