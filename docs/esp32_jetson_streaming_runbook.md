# ESP32-P4 to Jetson Streaming Runbook

This is the canonical bring-up path:

- Camera: OV5647 over MIPI-CSI
- Sender: Waveshare ESP32-P4-WIFI6 in esp32-p4
- Codec: ESP32-P4 hardware H.264
- Transport: TCP through the board's ESP32-C6 Wi-Fi coprocessor
- Framing: 4-byte big-endian length followed by one H.264 access unit
- Receiver: jetson-code/main.py with PAYLOAD_FORMAT=h264

The previous XIAO ESP32-S3 JPEG firmware remains in ov5647_capture only as
legacy reference and is not the project target.

## 1. Configure the Jetson network

ESP32-P4 and Jetson must be on the same LAN. For an offline demo, the Jetson can
host a NetworkManager hotspot:

    nmcli device status
    sudo nmcli device wifi hotspot ifname IFACE con-name ASLHotspot \
      ssid ASLHotspot password 'YourStrongPassphrase'
    ip -4 addr show dev IFACE

The hotspot address is commonly 10.42.0.1, but use the address actually shown.

## 2. Start the Jetson receiver first

Inside the prepared Jetson container:

    cd /workspace/group-project-team-narm/jetson-code
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PAYLOAD_FORMAT=h264 \
    MODEL_PATH=/workspace/group-project-team-narm/jetson-code/models/action.h5 \
    PREVIEW_HTTP=1 \
    python3 main.py

Expected output:

    [tcp] listening on 0.0.0.0:5000
    [main] listening 0.0.0.0:5000, payload=h264, ...

Confirm the port if necessary:

    ss -ltnp | grep ':5000'

## 3. Configure the ESP32-P4

Use ESP-IDF 5.4 or newer:

    cd esp32-p4
    idf.py set-target esp32p4
    idf.py menuconfig

Configure:

- Example Connection Configuration: SSID `ASLHotspot`, password
  `YourStrongPassphrase`.
- Example Video Initialization Configuration: keep the working MIPI-CSI,
  OV5647, and Waveshare SCCB/control settings.
- Camera Streaming Configuration -> Streaming mode: **Send H.264 over TCP to Jetson**.
- Jetson IP `10.42.0.1`, TCP port `5000`.
- Stream toggle button GPIO `35`: the Waveshare ESP32-P4-WIFI6 onboard BOOT
  button, confirmed in the [board schematic, page 1](https://files.waveshare.com/wiki/ESP32-P4-WIFI6/ESP32-P4-WIFI6-datasheet.pdf).
  Release BOOT during reset; use it after the firmware prints `ready`.
  For an external momentary switch, use a free configured GPIO and GND.
  The firmware enables the internal pull-up and detects an active-low press.
- Enable **Start streaming automatically after Wi-Fi connects**.
- Frame width `1920`, height `1080`, FPS `15`. The sender uses native capture
  without resizing. Start with 4 Mbps, I-frame period 30, QP 25 to 35.
  The sensor remains in its 1080p30 mode; the driver delivers every other frame
  to encoding and streaming. Use the Jetson default `FRAME_STRIDE=1` so the model
  receives the same 15 FPS timing used in training.

These demo values are in `esp32-p4/sdkconfig.defaults`; existing `sdkconfig`
settings take precedence, so check them before rebuilding. Avoid repeating
`set-target` for an already configured ESP32-P4 because it resets local settings.

## 4. Build, flash, and start

    idf.py build
    idf.py -p <PORT> flash monitor

After Wi-Fi connects, streaming starts automatically. Expected ESP logs:

    Wi-Fi modem sleep disabled (WIFI_PS_NONE)
    auto-starting camera stream

    connected to Jetson at 10.42.0.1:5000
    video started
    stats: encoded=... sent=... fps=... bitrate=... queue=... age=...ms

The demo target is roughly 15 measured FPS. Press BOOT to stop
capture and close the connection; expect `stream stopped`. Press again to
reconnect. Measure this on the board; a successful build does not verify FPS.

Expected Jetson logs:

    [tcp] client connected: (...)
    [stats] rx=... decode=... infer=... latency=...ms
    [gloss] <label> (conf=...)

## 5. Interpret performance counters

| Observation | Likely bottleneck |
| --- | --- |
| ESP encoded FPS below target | OV5647 mode, ISP, encoder, or camera configuration |
| ESP queue/age keeps increasing | Wi-Fi or TCP throughput |
| Jetson receive FPS below ESP sent FPS | Network or receiver backpressure |
| Jetson decode FPS below receive FPS | H.264 decoder |
| Decode FPS is healthy but inference FPS is low | MediaPipe or LSTM inference |
| Decoded-drop count rises | Inference is slower than video; latency remains bounded |

The Jetson intentionally applies TCP backpressure before H.264 decoding. It
never discards compressed P-frames. When inference falls behind, it drops only
fully decoded RGB frames.

## 6. Baseline tuning

Start with:

- 1920x1080 at 15 FPS
- 4 Mbps
- I-frame period 30
- QP range 25 to 35

If hand/finger detail is insufficient, increase bitrate before increasing
resolution. If inference is slow, resize/crop on the Jetson while keeping the
ESP32 sender at the sensor's native 1920x1080 capture size.

## Troubleshooting

### ESP cannot join Wi-Fi

- Verify SSID/password in menuconfig.
- Use a 2.4 GHz-capable hotspot.
- Rebuild and flash after changing credentials.

### TCP repeatedly reconnects

- Confirm the Jetson IP and port.
- Start main.py before powering/resetting the board, or let the sender retry
  until the receiver is listening.
- Check firewall rules and the ESP TCP send-timeout setting.

If `connected to Jetson` and `video started` are followed by `no valid buffer`
and `capture/encode failed`, the connection succeeded but the ESP encoder ran
out of queued output buffers. Rebuild and flash the firmware with the encoded
buffer availability guard. It waits for the sender to return a buffer before
encoding another frame, keeping all H.264 access units in order even when the
Jetson or TCP sender is slower than capture.

Once the ESP stream stays connected, compare its `stats:` FPS with the Jetson's
`[stats]` receive/decode/inference counters. The `main.py` browser preview is
updated by the inference loop, so a slow inference rate also slows the preview.
Both Jetson MJPEG preview implementations additionally wait 50 ms between JPEG
responses; browser preview FPS is separate from ESP encoded/sent FPS.

### H.264 decoder errors after connect

- Confirm PAYLOAD_FORMAT=h264.
- Keep the I-frame period near 30 while debugging.
- Verify the sender starts a fresh encoder session after reconnect.
- Do not add queue logic that discards encoded access units.

### Camera or encoder fails to start

- Verify OV5647 ribbon orientation and power.
- Confirm the MIPI-CSI/SCCB configuration against the Waveshare schematic.
- Confirm the P4 hardware H.264 device is enabled in sdkconfig.
