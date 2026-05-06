# ESP32 -> Jetson Streaming Runbook (JPEG over TCP)

This runbook is the canonical bring-up sequence for the current default path:

- ESP32 firmware: `ov5647_capture` in TCP mode
- Jetson receiver: `jetson-code/main.py` with `PAYLOAD_FORMAT=jpeg` (default)
- Framing protocol: `[len:u32be][jpeg_payload]...`

## 1) Configure ESP32 firmware

### 1.1 Set Jetson endpoint

Edit `ov5647_capture/main/ov5647_capture.c`:

- `JETSON_TCP_IP` = Jetson hotspot/LAN IPv4 (e.g. `10.42.0.1`)
- `JETSON_TCP_PORT` = `5000`

### 1.2 Set Wi-Fi credentials locally (do not commit real secrets)

```bash
cd ov5647_capture
idf.py menuconfig
```

Open:

- `TCP streaming Wi-Fi (ESP32-S3 STA)`

Set:

- `Remote Wi-Fi SSID`
- `Remote Wi-Fi Password`
- `Remote Wi-Fi maximum retry count`

After saving and rebuilding, you can verify resolved values in:

- `ov5647_capture/build/config/sdkconfig.json`

## 2) Start Jetson listener first

### 2.1 Full pipeline

```bash
cd jetson-code
MODEL_PATH=/absolute/path/to/jetson-code/models/action.h5 python3 main.py
```

Expected startup logs:

- `[tcp] listening on 0.0.0.0:5000`
- `[main] listening 0.0.0.0:5000`

### 2.2 Confirm port is open

```bash
ss -ltnp | grep ':5000'
```

Expected:

- `LISTEN ... 0.0.0.0:5000 ... users:(("python3",...))`

## 3) Build/flash ESP32

```bash
cd ov5647_capture
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash monitor
```

## 4) Expected healthy runtime signals

### Jetson

- `[tcp] client connected: (...)`
- payload decode/inference activity
- eventual `[gloss] ...` lines once sequence/stability thresholds are met

### ESP32

- Wi-Fi association succeeds (no retry exhaustion)
- TCP connect succeeds
- periodic camera fps/bitrate logs

## 5) Failure map (quick diagnosis)

### A) `Wi-Fi retry connect (attempt ...)` then `Wi-Fi failed to connect`

Cause:

- ESP32 did not join AP.

Check:

- SSID/password in menuconfig
- AP is active and 2.4 GHz capable
- credentials were reflashed after change

### B) `TCP connect failed, retrying...`

Cause:

- Wi-Fi connected but Jetson endpoint unreachable/not listening/wrong IP.

Check:

- `JETSON_TCP_IP` matches Jetson `ip -4 addr` on the AP/LAN interface
- Jetson `main.py` is running before ESP32 starts
- `ss -ltnp | grep ':5000'` shows python listening

### C) Repeated `cam_hal: FB-OVF`

Cause:

- Camera buffers overflow while network path is blocked or too slow.

Check:

- resolve TCP connectivity first
- if still present under healthy link, reduce stream load (FPS and/or JPEG quality/resolution)

### D) `cam_hal: NO-SOI - JPEG start marker missing` (sporadic)

Cause:

- bad/corrupted frame from capture path; often intermittent.

Check:

- camera ribbon seating, stable power, conservative camera settings
- treat as secondary unless persistent/high-rate

