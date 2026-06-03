# Raspberry Pi Camera 3 JPEG Sender

This sender keeps the same wire protocol your Jetson already expects:

- Transport: TCP
- Framing: `[4-byte big-endian payload length][JPEG payload bytes]`
- Default destination: `10.42.0.1:5000`

## 1) Raspberry Pi setup

Use Raspberry Pi OS (Bookworm recommended), then install dependencies:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-pil
```

Optional checks:

```bash
libcamera-hello -t 2000
python3 -c "from picamera2 import Picamera2; print('picamera2 ok')"
```

## 2) Run Jetson side first

Inside your existing Jetson container/session, keep JPEG mode enabled:

```bash
export PAYLOAD_FORMAT=jpeg
export PREVIEW_HTTP=1
export PREVIEW_PORT=8000
python3 main.py
```

You should see:

- `[tcp] listening on 0.0.0.0:5000`

## 3) Run Raspberry Pi sender

On the Raspberry Pi:

```bash
cd ~/group-project-team-narm/pi-code
python3 pi_jpeg_sender.py --host 10.42.0.1 --port 5000 --width 640 --height 480 --fps 10 --jpeg-quality 78
```

Expected sender logs:

- `[pi] connected`
- `[pi] camera ok: fps=... avg_frame=... bitrate=...`

Expected Jetson logs:

- `[tcp] client connected`
- `[gloss] ...` predictions after enough frames

## 4) Tuning knobs

You can pass CLI flags or env vars:

- `JETSON_HOST` / `--host`
- `JETSON_PORT` / `--port`
- `WIDTH` / `--width`
- `HEIGHT` / `--height`
- `FPS` / `--fps`
- `JPEG_QUALITY` / `--jpeg-quality`
- `AF_CONTINUOUS` / `--af-continuous`

Suggested stable demo baseline:

- `640x480`
- `10 FPS`
- `JPEG_QUALITY=75..80`

## 5) Troubleshooting

- If Pi says `connect failed`:
  - Verify Jetson hotspot is up and Pi is connected to it.
  - Verify Jetson IP (often `10.42.0.1`) and port `5000`.
- If Jetson says decode failures:
  - Lower FPS (`8`) and/or lower quality number effect? Note: in JPEG, higher quality value means larger frames.
  - Try `--jpeg-quality 70`.
- If stream is choppy:
  - Reduce resolution to `640x480` and/or `FPS=8`.
