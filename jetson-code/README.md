# Jetson Orin Nano Super ASL Glasses Pipeline (PoC)

This folder contains the Jetson-side code for the ESP32 glasses project.

## Stream protocol (matches ESP32 `tryout`)

Single TCP connection. Repeated messages:

- 4-byte big-endian unsigned length
- followed by exactly `length` bytes of H.264 payload (one access unit)

## Quick start (PoC: predict glosses)

There are two ways to run this on JetPack 6.2:

- **Recommended (Docker):** use NVIDIA's TensorFlow container (avoids TensorFlow pip wheel issues on JP 6.2).
- **Alternative (venv):** install TensorFlow separately using NVIDIA's Jetson instructions, then install the rest from `requirements.txt`.

2. Ensure the Jetson is the Wi‑Fi access point (AP)

- ESP32 connects to the Jetson AP, then opens a TCP connection to the Jetson.
- Default receiver port is `5000` (matches ESP32 project config).

3. Set required env vars:

   - `MODEL_PATH` = path to your trained LSTM model file (e.g. `action.h5` or `action.keras`)

4. Optional env vars:

   - `HOST` (default `0.0.0.0`)
   - `PORT` (default `5000`)
   - `SEQUENCE_LENGTH` (default `30`)
   - `STABLE_N` (default `10`)
   - `THRESHOLD` (default `0.5`)
   - `ACTIONS` (comma-separated class names; default `hello,thanks,iloveyou`)

5. Run:

   - `python main.py`

## Docker run (recommended on JetPack 6.2)

1. Ensure Docker works:

- `docker --version`

2. Put this repo folder and your model on the Jetson, for example:

- `~/jetson-code/` contains `main.py`, etc.
- `~/jetson-code/action.h5` (or `action.keras`)

3. Start an NVIDIA TensorFlow container and mount your code:

```bash
docker run --rm -it --net=host --runtime=nvidia \
  -v "$HOME/jetson-code:/workspace/jetson-code" \
  -w /workspace/jetson-code \
  nvcr.io/nvidia/tensorflow:25.02-tf2-py3 \
  bash
```

If `--runtime=nvidia` doesn't work on your setup, try `--gpus all` instead.

4. Inside the container, install the remaining Python deps and run:

```bash
pip install -r requirements.txt

export MODEL_PATH=/workspace/jetson-code/action.h5
export HOST=0.0.0.0
export PORT=5000

python main.py
```

## Expected output

When the server starts:

- `[tcp] listening on 0.0.0.0:5000`
- `[main] listening 0.0.0.0:5000`

When ESP32 connects:

- `[tcp] client connected: ('<ip>', <port>)`

When predictions begin (after the rolling window fills and stability/threshold pass):

- `[gloss] <label> (conf=0.xxx)`

## Notes

- This PoC prints stable predicted gloss tokens to stdout (no gloss→sentence and no TTS).
- The TCP stream protocol is length-prefixed H.264 access units: `[len:u32be][payload]*` (matches ESP32 sender).

