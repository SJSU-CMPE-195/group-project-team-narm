# Jetson Orin Nano Super ASL Glasses Pipeline

This folder contains the Jetson-side code for the ESP32 glasses project.

## Stream protocol (matches ESP32 `tryout`)

Single TCP connection. Repeated messages:

- 4-byte big-endian unsigned length
- followed by exactly `length` bytes of H.264 payload (one access unit)

## Quick start

1. Install deps:

   - `pip install -r requirements.txt`

2. Set required env vars:

   - `WLASL_ROOT` = path to your WLASL checkout (contains `code/`)
   - `I3D_CHECKPOINT_PATH` = path to your trained I3D checkpoint file

3. Optional env vars:

   - `HOST` (default `0.0.0.0`)
   - `PORT` (default `5000`)
   - `OLLAMA_BASE_URL` (default `http://localhost:11434`)
   - `OLLAMA_MODEL` (set to `gemma3:270m` to enable gloss→sentence)

4. Run:

   - `python main.py`

## WLASL I3D webcam demo (laptop / local test)

- Notebook: [modelwebcamtest.ipynb](modelwebcamtest.ipynb) — run in Jupyter; set `WLASL_ROOT` (environment variable or edit the first config cell) to your WLASL checkout so `code/I3D` and weights resolve. Same `WLASL_ROOT` idea as step 2 above.

