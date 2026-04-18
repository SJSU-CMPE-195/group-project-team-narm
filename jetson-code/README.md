# Jetson Orin Nano Super ASL Glasses Pipeline (PoC)

This folder contains the Jetson-side code for the ESP32 glasses project.

## Stream protocol (matches ESP32 `tryout`)

Single TCP connection. Repeated messages:

- 4-byte big-endian unsigned length
- followed by exactly `length` bytes of H.264 payload (one access unit)

## Pipeline overview

ESP32 sends length-prefixed H.264 access units over TCP. The receiver decodes frames, runs MediaPipe Holistic for keypoints, runs the Keras LSTM for gloss prediction, and prints stable predictions to stdout.

Until the ESP32 connects and sends frames, `main.py` blocks waiting on an internal queue—no `[gloss]` lines will appear.

```mermaid
flowchart LR
  esp32[ESP32_tcp_sender] -->|len_u32be_payload| jetsonTcp[jetson_tcp_ingest]
  jetsonTcp --> decode[decode_h264_PyAV]
  decode --> mp[MediaPipe_holistic]
  mp --> lstm[TensorFlow_Keras_LSTM]
  lstm --> stdout[gloss_print_stdout]
```

## Documented runtime: container + Jetson AI Lab wheels (what we use)

This README matches the setup that was validated on **JetPack 6** using:

- A **container** (or dev environment) with the repo mounted at **`/workspace/group-project-team-narm`**
- Python **3.10** and packages under **`/opt/venv`** (typical for NVIDIA/Jetson dev images)
- Pip using the **`https://pypi.jetson-ai-lab.io/jp6/cu126`** wheel index (shown in successful installs as “Looking in indexes: …jp6/cu126”)

This is **not** the same as creating a **venv on the bare Jetson host** (`python3 -m venv` outside Docker); if you install on the host only, you may need different wheels—use the same pins only if your platform provides matching binaries.

### First-time setup (create the container)

This creates a **persistent** named container `asl_infer` (no `--rm`). After this one-time step, you can follow the **After reboot** section (`docker start -ai asl_infer`) to re-enter the same environment.

```bash
# On the Jetson host (outside the container)
cd ~/group-project-team-narm

docker run -it --name asl_infer --net=host \
  -v "$HOME/group-project-team-narm:/workspace/group-project-team-narm" \
  tensorflow2:2.21.0-r36.4.tegra-aarch64-cu126-22.04-tensorflow2_2.21.0 \
  bash
```

Then run the rest of the steps in this README **inside** the container (pip installs + `MODEL_PATH=... python3 main.py`).

**Working directory to run the app:**

```text
/workspace/group-project-team-narm/jetson-code
```

### Git on a bind-mounted repo

If Git refuses commands with **“detected dubious ownership”**:

```bash
git config --global --add safe.directory /workspace/group-project-team-narm
```

Pull updates as usual:

```bash
cd /workspace/group-project-team-narm
git pull origin main
```

### System packages (apt)

Headless OpenCV wheels may still load native libraries such as **`libxcb.so.1`**. On Ubuntu/jammy-style images:

```bash
apt-get update
apt-get install -y libxcb1
```

- If you see **`libGL.so.1` missing** and you used full **`opencv-python`**, either install `libgl1` **or** (recommended) switch to **`opencv-python-headless`** at the version pinned below so you avoid pulling GUI/OpenGL stacks into minimal containers.

## Known-good Python stack (pinned)

These versions work together for **TensorFlow 2.15.1 + MediaPipe 0.10.x + NumPy 1.26.x** when using the Jetson AI Lab index on **aarch64**:

| Package | Version | Notes |
| --- | --- | --- |
| `numpy` | `1.26.4` | TF 2.15.1 requires `<2.0` in practice here. |
| `protobuf` | `4.25.9` | TF 2.15.x expects protobuf `<5`. **Do not** pair this stack with TF 2.21 + protobuf 6 in the same env as MediaPipe 0.10.x (protobuf/runtime errors). |
| `tensorflow` | `2.15.1` | On aarch64 Jetson index, this may pull **`tensorflow-cpu-aws==2.15.1`**. |
| `opencv-python-headless` | `4.10.0.84` | Avoid `opencv-python` **4.13+** with `numpy==1.26.4` (OpenCV 4.13 wants `numpy>=2`, which conflicts with this stack). |
| `matplotlib` | `>=3.8` (binary wheels) | Install **before** resolving MediaPipe if you use full `pip install mediapipe` (see pitfalls). **`3.10.8`** was verified. |
| `mediapipe` | `0.10.18` | Install with **`--no-deps`** after matplotlib + TF to avoid pip pulling an old matplotlib **sdist** (see pitfalls). |
| `attrs`, `sentencepiece`, `sounddevice` | (latest compatible from index) | Needed for MediaPipe imports when installed with `--no-deps`. **`jax` / `jaxlib`** were **not** required for a successful `import mediapipe` in this setup. |
| `av` | `17.0.0` | PyAV for H.264 decode in `decode_h264.py`. |

See [requirements.txt](requirements.txt) for the same pins in machine-readable form (follow the header there—**MediaPipe is a separate step** with `--no-deps`).

### Pitfalls (from real installs)

1. **`pip install mediapipe` (without prep)** can try to build an old **matplotlib** from source and fail with `platform.linux_distribution` on Python 3.10. Fix: install **binary matplotlib first** (`matplotlib>=3.8 --only-binary=:all:`), **or** install **`mediapipe==0.10.18 --no-deps`** and add deps manually (see order below).
2. Mixing **TF 2.21 + protobuf 6** with **MediaPipe 0.10.x** caused protobuf / `RegisterExtension` / `GetPrototype` class of errors in troubleshooting; use the **TF 2.15.1 + protobuf 4.25.x** stack here instead.
3. After changing TensorFlow versions, **`pip`** may warn about **`onnx`** vs **`ml_dtypes`**—harmless unless you rely on ONNX in this environment.

## Exact install order (copy/paste)

Run inside the container from **`/workspace/group-project-team-narm/jetson-code`**. Use **`python3`** consistently with your image.

**1. Tooling (optional but used in practice)**

```bash
python3 -m pip install -U pip setuptools wheel
```

**2. Binary matplotlib first**

```bash
python3 -m pip install "matplotlib>=3.8" --only-binary=:all:
```

**3. Core pins**

```bash
python3 -m pip install "numpy==1.26.4" "protobuf==4.25.9"
```

**4. OpenCV headless**

Uninstall a conflicting full `opencv-python` if present, then:

```bash
python3 -m pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
python3 -m pip install "opencv-python-headless==4.10.0.84"
```

**5. TensorFlow**

```bash
python3 -m pip install "tensorflow==2.15.1"
```

**6. MediaPipe without pulling broken matplotlib deps**

```bash
python3 -m pip install "mediapipe==0.10.18" --no-deps
python3 -m pip install attrs sentencepiece sounddevice
```

**7. PyAV**

```bash
python3 -m pip install "av==17.0.0"
```

**Alternative:** after step 2 (binary matplotlib), you can install most pinned packages in one shot, then MediaPipe:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install "mediapipe==0.10.18" --no-deps
```

(`requirements.txt` omits a plain `mediapipe` line so `pip` does not resolve it with full dependencies.)

### Verification (should all succeed)

```bash
python3 -c "import cv2, numpy as np; print('cv2', cv2.__version__, 'numpy', np.__version__)"
python3 -c "import google.protobuf; print('protobuf', google.protobuf.__version__)"
python3 -c "import tensorflow as tf; print('tf', tf.__version__)"
python3 -c "import mediapipe as mp; print('mediapipe', mp.__version__)"
python3 -c "import av; print('av', av.__version__)"
```

## Model file and `MODEL_PATH`

**Supported default for this stack:** `jetson-code/models/action.h5` (loads with **TensorFlow 2.15.1 / Keras 2.15** as used above).

Other `.h5` files under `models/` are not validated for this README’s pinned environment; they may require different TensorFlow/Keras versions.

Set the absolute path inside the container, e.g.:

```bash
export MODEL_PATH=/workspace/group-project-team-narm/jetson-code/models/action.h5
```

## Wi-Fi / networking (ESP32)

Ensure the Jetson is the Wi‑Fi access point (AP) if that is your project topology: the ESP32 connects to the Jetson AP and opens a TCP connection to the Jetson. Default receiver port is **`5000`** (matches ESP32 project config).

## Environment variables

**Required:**

- **`MODEL_PATH`** — path to the trained LSTM model (use `models/action.h5` for the documented stack)

**Optional:**

- **`HOST`** (default `0.0.0.0`)
- **`PORT`** (default `5000`)
- **`SEQUENCE_LENGTH`** (default `30`)
- **`STABLE_N`** (default `10`)
- **`THRESHOLD`** (default `0.5`)
- **`ACTIONS`** — comma-separated class names (default `hello,thanks,iloveyou`)

## Run

From `jetson-code/`:

```bash
cd /workspace/group-project-team-narm/jetson-code
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
MODEL_PATH=/workspace/group-project-team-narm/jetson-code/models/action.h5 \
python3 main.py
```

The process **does not exit** on its own: it listens for TCP and blocks on an internal queue until H.264 payloads arrive. **Ctrl+C** may interrupt while waiting (e.g. `KeyboardInterrupt` in `queue.get`)—expected when stopping the server.

## After reboot (persistent container: `asl_infer`)

If you are using the named container **`asl_infer`** (validated workflow), dependencies installed inside it will persist across reboots. You only need to reinstall dependencies if you delete/recreate the container.

On the Jetson host:

```bash
docker ps -a | grep asl_infer
docker start -ai asl_infer
```

Inside the container:

```bash
cd /workspace/group-project-team-narm/jetson-code
# One-time (per container): avoid TensorFlow import failures when protobuf C++ backend isn't available.
echo 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' >> ~/.bashrc
source ~/.bashrc

python3 -c "import tensorflow, mediapipe, cv2, av; print('deps ok')"
MODEL_PATH=/workspace/group-project-team-narm/jetson-code/models/action.h5 python3 main.py
```

## Expected output

When the server starts:

- `[tcp] listening on 0.0.0.0:5000`
- `[main] listening 0.0.0.0:5000`

When the ESP32 connects:

- `[tcp] client connected: ('<ip>', <port>)`

When predictions begin (after the rolling window fills and stability/threshold pass):

- `[gloss] <label> (conf=0.xxx)`

You may see **non-fatal** messages such as:

- `Error in cpuinfo: prctl(PR_SVE_GET_VL) failed` (common in containers)
- TensorFlow Lite / `inference_feedback_manager` warnings during MediaPipe initialization

## Troubleshooting (quick)

| Symptom | What to try |
| --- | --- |
| matplotlib `linux_distribution` / failed to build wheel | Install `matplotlib>=3.8` with `--only-binary=:all:` first, or install `mediapipe --no-deps` as above. |
| `libxcb.so.1` missing | `apt-get install -y libxcb1` |
| `libGL.so.1` missing | Prefer `opencv-python-headless==4.10.0.84` instead of full `opencv-python`, or install `libgl1`. |
| MediaPipe + protobuf errors with TF 2.21 | Use **TF 2.15.1** + **protobuf 4.25.9** stack documented here; do not mix TF 2.21 + protobuf 6 with MediaPipe 0.10.x in one env. |
| `import tensorflow` fails with protobuf `_message` / “Selected implementation cpp is not available” | Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (one-off prefix or add `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to `~/.bashrc` inside the container). |
| No gloss output, terminal looks idle | Expected until the ESP32 sends framed H.264 to the open port. |

## Notes

- This PoC prints stable predicted gloss tokens to stdout (no gloss→sentence and no TTS).
- The TCP stream protocol is length-prefixed H.264 access units: `[len:u32be][payload]*` (matches ESP32 sender).
