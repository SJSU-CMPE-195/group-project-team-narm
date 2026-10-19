# Colab ASL data collection notebooks

One Google Colab notebook per word. Each records takes from your laptop webcam in the
browser, runs MediaPipe Holistic on every take, lets you replay it and mark it Good or
Bad, and saves the approved takes for training the new model.

```
training/collection/
  make_notebooks.py        generator (run this to (re)build the notebooks)
  template/helpers.py      Python side: MediaPipe, saving, Colab callbacks
  template/app.js          browser UI: flash, countdown, recording, review/replay
  notebooks/collect_<word>.ipynb   generated output
```

Never hand-edit a generated notebook. Change `template/` and regenerate:

```bash
python make_notebooks.py --word back        # one notebook (the test notebook)
python make_notebooks.py --all              # all words in WORDS
```

## Using a notebook

1. Open Colab (desktop Chrome), choose **File > Upload notebook**, pick `notebooks/collect_<word>.ipynb`
   (or open it from GitHub once it is pushed).
2. **Set the runtime version first:** Runtime > Change runtime type > **Runtime version** > **2025.07**
   (Python 3.11) > Save. The default "Latest" runtime is Python 3.13, where `mediapipe==0.10.18` cannot install.
   Then run **1. Install** (about a minute) and do **Runtime > Restart session**. Do not re-run the install cell.
3. In **2. Config** type your first name into `PERSON` (lowercase). Leave everything else alone.
4. Run **Setup** (it mounts Google Drive, click Allow), then **3. Record takes**, and allow the camera.
5. Per take: the word flashes, 5-4-3-2-1 countdown, then recording with a `REC n/90` counter in the corner.
6. Review: replay with Pause, 1x / 0.5x / 0.25x, a scrub slider, and an optional skeleton overlay.
   The panel also shows the share of frames where each hand, the pose and the face were detected, and
   warns if hands were visible in too few frames or the capture stuttered.
7. **Good - save** keeps the take and starts the next one. **Bad - retry** discards it and redoes the
   same take. **Stop session** ends early (the unreviewed take is discarded).
8. It stops at `TAKES` good takes (default 30). Re-running the cell resumes where you left off.
9. Run **4. Summary** to check the result (tick `DOWNLOAD_ZIP` to also get a zip).

## Capture spec (identical for the whole team)

| Setting | Value | Why |
| --- | --- | --- |
| Resolution | 640x360 (16:9) | matches the glasses stream aspect ratio; center-cropped, never stretched |
| Capture rate | 30 fps | derive 15 fps (`[::2]`) or 10 fps (`[::3]`) later without re-recording |
| Clip length | 3.0 s = 90 frames | signs run roughly 1.5-3.5 s |
| MediaPipe | Holistic, `model_complexity=0`, confidences 0.5 | same as `jetson-code/holistic_lstm_infer.py` |
| Features | 1662 values per frame | `extract_keypoints` is the same function as the Jetson's |

Saved frames are never mirrored (`MIRROR_PREVIEW` only affects the live preview).

## What gets saved

Per good take, under `<Drive>/ASL_collection/<word>/` (or `/content/asl_collection_output` if Drive is off):

- `<word>_<person>_<NNN>.mp4`: the 640x360, 30 fps clip (keeps you able to re-extract keypoints later)
- `<word>_<person>_<NNN>.npy`: keypoints, shape `(90, 1662)`, float32
- one row in `manifest.csv` (next to the word folders): word, person, take, fps, frames, hand/pose/face
  coverage, measured capture fps, repeated-frame count, mediapipe version, timestamp

To bring takes into the repo, copy them into `training/new_videos/<word>/` (one folder per word already
exists there). The old dataset in `training/raw_videos/` is kept separately and is never touched by the notebooks.

Each take is saved the moment you click Good, so a Colab disconnect only loses the take in progress.
If you run without Drive, download the zip from step 9 before closing the tab.

## Merging everyone's data

Filenames include the person, so folders can be combined without collisions:

1. Copy every teammate's `<word>/` folders into one shared folder.
2. Combine manifests (each has its own header):

```python
import glob, pandas as pd
pd.concat([pd.read_csv(p) for p in glob.glob("*/manifest.csv")]).to_csv("manifest_all.csv", index=False)
```

Turning a take into a training sample (example: 15 fps x 45 frames):

```python
import numpy as np
kp = np.load("back/back_newton_001.npy")   # (90, 1662) at 30 fps
sample = kp[::2]                            # (45, 1662) at 15 fps
```

Split train/test by **person**, not by take, to get an honest accuracy number.

## Notes and limits

- The Python side was tested against real MediaPipe 0.10.18 (the Jetson's version) using real video
  frames, and the browser UI was exercised end to end in a simulated browser. It has not yet been run
  in a live Colab session with a real webcam, so use the single test notebook first.
- Colab cannot use `cv2.VideoCapture(0)`, so capture runs in the browser and frames are uploaded to
  Python after each take (a few seconds per take).
- Words come from the folders in `training/raw_videos/`. `upper` and `lower` were in the original
  34-word list but have no folders yet; add them to `WORDS` in `make_notebooks.py` if they are kept.
- Newer mediapipe builds (0.10.30+, 1.x) lack `mp.solutions`, and 0.10.18 has no Python 3.13 wheel, so the
  notebooks need the Colab **2025.07** runtime. Do not switch to the MediaPipe Tasks API: the keypoints must
  match the Jetson's `mp.solutions.holistic`.

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `RuntimeError: This Colab runtime uses Python 3.13 ...` | runtime is on "Latest" | Runtime > Change runtime type > Runtime version > 2025.07 > Save, re-run **1. Install** |
| pip `No matching distribution found for mediapipe==0.10.18` | same as above (Python too new) | same as above |
| `RuntimeError: ... no legacy mp.solutions.holistic` | wrong mediapipe installed, or session not restarted | switch to 2025.07, re-run **1. Install**, then Runtime > Restart session |
| pip "dependency resolver ... incompatible" warnings | other preinstalled Colab packages disagree on numpy/protobuf | harmless; ignore, then restart the session |
| `ValueError: Set PERSON ...` | `PERSON` is empty | type your first name (lowercase), re-run Config |
| no camera preview | not desktop Chrome, permission denied, or webcam in use | use desktop Chrome, click Allow, close other apps using the camera |
| everything vanished after a restart | restart clears Python state | skip **1. Install**; re-run Config, Setup, then Record (saved takes are on Drive and resume) |

Local editors (Cursor/VS Code) cannot run these notebooks: the camera UI needs Colab's browser runtime.
