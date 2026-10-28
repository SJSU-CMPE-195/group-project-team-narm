# Colab ASL data collection notebooks (video only)

One Google Colab notebook per word. Each records takes from your laptop webcam in the browser,
lets you replay each take and mark it Good or Bad, and saves the approved takes as video clips.

The notebooks only record video. **Keypoints are not extracted here**: they are extracted later, from
these clips, with whichever extractor the model is trained on (and the Jetson runs). That keeps the
notebooks free of MediaPipe (nothing to install, no Python-version pinning) and means a change of
extractor, e.g. dropping face landmarks, never requires re-recording.

```
training/collection/
  make_notebooks.py        generator (run this to (re)build the notebooks)
  template/helpers.py      Python side: saving clips, capture check, Colab callbacks
  template/app.js          browser UI: flash, countdown, recording, review/replay
  notebooks/collect_<word>.ipynb   generated output
```

Never hand-edit a generated notebook. Change `template/` and regenerate:

```bash
python make_notebooks.py --word back        # one notebook
python make_notebooks.py --all              # all words in WORDS
```

## Before you record: check your hands

Run [`../hands_check.ipynb`](../hands_check.ipynb) once, in the place where you will record. It shows MediaPipe
dots on your fingers and a live percentage per hand, and saves nothing. Both hands should stay green while you
move the way you will when signing. (It is a separate notebook because it needs MediaPipe, which these
collection notebooks deliberately do not.)

## Using a notebook

1. Open Colab (desktop Chrome), choose **File > Upload notebook**, pick `notebooks/collect_<word>.ipynb`
   (or open it from GitHub). The default runtime is fine; there is nothing to install.
2. In **1. Config** type your first name into `PERSON` (lowercase). Leave everything else alone.
3. Run **Setup** (it mounts Google Drive, click Allow), then **2. Record takes**, and allow the camera.
4. Per take: the word flashes, 3-2-1 countdown, then recording with a `REC n/90` counter in the corner.
5. Review: replay with Pause, 1x / 0.5x / 0.25x and a scrub slider. Check that your head and both hands
   stayed in frame and that you signed once. It also warns if the capture stuttered (low fps or many
   repeated frames).
6. **Good - save** keeps the take and starts the next one. **Bad - retry** discards it and redoes the
   same take. **Stop session** ends early (the unreviewed take is discarded).
7. It stops at `TAKES` good takes (default 30). Re-running the cell resumes where you left off.
8. Run **3. Summary** to check the result (tick `DOWNLOAD_ZIP` to also get a zip).

## Capture spec (identical for the whole team)

| Setting | Value | Why |
| --- | --- | --- |
| Resolution | 640x360 (16:9) | matches the glasses stream aspect ratio; center-cropped, never stretched |
| Capture rate | 30 fps | derive 15 fps (`[::2]`) or 10 fps (`[::3]`) later without re-recording |
| Clip length | 3.0 s = 90 frames | signs run roughly 1.5-3.5 s |

Saved frames are never mirrored (`MIRROR_PREVIEW` only affects the live preview).

## What gets saved

Per good take, under `<Drive>/ASL_collection/<word>/` (or `/content/asl_collection_output` if Drive is off):

- `<word>_<person>_<NNN>.mp4`: the 640x360, 30 fps, 90-frame clip (`.avi` MJPG only if the mp4 writer is unavailable)
- one row in `manifest.csv` (next to the word folders): word, person, take, fps, frames, size, measured
  capture fps, longest frame gap, repeated-frame count, timestamp

The clip is re-opened after writing and a take is only reported as saved if it has the expected frame count.

To bring takes into the repo, copy them into `training/new_videos/<word>/` (one folder per word already
exists there). The old dataset in `training/raw_videos/` is kept separately and is never touched by the notebooks.

Each take is saved the moment you click Good, so a Colab disconnect only loses the take in progress.
If you run without Drive, download the zip from step 8 before closing the tab.

## Merging everyone's data

Filenames include the person, so folders can be combined without collisions:

1. Copy every teammate's `<word>/` folders into one shared folder.
2. Combine manifests (each has its own header):

```python
import glob, pandas as pd
pd.concat([pd.read_csv(p) for p in glob.glob("*/manifest.csv")]).to_csv("manifest_all.csv", index=False)
```

Split train/test by **person**, not by take, to get an honest accuracy number.

## Turning clips into training data

Run the extractor over the clips after collection. Use the same extractor code the Jetson uses, so the
training features and the live features cannot drift apart. Sample every 2nd frame for 15 fps:

```python
import cv2
cap = cv2.VideoCapture("back/back_newton_001.mp4")   # 90 frames at 30 fps
frames = []
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    frames.append(bgr)
frames = frames[::2]                                   # 45 frames at 15 fps
```

Checking hand visibility (e.g. flag clips where the hands are missing in most frames) belongs in that
extraction step, since the notebooks no longer run MediaPipe.

## Notes and limits

- The Python side was tested end to end against a simulated browser using real JPEG frames and real
  video writing. It has not yet been run in a live Colab session with a real webcam, so use one
  notebook first before handing the rest out.
- Colab cannot use `cv2.VideoCapture(0)`, so capture runs in the browser and frames are uploaded to
  Python after each take (a few seconds per take).
- Words come from the folders in `training/raw_videos/`. `upper` and `lower` were in the original
  34-word list but have no folders yet; add them to `WORDS` in `make_notebooks.py` if they are kept.
- If a Drive `manifest.csv` was written by the older MediaPipe version of these notebooks, new rows are
  appended under that older header (the extra columns stay empty), so nothing breaks.

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `ValueError: Set PERSON ...` | `PERSON` is empty | type your first name (lowercase), re-run Config |
| no camera preview | not desktop Chrome, permission denied, or webcam in use | use desktop Chrome, click Allow, close other apps using the camera |
| "Capture stuttered" warning | low fps or many repeated frames | close other camera apps, improve lighting, retry the take |
| `the written clip looks wrong ...` | the video writer produced a bad file | click Bad, retry; if it repeats, tell Newton |
| everything vanished after a restart | restart clears Python state | re-run Config, Setup, then Record (saved takes are on Drive and resume) |

Local editors (Cursor/VS Code) cannot run these notebooks: the camera UI needs Colab's browser runtime.
