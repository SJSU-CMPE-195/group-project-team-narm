# Python side of the Colab ASL data collection notebook (video only).
#
# make_notebooks.py inlines this file verbatim into the hidden "Setup" cell.
# It expects these names to already exist (defined in the visible config cell):
#   WORD, WORD_DISPLAY, PERSON, TAKES, FPS, CLIP_SECONDS, WIDTH, HEIGHT,
#   JPEG_QUALITY, COUNTDOWN_SECONDS, AUTO_CONTINUE, MIRROR_PREVIEW,
#   USE_DRIVE, DRIVE_FOLDER, LOCAL_FOLDER
#
# What it does:
#   - resolves the output folder (Google Drive if possible, local fallback)
#   - receives the frames of each take from the browser and checks the capture
#     timing (stutter / repeated frames)
#   - saves  <word>/<word>_<person>_<NNN>.mp4  and appends a row to manifest.csv
#   - exposes the callbacks the browser UI (app.js) calls via google.colab.output
#
# No MediaPipe here on purpose: keypoints are extracted later, from these videos,
# with whichever extractor the model is trained on. Only OpenCV and NumPy are
# needed, and both are preinstalled in Colab.

import base64
import csv
import datetime
import json
import os
import re
import shutil
import tempfile
import traceback
import zipfile

import cv2
import numpy as np

try:
    from IPython.display import JSON as _JSON
except Exception:  # pragma: no cover - only hit outside IPython
    _JSON = None


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


WORD_ID = _slug(WORD)
PERSON_ID = _slug(PERSON)
if not WORD_ID:
    raise ValueError("WORD is empty - do not edit it; it is the label/folder name.")
if not PERSON_ID:
    raise ValueError("Set PERSON in the config cell (your first name, e.g. 'newton') and re-run.")

NUM_FRAMES = int(round(FPS * CLIP_SECONDS))

# ----------------------------------------------------------------- output folder
ON_DRIVE = False


def _resolve_output_root():
    global ON_DRIVE
    if USE_DRIVE:
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
            root = os.path.join("/content/drive/MyDrive", DRIVE_FOLDER)
            os.makedirs(root, exist_ok=True)
            ON_DRIVE = True
            return root
        except Exception as exc:
            print(f"[setup] Google Drive not available ({exc!r}); saving locally instead.")
    root = os.path.abspath(LOCAL_FOLDER)
    os.makedirs(root, exist_ok=True)
    return root


OUT_ROOT = _resolve_output_root()
WORD_DIR = os.path.join(OUT_ROOT, WORD_ID)
os.makedirs(WORD_DIR, exist_ok=True)
MANIFEST_PATH = os.path.join(OUT_ROOT, "manifest.csv")
TMP_DIR = tempfile.mkdtemp(prefix="asl_collect_")

MANIFEST_COLUMNS = [
    "word", "person", "take", "file_base", "clip_file", "fps", "num_frames", "width", "height",
    "camera_fps", "measured_fps", "max_gap_ms", "dup_frames", "jpeg_quality", "saved_utc",
]


# -------------------------------------------------------------------- capture QA
def build_qa(n_frames, meta):
    """Capture-timing check only. Hands/pose are checked later, when keypoints are extracted."""
    n = max(int(n_frames), 1)
    ts = [float(t) for t in (meta.get("ts") or [])]
    if len(ts) >= 2 and ts[-1] > ts[0]:
        measured_fps = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1000.0)
        max_gap = max(b - a for a, b in zip(ts[:-1], ts[1:]))
    else:
        measured_fps, max_gap = 0.0, 0.0
    dups = int(meta.get("dup_frames", -1))
    stuttered = (measured_fps < 0.85 * FPS) or (dups > 0.10 * n) or (max_gap > 3.0 * 1000.0 / FPS)
    return {
        "frames": int(n_frames),
        "measured_fps": float(measured_fps),
        "max_gap_ms": float(max_gap),
        "stuttered": bool(stuttered),
    }


# ------------------------------------------------------------------- saving
def _decode_jpeg_b64(b64):
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode a JPEG frame from the browser")
    if img.shape[1] != WIDTH or img.shape[0] != HEIGHT:
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return img


def _write_clip(frames_bgr, base):
    """Write the downscaled clip locally. mp4 first, MJPG .avi as a safety net."""
    path = os.path.join(TMP_DIR, base + ".mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer.release()
        path = os.path.join(TMP_DIR, base + ".avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), float(FPS), (WIDTH, HEIGHT))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("OpenCV could not open a video writer (mp4v or MJPG).")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    return path


def _verify_clip(path, expected_frames):
    """Re-open the written clip so an unreadable or truncated file is never saved as 'good'."""
    cap = cv2.VideoCapture(path)
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
    finally:
        cap.release()
    if not ok or abs(count - expected_frames) > 1:
        raise RuntimeError(
            "the written clip looks wrong (%d frames, expected %d); take not saved" % (count, expected_frames)
        )


_TAKE_RE = re.compile(r"^%s_%s_(\d+)\.(?:mp4|avi)$" % (re.escape(WORD_ID), re.escape(PERSON_ID)))


def existing_take_indices():
    found = []
    for name in os.listdir(WORD_DIR):
        m = _TAKE_RE.match(name)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)


def _append_manifest(row):
    """Append one row. If manifest.csv already has a header (e.g. from an older notebook
    version with extra columns), keep that header and fill what we have."""
    fieldnames = MANIFEST_COLUMNS
    new_file = not os.path.exists(MANIFEST_PATH) or os.path.getsize(MANIFEST_PATH) == 0
    if not new_file:
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None)
        if header:
            fieldnames = header
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_current_take():
    """Persist the checked take. Order: clip, then manifest row (manifest = completed takes)."""
    s = SESSION
    if not s["bgr"] or s["qa"] is None:
        raise RuntimeError("no checked take to save")
    indices = existing_take_indices()
    idx = (indices[-1] + 1) if indices else 1
    base = "%s_%s_%03d" % (WORD_ID, PERSON_ID, idx)

    clip_tmp = _write_clip(s["bgr"], base)
    _verify_clip(clip_tmp, len(s["bgr"]))
    clip_name = os.path.basename(clip_tmp)
    shutil.copyfile(clip_tmp, os.path.join(WORD_DIR, clip_name))
    os.remove(clip_tmp)

    qa, meta = s["qa"], s["meta"]
    _append_manifest(
        {
            "word": WORD_ID,
            "person": PERSON_ID,
            "take": idx,
            "file_base": base,
            "clip_file": clip_name,
            "fps": FPS,
            "num_frames": len(s["bgr"]),
            "width": WIDTH,
            "height": HEIGHT,
            "camera_fps": meta.get("camera_fps") or "",
            "measured_fps": round(qa["measured_fps"], 2),
            "max_gap_ms": round(qa["max_gap_ms"], 1),
            "dup_frames": meta.get("dup_frames", -1),
            "jpeg_quality": JPEG_QUALITY,
            "saved_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return idx, base


# ---------------------------------------------------------- browser callbacks
SESSION = {"take_id": None, "bgr": [], "qa": None, "meta": None}


def _reset_session(take_id=None):
    SESSION.update({"take_id": take_id, "bgr": [], "qa": None, "meta": None})


def _reply(obj):
    return _JSON(obj) if _JSON is not None else obj


def _safe(fn):
    """Return {'error': ...} to the browser instead of raising (so the UI can show it)."""

    def wrapper(*args):
        try:
            return _reply(fn(*args))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return _reply({"error": "%s: %s" % (type(exc).__name__, exc)})

    wrapper.__name__ = fn.__name__
    return wrapper


def _check_take(take_id):
    if SESSION["take_id"] != take_id:
        raise RuntimeError("take id mismatch (browser %r vs python %r) - re-run the cell" % (take_id, SESSION["take_id"]))


@_safe
def cb_init():
    _reset_session()
    return {
        "ok": True,
        "saved": len(existing_take_indices()),
        "out_dir": WORD_DIR,
        "on_drive": ON_DRIVE,
    }


@_safe
def cb_begin(take_id):
    _reset_session(take_id)
    return {"ok": True}


@_safe
def cb_upload(take_id, start, *chunks):
    _check_take(take_id)
    if int(start) != len(SESSION["bgr"]):
        raise RuntimeError("frame upload out of order (expected %d, got %s)" % (len(SESSION["bgr"]), start))
    for b64 in chunks:
        SESSION["bgr"].append(_decode_jpeg_b64(b64))
    return {"ok": True, "n": len(SESSION["bgr"])}


@_safe
def cb_check(take_id, meta_json):
    _check_take(take_id)
    frames = SESSION["bgr"]
    if len(frames) != NUM_FRAMES:
        raise RuntimeError("expected %d frames, got %d" % (NUM_FRAMES, len(frames)))
    meta = json.loads(meta_json) if meta_json else {}
    qa = build_qa(len(frames), meta)
    SESSION.update({"qa": qa, "meta": meta})
    return {"ok": True, "qa": qa}


@_safe
def cb_save(take_id):
    _check_take(take_id)
    idx, base = save_current_take()
    _reset_session()
    return {"ok": True, "saved": len(existing_take_indices()), "take": idx, "file": base}


@_safe
def cb_discard(take_id):
    _reset_session()
    return {"ok": True}


@_safe
def cb_finish():
    _reset_session()
    return {"ok": True, "saved": len(existing_take_indices()), "out_dir": WORD_DIR}


CALLBACKS = {
    "asl.init": cb_init,
    "asl.begin": cb_begin,
    "asl.upload": cb_upload,
    "asl.check": cb_check,
    "asl.save": cb_save,
    "asl.discard": cb_discard,
    "asl.finish": cb_finish,
}


def register_callbacks(output_module=None):
    """Register the Python callbacks that the browser UI invokes."""
    if output_module is None:
        from google.colab import output as output_module  # type: ignore
    for name, fn in CALLBACKS.items():
        output_module.register_callback(name, fn)


def ui_config():
    """Settings handed to app.js (replaces the __CFG__ token)."""
    return {
        "word": WORD_ID,
        "wordDisplay": WORD_DISPLAY,
        "person": PERSON_ID,
        "fps": int(FPS),
        "numFrames": int(NUM_FRAMES),
        "width": int(WIDTH),
        "height": int(HEIGHT),
        "jpegQuality": float(JPEG_QUALITY),
        "countdown": int(COUNTDOWN_SECONDS),
        "target": int(TAKES),
        "autoContinue": bool(AUTO_CONTINUE),
        "mirrorPreview": bool(MIRROR_PREVIEW),
        "flashMs": 1500,
        "stepMs": 1000,
    }


# ------------------------------------------------------------ after the session
def summarize():
    """Print what was saved for this word/person (reads manifest.csv)."""
    rows = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("word") == WORD_ID and r.get("person") == PERSON_ID]
    print("Word: %s   Person: %s" % (WORD_ID, PERSON_ID))
    print("Saved takes: %d (target %d)" % (len(rows), TAKES))
    print("Folder: %s%s" % (WORD_DIR, "   [Google Drive]" if ON_DRIVE else "   [local Colab disk - download the zip!]"))
    if rows:
        fps = [float(r["measured_fps"]) for r in rows if r.get("measured_fps")]
        if fps:
            print("Measured capture fps: mean %.1f, min %.1f" % (np.mean(fps), min(fps)))
            slow = [r["file_base"] for r in rows if r.get("measured_fps") and float(r["measured_fps"]) < 0.85 * FPS]
            if slow:
                print("Low capture fps, consider re-recording:", ", ".join(slow))
    return rows


def make_zip():
    """Zip this word's folder plus the manifest and offer it as a download (Colab)."""
    zip_path = os.path.join(tempfile.gettempdir(), "asl_%s_%s.zip" % (WORD_ID, PERSON_ID))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(WORD_DIR)):
            zf.write(os.path.join(WORD_DIR, name), os.path.join(WORD_ID, name))
        if os.path.exists(MANIFEST_PATH):
            zf.write(MANIFEST_PATH, "manifest.csv")
    print("Wrote", zip_path)
    try:
        from google.colab import files  # type: ignore

        files.download(zip_path)
    except Exception:
        print("(Not in Colab - copy the zip from the path above.)")
    return zip_path


print(
    "[setup] word=%s person=%s | %d frames @ %d fps (%.1fs) at %dx%d"
    % (WORD_ID, PERSON_ID, NUM_FRAMES, FPS, CLIP_SECONDS, WIDTH, HEIGHT)
)
print("[setup] output: %s%s | already saved: %d" % (WORD_DIR, " (Google Drive)" if ON_DRIVE else " (local)", len(existing_take_indices())))
