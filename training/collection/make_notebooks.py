#!/usr/bin/env python3
"""Generate the per-word Google Colab data-collection notebooks.

The notebooks are assembled from two source files so a fix is made once and
regenerated, never hand-edited in 32 places:

    template/helpers.py   Python side (saving clips, capture check, Colab callbacks)
    template/app.js       browser UI (word flash, countdown, recording, review)

Usage (run from anywhere):

    python make_notebooks.py --word back          # one notebook (test notebook)
    python make_notebooks.py --words back,pain    # a few
    python make_notebooks.py --all                # every word in WORDS
    python make_notebooks.py --word back --out /tmp/x   # custom output folder

Output goes to  training/collection/notebooks/collect_<word>.ipynb  by default.
"""

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "template")
DEFAULT_OUT = os.path.join(HERE, "notebooks")

# The label folders under training/raw_videos/. "upper" and "lower" were in the
# original 34-word list but have no folders yet; add them here if they are kept.
WORDS = [
    "again", "allergy", "arm", "back", "better", "breathe", "chest", "cough",
    "dizzy", "dont_understand", "fever", "head", "help", "here", "i_me", "leg",
    "left", "medicine", "nausea", "need", "no", "pain", "right", "sick",
    "slow", "stomach", "strong", "throat", "vomit", "weak", "worse", "yes",
]

# On-screen text when it should differ from the folder name.
DISPLAY_OVERRIDES = {
    "i_me": "I / ME",
    "dont_understand": "DON'T UNDERSTAND",
}


def display_name(word):
    return DISPLAY_OVERRIDES.get(word, word.replace("_", " ").upper())


def _read(name):
    with open(os.path.join(TEMPLATE_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def _cell_id(word, index):
    return hashlib.sha1(("%s:%d" % (word, index)).encode()).hexdigest()[:8]


def _source_lines(text):
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def _md(word, index, text):
    return {"cell_type": "markdown", "id": _cell_id(word, index), "metadata": {}, "source": _source_lines(text.strip("\n"))}


def _code(word, index, text, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    return {
        "cell_type": "code",
        "id": _cell_id(word, index),
        "metadata": meta,
        "execution_count": None,
        "outputs": [],
        "source": _source_lines(text.strip("\n")),
    }


INTRO_MD = """\
# ASL data collection - **{display}**

Records short webcam takes of the sign **{display}** for the new training set. Every take you approve is
saved as a video clip plus a row in `manifest.csv`. Keypoints are extracted later from the clips, so there is
**nothing to install** in this notebook.

**How to use (desktop Chrome, laptop webcam):**

1. In **1. Config**, type your first name in `PERSON` (lowercase) and press play.
2. Press play on **Setup** (it mounts Google Drive so takes survive a disconnect - click Allow).
3. Press play on **2. Record takes**. Allow camera access when Chrome asks.
4. For each take the word flashes, a 3-2-1 countdown runs, then it records
   {frames} frames ({seconds:.1f} s) with a frame counter in the corner. Sign **{display}** once during the take.
5. Replay the take (pause, slow-mo, scrub), then click **Good - save** or **Bad - retry**.
6. It stops after **{takes}** good takes. Re-running the cell resumes where you left off.
7. Run **3. Summary** at the end to check the results (and optionally download a zip).

**Tips:** keep your head and both hands in frame the whole time, sit about where the glasses camera would see a
signer, use even lighting, and use the same camera position for every take. Saved frames are never mirrored.
Do not change `FPS`, `CLIP_SECONDS`, `WIDTH` or `HEIGHT` - they must match across the whole team.

**If you see an error:**

- *"Set PERSON in the config cell"*: type your first name in `PERSON` and re-run the Config cell.
- *No camera preview*: use desktop Chrome, click Allow on the camera prompt, and close other apps using the webcam.
- *"Capture stuttered"* warning: close other apps using the camera and retry the take.
- After any restart, re-run Config, Setup, then Record.
"""

CONFIG_CODE = '''\
# ---- 1. Config ------------------------------------------------------------
# Only PERSON (and optionally TAKES) should be edited. Everything else must stay
# identical for every teammate so the datasets line up.

WORD = {word!r}                  # label / folder name - do not change
WORD_DISPLAY = {display!r}       # text flashed on screen

# Your first name, lowercase letters/digits, e.g. "newton"
PERSON = "" #@param {{type:"string"}}
# Good takes to collect for this word
TAKES = 30 #@param {{type:"integer"}}

FPS = 30                   # capture rate; 30 lets you derive 15 or 10 fps later
CLIP_SECONDS = 3.0         # 3.0 s -> 90 frames
WIDTH, HEIGHT = 640, 360   # 16:9, matches the glasses stream aspect ratio
JPEG_QUALITY = 0.85
COUNTDOWN_SECONDS = 3
AUTO_CONTINUE = True       # False = click "Start next take" between takes
MIRROR_PREVIEW = False     # cosmetic: mirrors only the live preview, never the saved frames

USE_DRIVE = True           # save to Google Drive (recommended; survives disconnects)
DRIVE_FOLDER = "ASL_collection"
LOCAL_FOLDER = "/content/asl_collection_output"   # used if Drive is off/unavailable
'''

RUN_PREFIX = '''\
#@title 2. Record takes - press play { display-mode: "form" }
import json
from IPython.display import display, Javascript

APP_JS = r\'\'\'
'''

RUN_SUFFIX = """
\'\'\'

register_callbacks()
display(Javascript(APP_JS.replace("__CFG__", json.dumps(ui_config()))))
"""

SUMMARY_CODE = """\
#@title 3. Summary + optional zip download { display-mode: "form" }
DOWNLOAD_ZIP = False #@param {type:"boolean"}
summarize()
if DOWNLOAD_ZIP:
    make_zip()
"""


def build_notebook(word):
    if not re.fullmatch(r"[a-z0-9_]+", word):
        raise ValueError("word must be lowercase letters/digits/underscores: %r" % word)
    display = display_name(word)
    helpers = _read("helpers.py")
    app_js = _read("app.js")
    if "'''" in app_js:
        raise ValueError("app.js must not contain ''' (it is embedded in a Python raw string)")

    frames = 90
    cells = [
        _md(word, 0, INTRO_MD.format(display=display, frames=frames, seconds=3.0, takes=30)),
        _code(word, 1, CONFIG_CODE.format(word=word, display=display)),
        _code(word, 2, '#@title Setup - press play (do not edit) { display-mode: "form" }\n' + helpers, hidden=True),
        _code(word, 3, RUN_PREFIX + app_js.strip("\n") + RUN_SUFFIX, hidden=True),
        _code(word, 4, SUMMARY_CODE, hidden=True),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "collect_%s.ipynb" % word, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(word, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "collect_%s.ipynb" % word)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(build_notebook(word), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--word", help="generate a single notebook for this word")
    group.add_argument("--words", help="comma-separated words")
    group.add_argument("--all", action="store_true", help="generate every word in WORDS")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder (default: %(default)s)")
    args = parser.parse_args(argv)

    if args.all:
        words = list(WORDS)
    elif args.words:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
    else:
        words = [args.word.strip()]

    for word in words:
        print("wrote", write_notebook(word, args.out))
    print("%d notebook(s) -> %s" % (len(words), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
