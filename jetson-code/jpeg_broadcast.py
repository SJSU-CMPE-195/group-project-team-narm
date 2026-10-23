"""Encode preview JPEGs once and share them with every browser tab.

The old preview re-encoded a full-size JPEG separately for every open tab and
every loop iteration, even when no new frame had arrived. A single background
thread now renders and encodes at most `max_fps` times per second, and only
when the source reports a new version. HTTP handlers just wait for the next
finished JPEG and write the bytes.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

# render(last_version) -> (version, frame_bgr) when there is something new to
# show, or None when nothing changed since last_version.
RenderFn = Callable[[Optional[int]], Optional[Tuple[int, np.ndarray]]]


class JpegBroadcaster:
    def __init__(self, render: RenderFn, *, max_fps: float = 12.0, quality: int = 75) -> None:
        self._render = render
        self._interval = 1.0 / max(float(max_fps), 1.0)
        self._quality = int(quality)
        self._cond = threading.Condition()
        self._seq = 0
        self._jpeg: Optional[bytes] = None
        self._last_version: Optional[int] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="preview-jpeg", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            started = time.monotonic()
            try:
                rendered = self._render(self._last_version)
                if rendered is not None:
                    version, frame = rendered
                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
                    if ok:
                        with self._cond:
                            self._jpeg = buf.tobytes()
                            self._seq += 1
                            self._last_version = version
                            self._cond.notify_all()
            except Exception:
                time.sleep(0.2)  # transient render problem; keep the thread alive
            remaining = self._interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    def wait_for_next(self, last_seq: int, timeout: float = 2.0) -> Tuple[int, Optional[bytes]]:
        """Block until a JPEG newer than last_seq exists. Returns (seq, jpeg or None on timeout)."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._jpeg
