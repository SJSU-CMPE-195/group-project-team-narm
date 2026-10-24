from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from jpeg_broadcast import JpegBroadcaster

_PLACEHOLDER_VERSION = -1


def _draw_overlay(
    frame_bgr: np.ndarray,
    *,
    sentence: Sequence[str],
    actions: Sequence[str],
    probs: Optional[np.ndarray],
) -> np.ndarray:
    out = frame_bgr
    h, w = out.shape[:2]
    # Text was laid out for ~960 px wide frames; shrink it for smaller previews.
    scale = max(0.5, min(1.0, w / 960.0))
    bar_h = int(40 * scale)
    font_main = 1.0 * scale
    font_prob = 0.9 * scale
    thick = 2 if scale >= 0.75 else 1

    # Top sentence bar (notebook-style).
    cv2.rectangle(out, (0, 0), (w, bar_h), (245, 117, 16), -1)
    cv2.putText(
        out,
        " ".join(sentence),
        (3, int(30 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_main,
        (255, 255, 255),
        thick,
        cv2.LINE_AA,
    )

    # Optional probability bars (notebook-style).
    if probs is not None and len(actions) > 0:
        colors = [
            (245, 117, 16),
            (117, 245, 16),
            (16, 117, 245),
            (245, 16, 117),
            (16, 245, 117),
            (117, 16, 245),
        ]
        max_w = min(int(300 * scale), w // 2)
        row = int(40 * scale)
        top = bar_h + int(20 * scale)
        for i, action in enumerate(actions):
            if i >= len(probs):
                break
            p = float(probs[i])
            p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else p
            y0 = top + i * row
            y1 = y0 + int(30 * scale)
            if y1 > h:
                break
            cv2.rectangle(out, (0, y0), (int(p * max_w), y1), colors[i % len(colors)], -1)
            cv2.putText(
                out,
                action,
                (3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_prob,
                (255, 255, 255),
                thick,
                cv2.LINE_AA,
            )

    return out


def _connections():
    """MediaPipe skeleton connections, or empty lists if mediapipe is unavailable."""
    try:
        import mediapipe as mp  # local import: the preview must still work without it

        return (
            sorted(tuple(int(i) for i in c) for c in mp.solutions.pose.POSE_CONNECTIONS),
            sorted(tuple(int(i) for i in c) for c in mp.solutions.hands.HAND_CONNECTIONS),
        )
    except Exception:
        return [], []


def _draw_landmarks(
    frame_bgr: np.ndarray,
    landmarks: Optional[dict],
    pose_conn: Sequence[Tuple[int, int]],
    hand_conn: Sequence[Tuple[int, int]],
) -> None:
    """Draw the skeleton MediaPipe produced for the model (normalized x, y)."""
    if not landmarks:
        return
    h, w = frame_bgr.shape[:2]

    def px(pt):
        return int(pt[0] * w), int(pt[1] * h)

    def draw(points, connections, color, radius):
        if not points:
            return
        for a, b in connections:
            if a < len(points) and b < len(points):
                cv2.line(frame_bgr, px(points[a]), px(points[b]), color, 1, cv2.LINE_AA)
        for pt in points:
            cv2.circle(frame_bgr, px(pt), radius, color, -1, cv2.LINE_AA)

    draw(landmarks.get("pose"), pose_conn, (0, 200, 255), 3)
    draw(landmarks.get("lh"), hand_conn, (0, 255, 0), 2)
    draw(landmarks.get("rh"), hand_conn, (255, 128, 0), 2)
    for pt in landmarks.get("face") or []:
        cv2.circle(frame_bgr, px(pt), 1, (200, 200, 200), -1)


def _placeholder_frame(width: int = 640, height: int = 360) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "Preview running - waiting for frames...",
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _downscale_to_bgr(frame_rgb: np.ndarray, width: int) -> np.ndarray:
    """Shrink first (INTER_AREA), then convert color, so both steps touch fewer pixels."""
    h, w = frame_rgb.shape[:2]
    if width > 0 and w > width:
        new_h = max(1, int(round(h * width / float(w))))
        frame_rgb = cv2.resize(frame_rgb, (width, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


@dataclass(frozen=True)
class PreviewHTTPServer:
    latest_state: object
    host: str = "0.0.0.0"
    port: int = 8000
    max_fps: float = 12.0   # page refresh cap; decoding and inference are unaffected
    width: int = 640        # preview width in px (0 = keep decoded size)
    quality: int = 75       # JPEG quality

    def _make_render(self):
        latest_state = self.latest_state
        width = int(self.width)
        pose_conn, hand_conn = _connections()

        def render(last_version):
            version, frame_rgb, sentence, probs, actions, landmarks = latest_state.preview_snapshot()
            if frame_rgb is None:
                if last_version == _PLACEHOLDER_VERSION:
                    return None
                return _PLACEHOLDER_VERSION, _placeholder_frame()
            if version == last_version:
                return None
            frame_bgr = _downscale_to_bgr(frame_rgb, width)
            _draw_landmarks(frame_bgr, landmarks, pose_conn, hand_conn)
            frame_bgr = _draw_overlay(frame_bgr, sentence=sentence, actions=actions, probs=probs)
            return version, frame_bgr

        return render

    def serve_forever(self) -> None:
        broadcaster = JpegBroadcaster(self._make_render(), max_fps=self.max_fps, quality=self.quality)
        broadcaster.start()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                # Keep stdout clean; main.py already prints status.
                return

            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    body = (
                        "<!doctype html>"
                        "<html><head><meta charset='utf-8'>"
                        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                        "<title>ASL Preview</title>"
                        "<style>body{font-family:system-ui;margin:12px}img{max-width:100%;height:auto}</style>"
                        "</head><body>"
                        "<h3>ASL Preview</h3>"
                        "<p><img src='/stream.mjpg' alt='stream'></p>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path.startswith("/stream.mjpg"):
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()

                    seq = 0
                    while True:
                        try:
                            seq, jpg = broadcaster.wait_for_next(seq, timeout=2.0)
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                            return

                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"not found")

        httpd = ThreadingHTTPServer((self.host, int(self.port)), Handler)
        httpd.daemon_threads = True
        httpd.serve_forever()
