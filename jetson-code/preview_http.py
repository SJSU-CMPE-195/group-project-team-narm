from __future__ import annotations

import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Sequence

import cv2
import numpy as np


def _draw_overlay(
    frame_bgr: np.ndarray,
    *,
    sentence: Sequence[str],
    actions: Sequence[str],
    probs: Optional[np.ndarray],
) -> np.ndarray:
    out = frame_bgr
    h, w = out.shape[:2]

    # Top sentence bar (notebook-style).
    cv2.rectangle(out, (0, 0), (w, 40), (245, 117, 16), -1)
    cv2.putText(
        out,
        " ".join(sentence),
        (3, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
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
        max_w = min(300, w // 2)
        for i, action in enumerate(actions):
            if i >= len(probs):
                break
            p = float(probs[i])
            p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else p
            y0 = 60 + i * 40
            y1 = 90 + i * 40
            if y1 > h:
                break
            cv2.rectangle(out, (0, y0), (int(p * max_w), y1), colors[i % len(colors)], -1)
            cv2.putText(
                out,
                action,
                (3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return out


def _placeholder_frame(width: int = 960, height: int = 540) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "Preview running - waiting for frames...",
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _jpeg_bytes(frame_bgr: np.ndarray, *, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed")
    return buf.tobytes()


@dataclass(frozen=True)
class PreviewHTTPServer:
    latest_state: object
    host: str = "0.0.0.0"
    port: int = 8000

    def serve_forever(self) -> None:
        latest_state = self.latest_state

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

                    while True:
                        try:
                            frame, sentence, probs, actions = latest_state.snapshot()
                            if frame is None:
                                frame = _placeholder_frame()
                                sentence = ["(waiting)"]
                                probs = None
                                actions = []

                            frame = _draw_overlay(
                                frame,
                                sentence=sentence,
                                actions=actions,
                                probs=probs,
                            )
                            jpg = _jpeg_bytes(frame, quality=80)

                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()

                            time.sleep(0.05)  # ~20 fps max; keeps CPU reasonable
                        except BrokenPipeError:
                            return
                        except ConnectionResetError:
                            return
                        except Exception:
                            # If anything transient happens, keep the stream alive.
                            time.sleep(0.2)
                    return

                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"not found")

        httpd = ThreadingHTTPServer((self.host, int(self.port)), Handler)
        httpd.serve_forever()

