"""Receive the ESP32-P4 framed H.264 stream and serve an MJPEG preview.

Protocol: repeated [4-byte big-endian length][H.264 access unit].
This is intentionally independent of the TensorFlow/ASL pipeline.

Cost control: frames are decoded at full size, but the preview is shrunk with
INTER_AREA before color conversion, JPEG-encoded once per new frame (shared by
every browser tab), and capped at --fps frames per second.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
import cv2
import numpy as np

from jpeg_broadcast import JpegBroadcaster


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame_rgb: np.ndarray | None = None
        self.frames = 0
        self.bytes = 0
        self.started = time.monotonic()
        self.last_frame = 0.0

    def update(self, frame_rgb: np.ndarray, payload_size: int) -> None:
        with self.lock:
            self.frame_rgb = frame_rgb
            self.frames += 1
            self.bytes += payload_size
            self.last_frame = time.monotonic()

    def counters(self) -> tuple[int, int]:
        with self.lock:
            return self.frames, self.bytes


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("sender disconnected")
        data.extend(chunk)
    return bytes(data)


def new_decoder():
    decoder = av.CodecContext.create("h264", "r")
    decoder.thread_type = "AUTO"
    return decoder


def receiver(state: State, bind_host: str, tcp_port: int) -> None:
    while True:
        decoder = new_decoder()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((bind_host, tcp_port))
                server.listen(1)
                print(f"[tcp] listening on {bind_host}:{tcp_port}")
                client, address = server.accept()
                print(f"[tcp] connected: {address}")
                with client:
                    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                    while True:
                        length = struct.unpack(">I", recv_exact(client, 4))[0]
                        if not 0 < length <= 8 * 1024 * 1024:
                            raise ConnectionError(
                                f"invalid payload length: {length} "
                                "(port 5000 is for the ESP32, not a browser; use the HTTP port)"
                            )
                        payload = recv_exact(client, length)
                        for decoded in decoder.decode(av.Packet(payload)):
                            state.update(decoded.to_ndarray(format="rgb24"), length)
        except (ConnectionError, OSError, av.error.FFmpegError) as exc:
            print(f"[tcp] disconnected: {exc}; waiting for reconnect")
            time.sleep(1)


def stats_reporter(state: State, interval: float = 5.0) -> None:
    previous_frames, previous_bytes = state.counters()
    previous_ts = time.monotonic()
    while True:
        time.sleep(interval)
        frames, total_bytes = state.counters()
        now = time.monotonic()
        elapsed = max(now - previous_ts, 0.001)
        fps = (frames - previous_frames) / elapsed
        mbps = (total_bytes - previous_bytes) * 8.0 / elapsed / 1_000_000.0
        print(f"[stats] decoded={fps:.1f}fps rx={mbps:.2f}Mbps")
        previous_frames, previous_bytes, previous_ts = frames, total_bytes, now


def make_render(state: State, width: int):
    waiting = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        waiting,
        "Waiting for the ESP32 stream...",
        (30, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    def render(last_version):
        with state.lock:
            frame = state.frame_rgb
            version = state.frames
        if frame is None:
            return (-1, waiting) if last_version != -1 else None
        if version == last_version:
            return None
        h, w = frame.shape[:2]
        if width > 0 and w > width:
            frame = cv2.resize(
                frame, (width, max(1, int(round(h * width / float(w))))), interpolation=cv2.INTER_AREA
            )
        return version, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return render


def http_server(broadcaster: JpegBroadcaster, bind_host: str, http_port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                body = (
                    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                    "<title>ESP32-P4 camera</title>"
                    "<h3>ESP32-P4 / OV5647</h3>"
                    "<img src='/stream.mjpg' style='max-width:100%;height:auto'>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seq = 0
            while True:
                try:
                    seq, data = broadcaster.wait_for_next(seq, timeout=2.0)
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return

    print(f"[http] preview: http://{bind_host}:{http_port}")
    server = ThreadingHTTPServer((bind_host, http_port), Handler)
    server.daemon_threads = True
    server.serve_forever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp-port", type=int, default=5000)
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640, help="preview width in px (0 = full size)")
    parser.add_argument("--fps", type=float, default=15.0, help="max preview refresh rate")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality 1-100")
    args = parser.parse_args()

    state = State()
    threading.Thread(target=receiver, args=(state, "0.0.0.0", args.tcp_port), daemon=True).start()
    threading.Thread(target=stats_reporter, args=(state,), daemon=True).start()
    broadcaster = JpegBroadcaster(make_render(state, args.width), max_fps=args.fps, quality=args.quality)
    broadcaster.start()
    http_server(broadcaster, "0.0.0.0", args.http_port)


if __name__ == "__main__":
    main()
