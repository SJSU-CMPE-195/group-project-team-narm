"""Receive the ESP32-P4 framed H.264 stream and serve an MJPEG preview.

Protocol: repeated [4-byte big-endian length][H.264 access unit].
This is intentionally independent of the TensorFlow/ASL pipeline.
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


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frames = 0
        self.bytes = 0
        self.started = time.monotonic()
        self.last_frame = 0.0

    def update(self, frame: np.ndarray, payload_size: int) -> None:
        with self.lock:
            self.frame = frame
            self.frames += 1
            self.bytes += payload_size
            self.last_frame = time.monotonic()


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("sender disconnected")
        data.extend(chunk)
    return bytes(data)


def receiver(state: State, bind_host: str, tcp_port: int) -> None:
    while True:
        decoder = av.CodecContext.create("h264", "r")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((bind_host, tcp_port))
                server.listen(1)
                print(f"[tcp] listening on {bind_host}:{tcp_port}")
                client, address = server.accept()
                print(f"[tcp] connected: {address}")
                with client:
                    while True:
                        length = struct.unpack(">I", recv_exact(client, 4))[0]
                        if not 0 < length <= 8 * 1024 * 1024:
                            raise ConnectionError(f"invalid payload length: {length}")
                        payload = recv_exact(client, length)
                        for decoded in decoder.decode(av.Packet(payload)):
                            # OpenCV uses BGR; PyAV gives RGB.
                            rgb = decoded.to_ndarray(format="rgb24")
                            state.update(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), length)
        except (ConnectionError, OSError, av.error.FFmpegError) as exc:
            print(f"[tcp] disconnected: {exc}; waiting for reconnect")
            time.sleep(1)


def jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def http_server(state: State, bind_host: str, http_port: int) -> None:
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
            while True:
                with state.lock:
                    frame = None if state.frame is None else state.frame.copy()
                if frame is not None:
                    data = jpeg(frame)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.05)

    print(f"[http] preview: http://{bind_host}:{http_port}")
    ThreadingHTTPServer((bind_host, http_port), Handler).serve_forever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp-port", type=int, default=5000)
    parser.add_argument("--http-port", type=int, default=8000)
    args = parser.parse_args()

    state = State()
    threading.Thread(target=receiver, args=(state, "0.0.0.0", args.tcp_port), daemon=True).start()
    http_server(state, "0.0.0.0", args.http_port)


if __name__ == "__main__":
    main()
