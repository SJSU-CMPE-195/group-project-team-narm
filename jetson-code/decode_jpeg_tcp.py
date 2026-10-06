from __future__ import annotations

import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("socket closed")
        got += r
    return bytes(buf)


def _u32be(b: bytes) -> int:
    (v,) = struct.unpack(">I", b)
    return int(v)


@dataclass
class Stats:
    frames: int = 0
    bytes: int = 0
    t0: float = time.time()

    def tick(self, nbytes: int) -> None:
        self.frames += 1
        self.bytes += int(nbytes)
        now = time.time()
        if now - self.t0 >= 2.0:
            dt = now - self.t0
            fps = self.frames / dt
            mbps = (self.bytes * 8) / dt / 1e6
            print(f"[jpeg] fps={fps:.1f}  rate={mbps:.2f} Mbps  last={nbytes/1024:.1f} KiB", flush=True)
            self.frames = 0
            self.bytes = 0
            self.t0 = now


def _decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))

    print(f"[tcp] listening on {host}:{port}", flush=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)

    while True:
        conn, addr = srv.accept()
        print(f"[tcp] client connected: {addr}", flush=True)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        stats = Stats()

        try:
            while True:
                n = _u32be(_recv_exact(conn, 4))
                if n == 0:
                    continue
                jpeg_bytes = _recv_exact(conn, n)
                frame = _decode_jpeg(jpeg_bytes)
                if frame is None:
                    print(f"[jpeg] decode failed (n={n})", flush=True)
                    continue
                stats.tick(n)
        except Exception as e:
            print(f"[tcp] client disconnected: {e}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

