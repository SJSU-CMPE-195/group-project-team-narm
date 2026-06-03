#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import socket
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from picamera2 import Picamera2

try:
    from libcamera import controls
except Exception:  # pragma: no cover
    controls = None


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Stats:
    t0: float
    frames: int = 0
    bytes_sent: int = 0

    def add(self, nbytes: int) -> Optional[str]:
        self.frames += 1
        self.bytes_sent += nbytes
        now = time.time()
        dt = now - self.t0
        if dt < 1.0:
            return None
        fps = self.frames / dt if dt > 0 else 0.0
        avg = int(self.bytes_sent / self.frames) if self.frames else 0
        kbps = (self.bytes_sent * 8.0 / 1000.0) / dt if dt > 0 else 0.0
        self.t0 = now
        self.frames = 0
        self.bytes_sent = 0
        return f"[pi] camera ok: fps={fps:.1f} avg_frame={avg} bytes bitrate={kbps:.0f} kbps"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Raspberry Pi Camera Module 3 -> Jetson TCP JPEG sender"
    )
    p.add_argument("--host", default=env_str("JETSON_HOST", "10.42.0.1"))
    p.add_argument("--port", type=int, default=env_int("JETSON_PORT", 5000))
    p.add_argument("--width", type=int, default=env_int("WIDTH", 640))
    p.add_argument("--height", type=int, default=env_int("HEIGHT", 480))
    p.add_argument("--fps", type=int, default=env_int("FPS", 10))
    p.add_argument("--jpeg-quality", type=int, default=env_int("JPEG_QUALITY", 78))
    p.add_argument("--connect-timeout", type=float, default=5.0)
    p.add_argument("--retry-delay", type=float, default=1.0)
    p.add_argument(
        "--af-continuous",
        action="store_true",
        default=env_bool("AF_CONTINUOUS", True),
        help="Enable continuous autofocus when available (Camera Module 3).",
    )
    return p


def connect_tcp(host: str, port: int, timeout_s: float) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sndbuf = 256 * 1024
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
    return sock


def encode_jpeg_rgb(frame_rgb, quality: int) -> bytes:
    quality = max(1, min(95, quality))
    image = Image.fromarray(frame_rgb, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def main() -> None:
    args = build_parser().parse_args()
    frame_period_s = 1.0 / args.fps if args.fps > 0 else 0.0

    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": float(args.fps)},
        queue=False,
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.2)

    if args.af_continuous and controls is not None:
        try:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except Exception:
            pass

    print(
        f"[pi] camera started: {args.width}x{args.height} fps={args.fps} "
        f"jpeg_quality={args.jpeg_quality}"
    )

    sock: Optional[socket.socket] = None
    stats = Stats(t0=time.time())

    try:
        while True:
            if sock is None:
                try:
                    print(f"[pi] connecting to {args.host}:{args.port} ...")
                    sock = connect_tcp(args.host, args.port, timeout_s=args.connect_timeout)
                    print("[pi] connected")
                except OSError as e:
                    print(f"[pi] connect failed: {e!r}")
                    time.sleep(args.retry_delay)
                    continue

            t_loop = time.time()
            try:
                frame = picam2.capture_array("main")
                jpeg = encode_jpeg_rgb(frame, quality=args.jpeg_quality)
                header = len(jpeg).to_bytes(4, byteorder="big", signed=False)
                sock.sendall(header)
                sock.sendall(jpeg)
                line = stats.add(len(jpeg))
                if line:
                    print(line)
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError) as e:
                print(f"[pi] send failed: {e!r}; reconnecting ...")
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
                time.sleep(args.retry_delay)
                continue

            if frame_period_s > 0:
                elapsed = time.time() - t_loop
                sleep_s = frame_period_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
