import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class H264Frame:
    payload: bytes
    received_ts: float
    stream_id: int = 0


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


class TCPIngestServer:
    """TCP server for ESP32 framing: [len:u32be][payload]*"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        on_frame: Optional[Callable[[H264Frame], None]] = None,
        backlog: int = 1,
        max_payload_bytes: int = 8 * 1024 * 1024,
    ):
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self.backlog = backlog
        self.max_payload_bytes = max_payload_bytes
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(self.backlog)
            print(f"[tcp] listening on {self.host}:{self.port}")
            stream_id = 0

            while not self._stop.is_set():
                try:
                    s.settimeout(1.0)
                    client, addr = s.accept()
                except TimeoutError:
                    continue

                with client:
                    stream_id += 1
                    print(f"[tcp] client connected: {addr}")
                    client.settimeout(10.0)
                    while not self._stop.is_set():
                        try:
                            header = _recv_exact(client, 4)
                            (length,) = struct.unpack(">I", header)
                            if length == 0:
                                continue
                            if length > self.max_payload_bytes:
                                raise ConnectionError(
                                    f"payload length {length} exceeds limit "
                                    f"{self.max_payload_bytes}"
                                )
                            payload = _recv_exact(client, length)
                            if self.on_frame:
                                self.on_frame(
                                    PayloadFrame(
                                        payload=payload,
                                        received_ts=time.monotonic(),
                                        stream_id=stream_id,
                                    )
                                )
                        except (ConnectionError, OSError) as e:
                            print(f"[tcp] disconnected: {e}")
                            break
                        except socket.timeout:
                            continue

