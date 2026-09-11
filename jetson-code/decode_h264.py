from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DecodedFrame:
    rgb: "object"  # numpy array (H,W,3) rgb24
    pts: object = None


class H264PayloadDecoder:
    def __init__(self):
        try:
            import av  # type: ignore
        except ImportError as e:
            raise RuntimeError("Missing dependency: av. Install with `pip install av`.") from e

        self._av = av
        self._codec = self._new_codec()

    def _new_codec(self):
        codec = self._av.CodecContext.create("h264", "r")
        codec.thread_type = "AUTO"
        return codec

    def reset(self) -> None:
        """Start a fresh decoder when the ESP reconnects with a new H.264 stream."""
        try:
            self._codec.close()
        except Exception:
            pass
        self._codec = self._new_codec()

    def decode_payload(self, payload: bytes) -> List[DecodedFrame]:
        pkt = self._av.Packet(payload)
        frames = self._codec.decode(pkt)
        out: List[DecodedFrame] = []
        for f in frames:
            out.append(DecodedFrame(rgb=f.to_ndarray(format="rgb24"), pts=getattr(f, "pts", None)))
        return out

