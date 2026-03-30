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
        self._codec = av.CodecContext.create("h264", "r")

    def decode_payload(self, payload: bytes) -> List[DecodedFrame]:
        # #region agent log
        import json, time
        try:
            with open("debug-9d06a1.log", "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "9d06a1",
                            "runId": "run1",
                            "hypothesisId": "H3",
                            "location": "jetson-code/decode_h264.py",
                            "message": "decode_payload",
                            "data": {"payload_bytes": len(payload), "prefix_hex": payload[:8].hex()},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion agent log
        pkt = self._av.Packet(payload)
        frames = self._codec.decode(pkt)
        out: List[DecodedFrame] = []
        for f in frames:
            out.append(DecodedFrame(rgb=f.to_ndarray(format="rgb24"), pts=getattr(f, "pts", None)))
        return out

