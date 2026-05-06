from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedFrame:
    rgb: "object"  # numpy array (H,W,3) rgb24
    pts: object = None


class JpegPayloadDecoder:
    """Decode one TCP payload containing exactly one JPEG frame."""

    def decode_payload(self, payload: bytes) -> List[DecodedFrame]:
        arr = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cv2.imdecode failed (payload_bytes={len(payload)})")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return [DecodedFrame(rgb=rgb, pts=None)]

