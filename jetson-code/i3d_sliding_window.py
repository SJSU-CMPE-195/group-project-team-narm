from __future__ import annotations

import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class GlossPrediction:
    gloss: str
    class_index: int
    confidence: float
    ts: float


def load_wlasl_class_list(wlasl_root: str) -> List[str]:
    path = os.path.join(wlasl_root, "code", "I3D", "preprocess", "wlasl_class_list.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def prepare_clip(frames_rgb: List[np.ndarray], size: int = 224) -> torch.Tensor:
    import cv2

    clip = []
    for fr in frames_rgb:
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_LINEAR)
        fr = (fr.astype(np.float32) / 255.0) * 2.0 - 1.0
        clip.append(fr)
    arr = np.stack(clip, axis=0)  # (T,H,W,C)
    arr = np.transpose(arr, (3, 0, 1, 2))  # (C,T,H,W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1,C,T,H,W)


class I3DSlidingWindow:
    def __init__(
        self,
        wlasl_root: str,
        checkpoint_path: str,
        window_t: int = 64,
        smooth_n: int = 5,
        device: Optional[str] = None,
    ):
        self.wlasl_root = wlasl_root
        self.window_t = window_t
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.class_list = load_wlasl_class_list(wlasl_root)
        if not self.class_list:
            raise RuntimeError("WLASL class list not found or empty.")
        num_classes = len(self.class_list)

        i3d_dir = os.path.join(wlasl_root, "code", "I3D")
        if i3d_dir not in sys.path:
            sys.path.insert(0, i3d_dir)
        from pytorch_i3d import InceptionI3d  # type: ignore

        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(num_classes)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

        self.frames: Deque[np.ndarray] = deque(maxlen=window_t)
        self.recent: Deque[int] = deque(maxlen=smooth_n)

    def push_frame(self, rgb: np.ndarray) -> Optional[GlossPrediction]:
        self.frames.append(rgb)
        if len(self.frames) < self.window_t:
            return None

        clip = prepare_clip(list(self.frames)).to(self.device)
        # #region agent log
        import json, time as _t
        try:
            with open("debug-9d06a1.log", "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "9d06a1",
                            "runId": os.environ.get("DEBUG_RUN_ID", "run1"),
                            "hypothesisId": "H4",
                            "location": "jetson-code/i3d_sliding_window.py",
                            "message": "i3d_infer",
                            "data": {"device": self.device, "clip_shape": list(clip.shape)},
                            "timestamp": int(_t.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion agent log
        with torch.no_grad():
            out = self.model(clip)
            if out.dim() == 3:
                logits = out.mean(dim=-1)
            else:
                logits = out
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            class_idx = int(idx[0].item())
            confidence = float(conf[0].item())

        self.recent.append(class_idx)
        smoothed = Counter(self.recent).most_common(1)[0][0]
        gloss = self.class_list[smoothed] if smoothed < len(self.class_list) else str(smoothed)
        return GlossPrediction(gloss=gloss, class_index=smoothed, confidence=confidence, ts=time.time())

