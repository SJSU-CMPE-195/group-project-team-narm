from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


def extract_keypoints(results) -> np.ndarray:
    """
    Match Action Detection Refined.ipynb keypoint layout (1662 dims):
    - pose: 33 * (x,y,z,visibility) = 132
    - face: 468 * (x,y,z) = 1404
    - left hand: 21 * (x,y,z) = 63
    - right hand: 21 * (x,y,z) = 63
    """
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


@dataclass(frozen=True)
class GlossPrediction:
    gloss: str
    confidence: float


class HolisticLSTMInfer:
    def __init__(
        self,
        model,
        actions: Sequence[str],
        *,
        sequence_length: int = 30,
        threshold: float = 0.5,
        stable_n: int = 10,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model = model
        self.actions = list(actions)
        self.sequence_length = int(sequence_length)
        self.threshold = float(threshold)
        self.stable_n = int(stable_n)

        import mediapipe as mp  # local import for easier install/debug

        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._sequence: List[np.ndarray] = []
        self._predictions: List[int] = []
        self._last_gloss: Optional[str] = None

    def close(self) -> None:
        try:
            self._holistic.close()
        except Exception:
            pass

    def _predict(self, sequence: List[np.ndarray]) -> Tuple[int, float, np.ndarray]:
        x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)  # (1, T, 1662)
        # Direct eager invocation avoids Keras predict()'s per-call data adapter.
        output = self.model(x, training=False)
        res = np.asarray(output)[0]
        idx = int(np.argmax(res))
        conf = float(res[idx])
        return idx, conf, res

    def push_frame(self, rgb_frame: np.ndarray) -> Optional[GlossPrediction]:
        """
        Push one RGB frame (H,W,3 uint8). Returns a new stable gloss prediction
        when conditions are met, otherwise None.
        """
        # MediaPipe expects RGB.
        img = rgb_frame
        img.flags.writeable = False
        results = self._holistic.process(img)
        img.flags.writeable = True

        keypoints = extract_keypoints(results)
        self._sequence.append(keypoints)
        self._sequence = self._sequence[-self.sequence_length :]

        if len(self._sequence) < self.sequence_length:
            return None

        idx, conf, _ = self._predict(self._sequence)
        self._predictions.append(idx)
        self._predictions = self._predictions[-max(self.stable_n, 1) :]

        # Notebook stability logic: last N predictions agree and pass threshold.
        if self.stable_n > 1:
            if len(self._predictions) < self.stable_n:
                return None
            if len(set(self._predictions[-self.stable_n :])) != 1:
                return None

        if conf < self.threshold:
            return None

        gloss = self.actions[idx]
        if self._last_gloss == gloss:
            return None

        self._last_gloss = gloss
        return GlossPrediction(gloss=gloss, confidence=conf)

