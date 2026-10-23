from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

POSE_DIMS = 33 * 4
FACE_DIMS = 468 * 3
HAND_DIMS = 21 * 3
KEYPOINT_DIMS = POSE_DIMS + FACE_DIMS + 2 * HAND_DIMS  # 1662


def _landmark_array(landmark_list, count: int, with_visibility: bool) -> np.ndarray:
    """Flatten a MediaPipe landmark list; zeros if the part was not detected."""
    width = 4 if with_visibility else 3
    if landmark_list is None:
        return np.zeros(count * width)
    if with_visibility:
        values = chain.from_iterable((p.x, p.y, p.z, p.visibility) for p in landmark_list.landmark)
    else:
        values = chain.from_iterable((p.x, p.y, p.z) for p in landmark_list.landmark)
    return np.fromiter(values, dtype=np.float64, count=count * width)


def extract_keypoints(results) -> np.ndarray:
    """
    Match Action Detection Refined.ipynb keypoint layout (1662 dims):
    - pose: 33 * (x,y,z,visibility) = 132
    - face: 468 * (x,y,z) = 1404
    - left hand: 21 * (x,y,z) = 63
    - right hand: 21 * (x,y,z) = 63
    """
    return np.concatenate(
        [
            _landmark_array(results.pose_landmarks, 33, True),
            _landmark_array(results.face_landmarks, 468, False),
            _landmark_array(results.left_hand_landmarks, 21, False),
            _landmark_array(results.right_hand_landmarks, 21, False),
        ]
    )


def landmarks_from_results(results, face_stride: int = 3) -> dict:
    """Normalized (x, y) points for drawing a skeleton preview.

    Only used when the preview is on; it does not affect the model input.
    Keys: pose, face (every `face_stride`-th point), lh, rh. Missing parts are [].
    """

    def points(landmarks, stride: int = 1):
        if landmarks is None:
            return []
        return [(p.x, p.y) for i, p in enumerate(landmarks.landmark) if i % stride == 0]

    return {
        "pose": points(results.pose_landmarks),
        "face": points(results.face_landmarks, face_stride),
        "lh": points(results.left_hand_landmarks),
        "rh": points(results.right_hand_landmarks),
    }


def landmarks_from_keypoints(
    keypoints: np.ndarray,
    present: Tuple[bool, bool, bool, bool],
    face_stride: int = 3,
) -> dict:
    """Same output as `landmarks_from_results`, built from the model's keypoint
    vector so the (slow) MediaPipe protobuf objects are not walked a second time.

    `present` is (pose, face, left_hand, right_hand) detection flags.
    """
    has_pose, has_face, has_lh, has_rh = present
    pose_end = POSE_DIMS
    face_end = pose_end + FACE_DIMS
    lh_end = face_end + HAND_DIMS
    rh_end = lh_end + HAND_DIMS
    return {
        "pose": keypoints[:pose_end].reshape(33, 4)[:, :2].tolist() if has_pose else [],
        "face": (
            keypoints[pose_end:face_end].reshape(468, 3)[::face_stride, :2].tolist() if has_face else []
        ),
        "lh": keypoints[face_end:lh_end].reshape(21, 3)[:, :2].tolist() if has_lh else [],
        "rh": keypoints[lh_end:rh_end].reshape(21, 3)[:, :2].tolist() if has_rh else [],
    }


@dataclass(frozen=True)
class GlossPrediction:
    gloss: str
    confidence: float


class StageTimer:
    """Thread-safe running averages (ms) for the stats line."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sum: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self._sum[name] = self._sum.get(name, 0.0) + seconds
            self._count[name] = self._count.get(name, 0) + 1

    def summary_and_reset(self) -> str:
        with self._lock:
            parts = [
                f"{name}={self._sum[name] / self._count[name] * 1000.0:.1f}ms"
                for name in self._sum
                if self._count.get(name)
            ]
            self._sum.clear()
            self._count.clear()
        return " ".join(parts)


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
        model_complexity: int = 1,
        infer_width: int = 0,
        repeat_reset: int = 0,
    ) -> None:
        """
        model_complexity: MediaPipe Holistic complexity (0 fastest, 1 default, 2 slowest).
        infer_width: shrink frames to this width before MediaPipe (0 = keep full size).
            Landmarks are normalized, so the model input is unchanged.
        repeat_reset: after this many non-confident inferences, forget the last gloss so
            the same word can be reported again (0 = never, the old behaviour).
        """
        self.model = model
        self.actions = list(actions)
        self.sequence_length = int(sequence_length)
        self.threshold = float(threshold)
        self.stable_n = int(stable_n)
        self.infer_width = int(infer_width)
        self.repeat_reset = int(repeat_reset)

        import mediapipe as mp  # local import for easier install/debug

        self._cv2 = None
        if self.infer_width > 0:
            import cv2

            self._cv2 = cv2

        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=int(model_complexity),
        )

        self._sequence: List[np.ndarray] = []
        self._predictions: List[int] = []
        self._last_gloss: Optional[str] = None
        self._idle = 0
        # Latest class probabilities and skeleton, read by the preview.
        self.last_probs: Optional[np.ndarray] = None
        self.last_keypoints: Optional[np.ndarray] = None
        self.last_present: Tuple[bool, bool, bool, bool] = (False, False, False, False)
        self.timer = StageTimer()

    def close(self) -> None:
        try:
            self._holistic.close()
        except Exception:
            pass

    def preview_landmarks(self, face_stride: int = 3) -> Optional[dict]:
        if self.last_keypoints is None:
            return None
        return landmarks_from_keypoints(self.last_keypoints, self.last_present, face_stride)

    def _predict(self, sequence: List[np.ndarray]) -> Tuple[int, float, np.ndarray]:
        x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)  # (1, T, 1662)
        # Direct eager invocation avoids Keras predict()'s per-call data adapter.
        output = self.model(x, training=False)
        res = np.asarray(output)[0]
        idx = int(np.argmax(res))
        conf = float(res[idx])
        return idx, conf, res

    def _note_idle(self) -> None:
        """Count inferences with no confident new word; forget the last gloss after a gap."""
        if self.repeat_reset > 0:
            self._idle += 1
            if self._idle >= self.repeat_reset:
                self._last_gloss = None

    def push_frame(self, rgb_frame: np.ndarray) -> Optional[GlossPrediction]:
        """
        Push one RGB frame (H,W,3 uint8). Returns a new stable gloss prediction
        when conditions are met, otherwise None.
        """
        t0 = time.perf_counter()
        img = rgb_frame
        if self._cv2 is not None and img.shape[1] > self.infer_width:
            height = max(int(round(img.shape[0] * self.infer_width / img.shape[1])), 1)
            img = self._cv2.resize(
                img, (self.infer_width, height), interpolation=self._cv2.INTER_AREA
            )
        t1 = time.perf_counter()

        # MediaPipe expects RGB.
        img.flags.writeable = False
        results = self._holistic.process(img)
        img.flags.writeable = True
        t2 = time.perf_counter()

        keypoints = extract_keypoints(results)
        self.last_keypoints = keypoints
        self.last_present = (
            results.pose_landmarks is not None,
            results.face_landmarks is not None,
            results.left_hand_landmarks is not None,
            results.right_hand_landmarks is not None,
        )
        self._sequence.append(keypoints)
        self._sequence = self._sequence[-self.sequence_length :]
        t3 = time.perf_counter()
        self.timer.add("resize", t1 - t0)
        self.timer.add("mediapipe", t2 - t1)
        self.timer.add("keypoints", t3 - t2)

        if len(self._sequence) < self.sequence_length:
            return None

        t4 = time.perf_counter()
        idx, conf, probs = self._predict(self._sequence)
        self.last_probs = probs
        self.timer.add("lstm", time.perf_counter() - t4)
        self._predictions.append(idx)
        self._predictions = self._predictions[-max(self.stable_n, 1) :]

        # Notebook stability logic: last N predictions agree and pass threshold.
        if self.stable_n > 1:
            if len(self._predictions) < self.stable_n:
                self._note_idle()
                return None
            if len(set(self._predictions[-self.stable_n :])) != 1:
                self._note_idle()
                return None

        if conf < self.threshold:
            self._note_idle()
            return None

        gloss = self.actions[idx]
        if self._last_gloss == gloss:
            # Still holding the same sign; not a gap.
            self._idle = 0
            return None

        self._last_gloss = gloss
        self._idle = 0
        return GlossPrediction(gloss=gloss, confidence=conf)
