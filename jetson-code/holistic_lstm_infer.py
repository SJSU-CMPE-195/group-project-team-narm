from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import asl_features as F

# Keypoints come from asl_features.py (MediaPipe Pose + Hands, no face mesh): 258 values per frame.
# That file is shared, unchanged, with the training notebook so training and live inference match.
FEATURE_DIM = F.FEATURE_DIM


def extract_keypoints(pose_result, hands_result) -> np.ndarray:
    """258-value feature vector from MediaPipe Pose and Hands results (see asl_features.py)."""
    return F.extract_keypoints(pose_result, hands_result)


def validate_model(model, sequence_length: int, actions: Sequence[str]) -> None:
    """Fail at startup, with a readable message, if the model does not fit this pipeline."""
    input_shape = getattr(model, "input_shape", None)
    output_shape = getattr(model, "output_shape", None)
    if isinstance(input_shape, list):  # multi-input models are not supported
        raise RuntimeError("Expected a single-input model, got %r" % (input_shape,))
    if input_shape is not None:
        features = input_shape[-1]
        if features != FEATURE_DIM:
            hint = (
                " That is the old Holistic model (pose + face + hands); it must be retrained with "
                "the no-face notebook, or run with the old code."
                if features == 1662
                else ""
            )
            raise RuntimeError(
                "Model expects %s features per frame but this pipeline produces %d "
                "(%s).%s" % (features, FEATURE_DIM, F.FEATURE_VERSION, hint)
            )
        steps = input_shape[1] if len(input_shape) >= 3 else None
        if steps is not None and int(steps) != int(sequence_length):
            raise RuntimeError(
                "Model expects %s frames per input but SEQUENCE_LENGTH is %d." % (steps, sequence_length)
            )
    if output_shape is not None and int(output_shape[-1]) != len(actions):
        raise RuntimeError(
            "Model has %s outputs but ACTIONS lists %d words (%s). ACTIONS must be in the "
            "same order as in training." % (output_shape[-1], len(actions), ",".join(actions))
        )


def landmarks_from_keypoints(keypoints: np.ndarray) -> dict:
    """Normalized (x, y) points for drawing a skeleton preview, built from the model's feature vector
    (so the MediaPipe protobuf objects are not walked a second time).

    Keys: pose, lh (hand slot A), rh (hand slot B). A part that was not detected is [].
    Only used when the preview is on; it does not affect the model input.
    """
    pose = keypoints[F.POSE_SLICE]
    hand_a = keypoints[F.HAND_A_SLICE]
    hand_b = keypoints[F.HAND_B_SLICE]
    return {
        "pose": pose.reshape(F.POSE_COUNT, 4)[:, :2].tolist() if np.any(pose != 0) else [],
        "lh": hand_a.reshape(F.HAND_COUNT, 3)[:, :2].tolist() if np.any(hand_a != 0) else [],
        "rh": hand_b.reshape(F.HAND_COUNT, 3)[:, :2].tolist() if np.any(hand_b != 0) else [],
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
    """Keypoints (MediaPipe Pose + Hands) -> LSTM -> stable gloss.

    The class keeps its old name so main.py and the tests keep working; it no longer runs
    MediaPipe Holistic or reads face landmarks.
    """

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
        model_complexity: int = 0,
        infer_width: int = 0,
        repeat_reset: int = 0,
        min_hand_frac: float = 0.0,
    ) -> None:
        """
        model_complexity: MediaPipe Pose/Hands complexity. The training notebook uses 0, so keep
            0 unless the model was retrained with another value.
        infer_width: shrink frames to this width before MediaPipe (0 = keep full size).
            Landmarks are normalized, so the model input is unchanged.
        repeat_reset: after this many non-confident inferences, forget the last gloss so
            the same word can be reported again (0 = never, the old behaviour).
        min_hand_frac: only predict when a hand was detected in at least this share of the
            window's frames (the training notebook only trains on such windows). 0 = off.
        """
        self.model = model
        self.actions = list(actions)
        self.sequence_length = int(sequence_length)
        self.threshold = float(threshold)
        self.stable_n = int(stable_n)
        self.infer_width = int(infer_width)
        self.repeat_reset = int(repeat_reset)
        self.min_hand_frac = float(min_hand_frac)

        self._cv2 = None
        if self.infer_width > 0:
            import cv2

            self._cv2 = cv2

        self._detector = F.Detector(
            model_complexity=int(model_complexity),
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._sequence: List[np.ndarray] = []
        self._hands: List[bool] = []
        self._predictions: List[int] = []
        self._last_gloss: Optional[str] = None
        self._idle = 0
        # Latest class probabilities and skeleton, read by the preview.
        self.last_probs: Optional[np.ndarray] = None
        self.last_keypoints: Optional[np.ndarray] = None
        self.timer = StageTimer()

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass

    def preview_landmarks(self) -> Optional[dict]:
        if self.last_keypoints is None:
            return None
        return landmarks_from_keypoints(self.last_keypoints)

    def _predict(self, sequence: List[np.ndarray]) -> Tuple[int, float, np.ndarray]:
        x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)  # (1, T, FEATURE_DIM)
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

    def _hands_ok(self) -> bool:
        if self.min_hand_frac <= 0:
            return True
        return (sum(self._hands) / max(len(self._hands), 1)) >= self.min_hand_frac

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

        # MediaPipe expects RGB. The frame must not be mirrored (training used unmirrored frames).
        keypoints = self._detector.process(img)
        t2 = time.perf_counter()

        self.last_keypoints = keypoints
        self._sequence.append(keypoints)
        self._sequence = self._sequence[-self.sequence_length :]
        self._hands.append(F.hands_present(keypoints))
        self._hands = self._hands[-self.sequence_length :]
        self.timer.add("resize", t1 - t0)
        self.timer.add("mediapipe", t2 - t1)

        if len(self._sequence) < self.sequence_length:
            return None

        # Same rule as the training windows: not enough hand frames -> no prediction, and the
        # agreement streak starts over (so a word needs stable_n fresh windows with hands).
        if not self._hands_ok():
            self._predictions = []
            self.last_probs = None
            self._note_idle()
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
