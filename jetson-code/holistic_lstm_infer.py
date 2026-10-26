from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import asl_features as F

# Keypoints come from asl_features.py (MediaPipe Pose + Hands, no face mesh): 258 values per frame.
# That file is shared, unchanged, with the training notebook so training and live inference match.
FEATURE_DIM = F.FEATURE_DIM


def extract_keypoints(pose_result, hands_result) -> np.ndarray:
    """258-value feature vector from MediaPipe Pose and Hands results (see asl_features.py)."""
    return F.extract_keypoints(pose_result, hands_result)


def load_model_meta(model_path: str) -> dict:
    """model_meta.json from the model's folder ({} if there is none).

    The training notebook writes it next to the .h5 with the word list, window, frame stride, frame
    size and decision rules the model was trained and tested with. Raises if it describes a different
    feature layout than this asl_features.py.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "model_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    version, dim = meta.get("feature_version"), meta.get("feature_dim")
    if version not in (None, F.FEATURE_VERSION) or dim not in (None, F.FEATURE_DIM):
        raise RuntimeError(
            "%s says the model uses features %r (%s values) but asl_features.py makes %r (%d values). "
            "Copy the asl_features.py that was exported with the model into jetson-code/."
            % (path, version, dim, F.FEATURE_VERSION, F.FEATURE_DIM)
        )
    meta["_path"] = path
    return meta


def prepare_frame(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Center-crop to the training aspect ratio (never stretch), then resize to `size` (w, h).

    Same geometry as the training notebook and live_local.py, so pose and hand coordinates mean the
    same thing as in the training clips.
    """
    import cv2

    fw, fh = int(size[0]), int(size[1])
    h, w = rgb.shape[:2]
    target = fw / fh
    if abs(w / h - target) > 0.01:
        if w / h > target:
            new_w = int(round(h * target))
            x0 = (w - new_w) // 2
            rgb = rgb[:, x0:x0 + new_w]
        else:
            new_h = int(round(w / target))
            y0 = (h - new_h) // 2
            rgb = rgb[y0:y0 + new_h]
    if rgb.shape[1] != fw or rgb.shape[0] != fh:
        rgb = cv2.resize(rgb, (fw, fh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


class ParallelDetector:
    """asl_features.Detector with Pose and Hands running at the same time (MediaPipe releases the GIL).

    Same solvers and settings, so the feature vector is identical to Detector.process; only the
    wall time drops from pose + hands to max(pose, hands).
    """

    def __init__(self, detector) -> None:
        self.det = detector
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mp-hands")

    def process(self, rgb: np.ndarray) -> np.ndarray:
        det = self.det
        rgb.flags.writeable = False
        hands_future = self._pool.submit(det._hands.process, rgb)
        try:
            pose_result = det._pose.process(rgb)
        finally:
            hands_result = hands_future.result()
            rgb.flags.writeable = True
        return F.extract_keypoints(pose_result, hands_result)

    def close(self) -> None:
        try:
            self.det.close()
        finally:
            self._pool.shutdown(wait=False)


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
        frame_size: Optional[Tuple[int, int]] = None,
        parallel: bool = True,
        predict_fn: Optional[Callable[[np.ndarray], object]] = None,
    ) -> None:
        """
        model_complexity: MediaPipe Pose/Hands complexity. The training notebook uses 0, so keep
            0 unless the model was retrained with another value.
        infer_width: shrink frames to this width before MediaPipe (0 = keep full size). Only used
            when frame_size is not given.
        repeat_reset: after this many non-confident inferences, forget the last gloss so
            the same word can be reported again (0 = never, the old behaviour).
        min_hand_frac: only predict when a hand was detected in at least this share of the
            window's frames (the training notebook only trains on such windows). 0 = off.
        frame_size: (w, h) the model was trained on; frames are center-cropped to that aspect ratio
            and resized to it before MediaPipe, exactly like the training clips.
        parallel: run MediaPipe Pose and Hands in two threads (same result, less wall time).
        predict_fn: optional fast callable x -> probabilities (e.g. a tf.function); default model(x).
        """
        self.model = model
        self.predict_fn = predict_fn
        self.actions = list(actions)
        self.sequence_length = int(sequence_length)
        self.threshold = float(threshold)
        self.stable_n = int(stable_n)
        self.infer_width = int(infer_width)
        self.frame_size = tuple(int(v) for v in frame_size) if frame_size else None
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
        if parallel:
            self._detector = ParallelDetector(self._detector)

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
        if getattr(self, "predict_fn", None) is not None:
            output = self.predict_fn(x)
        else:
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
        if getattr(self, "frame_size", None):
            img = prepare_frame(img, self.frame_size)
        elif self._cv2 is not None and img.shape[1] > self.infer_width:
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
