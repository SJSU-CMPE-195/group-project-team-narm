"""Shared keypoint extractor: MediaPipe Pose + Hands, NO face mesh.

This ONE file is used by the training notebook AND must be copied unchanged to the Jetson
(jetson-code/), so the features the model is trained on are exactly the features it sees live.

Feature vector per frame (FEATURE_DIM = 258 values, always this order):
    pose        33 landmarks x (x, y, z, visibility)   132   [0:132]
    hand slot A 21 landmarks x (x, y, z)                63   [132:195]  MediaPipe label "Left"
    hand slot B 21 landmarks x (x, y, z)                63   [195:258]  MediaPipe label "Right"
A part that is not detected is all zeros.

Hand slots: MediaPipe Hands assumes a mirrored (selfie) image. We feed it UNMIRRORED frames
(laptop webcam while training, glasses camera on the Jetson), so its "Left" label is the hand that
appears on the image's LEFT, i.e. the signer's right hand. The labels are only used to keep each hand
in a consistent slot; both training and the Jetson see the same unmirrored geometry, so they agree.
Never flip frames before extracting.

Compatible with Python 3.8+ and MediaPipe 0.10.x (mp.solutions API).
"""
from typing import Optional, Tuple

import numpy as np

FEATURE_VERSION = "pose-hands-v1"

POSE_COUNT = 33
HAND_COUNT = 21
POSE_DIMS = POSE_COUNT * 4          # 132
HAND_DIMS = HAND_COUNT * 3          # 63
FEATURE_DIM = POSE_DIMS + 2 * HAND_DIMS  # 258

POSE_SLICE = slice(0, POSE_DIMS)
HAND_A_SLICE = slice(POSE_DIMS, POSE_DIMS + HAND_DIMS)
HAND_B_SLICE = slice(POSE_DIMS + HAND_DIMS, FEATURE_DIM)
HANDS_SLICE = slice(POSE_DIMS, FEATURE_DIM)


def _hand_array(hand_landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32).flatten()


def _pose_array(pose_landmarks) -> np.ndarray:
    return np.array(
        [[p.x, p.y, p.z, p.visibility] for p in pose_landmarks.landmark], dtype=np.float32
    ).flatten()


def assign_hands(multi_hand_landmarks, multi_handedness) -> Tuple[Optional[object], Optional[object]]:
    """Return (slot_A_hand, slot_B_hand) from a MediaPipe Hands result. Either may be None."""
    if not multi_hand_landmarks:
        return None, None
    hands = []
    for i, lm in enumerate(multi_hand_landmarks):
        label, score = "Left", 0.0
        if multi_handedness and i < len(multi_handedness):
            top = multi_handedness[i].classification[0]
            label, score = top.label, float(top.score)
        hands.append({"lm": lm, "label": label, "score": score, "x": float(lm.landmark[0].x)})

    slot_a = [h for h in hands if h["label"] == "Left"]
    slot_b = [h for h in hands if h["label"] != "Left"]
    if len(slot_a) > 1 or len(slot_b) > 1:
        # Two hands got the same label (a misclassification): fall back to image position,
        # the hand further left in the image goes to slot A.
        hands.sort(key=lambda h: h["x"])
        if len(hands) == 1:
            return hands[0]["lm"], None
        return hands[0]["lm"], hands[1]["lm"]
    return (slot_a[0]["lm"] if slot_a else None), (slot_b[0]["lm"] if slot_b else None)


def extract_keypoints(pose_result, hands_result) -> np.ndarray:
    """Build the 258-value feature vector from MediaPipe Pose and Hands results."""
    pose = (
        _pose_array(pose_result.pose_landmarks)
        if pose_result.pose_landmarks
        else np.zeros(POSE_DIMS, dtype=np.float32)
    )
    a, b = assign_hands(hands_result.multi_hand_landmarks, hands_result.multi_handedness)
    hand_a = _hand_array(a) if a is not None else np.zeros(HAND_DIMS, dtype=np.float32)
    hand_b = _hand_array(b) if b is not None else np.zeros(HAND_DIMS, dtype=np.float32)
    return np.concatenate([pose, hand_a, hand_b]).astype(np.float32)


def hands_present(features: np.ndarray) -> bool:
    """True if at least one hand was detected in this frame's feature vector."""
    return bool(np.any(features[HANDS_SLICE] != 0))


class Detector:
    """Pose + Hands, configured identically for training and the Jetson.

    Use a NEW Detector for each video clip (it tracks between frames), and feed RGB frames in time order.
    """

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp  # imported here so this file can be inspected without MediaPipe

        self._init_args = (model_complexity, min_detection_confidence, min_tracking_confidence)
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 frame (H, W, 3) -> feature vector (FEATURE_DIM,)."""
        rgb.flags.writeable = False
        pose_result = self._pose.process(rgb)
        hands_result = self._hands.process(rgb)
        rgb.flags.writeable = True
        return extract_keypoints(pose_result, hands_result)

    def reset(self) -> None:
        """Forget tracking state (call between separate recordings on the Jetson)."""
        self.close()
        self.__init__(*self._init_args)

    def close(self) -> None:
        for solution in (self._pose, self._hands):
            try:
                solution.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
