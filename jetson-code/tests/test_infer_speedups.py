import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holistic_lstm_infer as hli  # noqa: E402


class P:
    def __init__(self, x, y, z, v=0.9):
        self.x, self.y, self.z, self.visibility = x, y, z, v


class LM:
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.landmark = [P(*rng.random(3), rng.random()) for _ in range(n)]


def old_extract(results):
    """The original per-point implementation, kept as the reference."""
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


def make_results(pose=True, face=True, lh=True, rh=False):
    return types.SimpleNamespace(
        pose_landmarks=LM(33, 1) if pose else None,
        face_landmarks=LM(468, 2) if face else None,
        left_hand_landmarks=LM(21, 3) if lh else None,
        right_hand_landmarks=LM(21, 4) if rh else None,
    )


class KeypointTests(unittest.TestCase):
    def test_extract_matches_original(self):
        for flags in [(True, True, True, True), (True, False, False, True), (False, False, False, False)]:
            res = make_results(*flags)
            new = hli.extract_keypoints(res)
            self.assertEqual(new.shape, (1662,))
            np.testing.assert_array_equal(new, old_extract(res))

    def test_preview_landmarks_match_protobuf_version(self):
        res = make_results(True, True, True, False)
        kp = hli.extract_keypoints(res)
        present = (True, True, True, False)
        fast = hli.landmarks_from_keypoints(kp, present)
        ref = hli.landmarks_from_results(res)
        self.assertEqual(set(fast), set(ref))
        for key in ref:
            self.assertEqual(len(fast[key]), len(ref[key]), key)
            if ref[key]:
                np.testing.assert_allclose(np.array(fast[key]), np.array(ref[key]))


class FakeHolistic:
    def __init__(self):
        self.seen_shapes = []

    def process(self, img):
        self.seen_shapes.append(img.shape)
        return make_results(True, True, True, False)


class PushFrameTests(unittest.TestCase):
    def build(self, **kwargs):
        # Skip __init__ (needs mediapipe); set the fields push_frame uses.
        obj = hli.HolisticLSTMInfer.__new__(hli.HolisticLSTMInfer)
        obj.model = lambda x, training=False: np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        obj.actions = ["hello", "thanks", "iloveyou"]
        obj.sequence_length = 3
        obj.threshold = 0.5
        obj.stable_n = 2
        obj.infer_width = kwargs.get("infer_width", 0)
        obj.repeat_reset = kwargs.get("repeat_reset", 0)
        obj._cv2 = None
        if obj.infer_width:
            import cv2

            obj._cv2 = cv2
        obj._holistic = FakeHolistic()
        obj._sequence, obj._predictions = [], []
        obj._last_gloss, obj._idle = None, 0
        obj.last_probs = None
        obj.last_keypoints = None
        obj.last_present = (False, False, False, False)
        obj.timer = hli.StageTimer()
        return obj

    def test_downscale_before_mediapipe(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")
        infer = self.build(infer_width=640)
        frame = np.zeros((1080, 1920, 3), np.uint8)
        infer.push_frame(frame)
        self.assertEqual(infer._holistic.seen_shapes[0], (360, 640, 3))
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertTrue(frame.flags.writeable)

    def test_landmarks_and_timing_available(self):
        infer = self.build()
        infer.push_frame(np.zeros((90, 160, 3), np.uint8))
        marks = infer.preview_landmarks()
        self.assertEqual(len(marks["pose"]), 33)
        self.assertEqual(len(marks["rh"]), 0)
        self.assertIn("mediapipe=", infer.timer.summary_and_reset())

    def feed(self, infer, n):
        out = []
        for _ in range(n):
            out.append(infer.push_frame(np.zeros((90, 160, 3), np.uint8)))
        return [o.gloss for o in out if o]

    def test_same_word_suppressed_without_reset(self):
        infer = self.build(repeat_reset=0)
        self.assertEqual(self.feed(infer, 12), ["thanks"])

    def test_same_word_repeats_after_gap(self):
        infer = self.build(repeat_reset=3)
        glosses = self.feed(infer, 6)
        self.assertEqual(glosses, ["thanks"])
        # Simulate a gap of non-confident inferences, then the sign again.
        for _ in range(3):
            infer._note_idle()
        self.assertIsNone(infer._last_gloss)
        self.assertEqual(self.feed(infer, 3), ["thanks"])


if __name__ == "__main__":
    unittest.main()
