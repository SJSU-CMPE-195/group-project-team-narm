import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asl_features as F  # noqa: E402
import holistic_lstm_infer as hli  # noqa: E402
import main as jetson_main  # noqa: E402


class P:
    def __init__(self, x, y, z, v=0.9):
        self.x, self.y, self.z, self.visibility = x, y, z, v


class LM:
    def __init__(self, n, seed, x0=None):
        rng = np.random.default_rng(seed)
        self.landmark = [P(*rng.random(3), rng.random()) for _ in range(n)]
        if x0 is not None:  # control the wrist x position (used for the same-label fallback)
            self.landmark[0].x = x0


class Handed:
    def __init__(self, label, score=0.9):
        self.classification = [types.SimpleNamespace(label=label, score=score)]


def pose_result(present=True):
    return types.SimpleNamespace(pose_landmarks=LM(33, 1) if present else None)


def hands_result(*hands):
    """hands: (label, landmark_list) pairs."""
    return types.SimpleNamespace(
        multi_hand_landmarks=[h[1] for h in hands] or None,
        multi_handedness=[Handed(h[0]) for h in hands] or None,
    )


def flat(lm, with_vis):
    cols = (lambda p: [p.x, p.y, p.z, p.visibility]) if with_vis else (lambda p: [p.x, p.y, p.z])
    return np.array([cols(p) for p in lm.landmark], dtype=np.float32).flatten()


class KeypointTests(unittest.TestCase):
    def test_feature_layout_constants(self):
        self.assertEqual(F.FEATURE_DIM, 258)
        self.assertEqual(hli.FEATURE_DIM, 258)
        self.assertEqual(F.FEATURE_DIM, 33 * 4 + 21 * 3 * 2)  # no face block

    def test_labels_go_to_fixed_slots(self):
        left, right = LM(21, 3), LM(21, 4)
        # order in the MediaPipe result must not matter
        for hands in ([("Left", left), ("Right", right)], [("Right", right), ("Left", left)]):
            kp = hli.extract_keypoints(pose_result(), hands_result(*hands))
            self.assertEqual(kp.shape, (258,))
            np.testing.assert_array_equal(kp[F.POSE_SLICE], flat(LM(33, 1), True))
            np.testing.assert_array_equal(kp[F.HAND_A_SLICE], flat(left, False))
            np.testing.assert_array_equal(kp[F.HAND_B_SLICE], flat(right, False))

    def test_missing_parts_are_zero(self):
        kp = hli.extract_keypoints(pose_result(False), hands_result())
        self.assertEqual(kp.shape, (258,))
        self.assertFalse(np.any(kp))
        self.assertFalse(F.hands_present(kp))
        only_right = hli.extract_keypoints(pose_result(), hands_result(("Right", LM(21, 4))))
        self.assertFalse(np.any(only_right[F.HAND_A_SLICE]))
        self.assertTrue(np.any(only_right[F.HAND_B_SLICE]))
        self.assertTrue(F.hands_present(only_right))

    def test_same_label_twice_falls_back_to_image_position(self):
        a, b = LM(21, 3, x0=0.2), LM(21, 4, x0=0.8)
        kp = hli.extract_keypoints(pose_result(), hands_result(("Right", b), ("Right", a)))
        np.testing.assert_array_equal(kp[F.HAND_A_SLICE], flat(a, False))  # further left -> slot A
        np.testing.assert_array_equal(kp[F.HAND_B_SLICE], flat(b, False))

    def test_preview_landmarks_from_keypoints(self):
        left = LM(21, 3)
        kp = hli.extract_keypoints(pose_result(), hands_result(("Left", left)))
        marks = hli.landmarks_from_keypoints(kp)
        self.assertEqual(set(marks), {"pose", "lh", "rh"})
        self.assertEqual(len(marks["pose"]), 33)
        self.assertEqual(len(marks["lh"]), 21)
        self.assertEqual(marks["rh"], [])
        np.testing.assert_allclose(
            np.array(marks["lh"]), np.array([[p.x, p.y] for p in left.landmark]), rtol=1e-6
        )


class FakeModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape


class ValidateModelTests(unittest.TestCase):
    actions = ["hello", "thanks", "iloveyou"]

    def test_accepts_matching_model(self):
        hli.validate_model(FakeModel((None, 30, 258), (None, 3)), 30, self.actions)

    def test_old_holistic_model_is_rejected_with_hint(self):
        with self.assertRaises(RuntimeError) as ctx:
            hli.validate_model(FakeModel((None, 30, 1662), (None, 3)), 30, self.actions)
        self.assertIn("1662", str(ctx.exception))
        self.assertIn("retrained", str(ctx.exception))

    def test_wrong_window_length_is_rejected(self):
        with self.assertRaises(RuntimeError):
            hli.validate_model(FakeModel((None, 45, 258), (None, 3)), 30, self.actions)

    def test_wrong_number_of_words_is_rejected(self):
        with self.assertRaises(RuntimeError):
            hli.validate_model(FakeModel((None, 30, 258), (None, 5)), 30, self.actions)


class FrameSamplerTests(unittest.TestCase):
    def test_stride_two_keeps_every_other_frame(self):
        s = jetson_main.FrameSampler(2)
        self.assertEqual([s.keep() for _ in range(6)], [True, False, True, False, True, False])

    def test_stride_one_keeps_everything_and_bad_values_are_safe(self):
        for stride in (1, 0, -3):
            s = jetson_main.FrameSampler(stride)
            self.assertTrue(all(s.keep() for _ in range(5)))


def features(hand=True, seed=0):
    kp = np.zeros(F.FEATURE_DIM, dtype=np.float32)
    kp[F.POSE_SLICE] = np.random.default_rng(seed).random(F.POSE_DIMS)
    if hand:
        kp[F.HAND_A_SLICE] = np.random.default_rng(seed + 1).random(F.HAND_DIMS) + 0.1
    return kp


class FakeDetector:
    def __init__(self, hands=True):
        self.seen_shapes = []
        self.hands = hands

    def process(self, img):
        self.seen_shapes.append(img.shape)
        return features(hand=self.hands() if callable(self.hands) else self.hands)


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
        obj.min_hand_frac = kwargs.get("min_hand_frac", 0.0)
        obj.frame_size = kwargs.get("frame_size")
        obj.predict_fn = kwargs.get("predict_fn")
        obj._cv2 = None
        if obj.infer_width:
            import cv2

            obj._cv2 = cv2
        obj._detector = FakeDetector(kwargs.get("hands", True))
        obj._sequence, obj._hands, obj._predictions = [], [], []
        obj._last_gloss, obj._idle = None, 0
        obj.last_probs = None
        obj.last_keypoints = None
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
        self.assertEqual(infer._detector.seen_shapes[0], (360, 640, 3))
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertTrue(frame.flags.writeable)

    def test_landmarks_and_timing_available(self):
        infer = self.build()
        infer.push_frame(np.zeros((90, 160, 3), np.uint8))
        marks = infer.preview_landmarks()
        self.assertEqual(len(marks["pose"]), 33)
        self.assertEqual(len(marks["lh"]), 21)
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

    def test_no_hands_means_no_prediction(self):
        calls = []
        infer = self.build(min_hand_frac=0.6, hands=False)
        infer.model = lambda x, training=False: calls.append(x.shape) or np.array([[0.1, 0.8, 0.1]], np.float32)
        self.assertEqual(self.feed(infer, 12), [])
        self.assertEqual(calls, [])  # the LSTM is not even run
        self.assertEqual(infer._predictions, [])

    def test_hands_return_and_word_is_reported(self):
        state = {"hands": False}
        infer = self.build(min_hand_frac=0.6, hands=lambda: state["hands"])
        self.assertEqual(self.feed(infer, 6), [])
        state["hands"] = True
        # needs the window to fill with hand frames, then stable_n agreeing windows
        self.assertEqual(self.feed(infer, 12), ["thanks"])

    def test_hand_gate_off_still_predicts_without_hands(self):
        infer = self.build(min_hand_frac=0.0, hands=False)
        self.assertEqual(self.feed(infer, 6), ["thanks"])

    def test_hand_gate_threshold_uses_share_of_window(self):
        # sequence_length=3: with 2 of 3 frames showing a hand the share is 0.67 >= 0.6, with 1 of 3 it is 0.33
        pattern = iter([True, True, False, True, False, False, False])
        infer = self.build(min_hand_frac=0.6, hands=lambda: next(pattern))
        calls = []
        infer.model = lambda x, training=False: calls.append(1) or np.array([[0.1, 0.8, 0.1]], np.float32)
        for _ in range(7):
            infer.push_frame(np.zeros((90, 160, 3), np.uint8))
        # windows (3 frames) after frames 3..7 -> hand shares 2/3, 2/3, 1/3, 1/3, 0 -> only the first two predict
        self.assertEqual(len(calls), 2)

    def test_frame_size_crops_to_training_geometry(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")
        infer = self.build(frame_size=(640, 360), infer_width=0)
        infer.push_frame(np.zeros((480, 640, 3), np.uint8))   # 4:3 -> cropped, not stretched
        infer.push_frame(np.zeros((1080, 1920, 3), np.uint8))  # 16:9 -> resized only
        self.assertEqual(infer._detector.seen_shapes, [(360, 640, 3), (360, 640, 3)])

    def test_predict_fn_is_used_instead_of_model(self):
        calls = []

        def fast(x):
            calls.append(x.shape)
            return np.array([[0.1, 0.8, 0.1]], np.float32)

        infer = self.build(predict_fn=fast)
        infer.model = lambda x, training=False: self.fail("eager model should not be called")
        self.assertEqual(self.feed(infer, 4), ["thanks"])
        self.assertEqual(calls[0], (1, 3, F.FEATURE_DIM))


class PrepareFrameTests(unittest.TestCase):
    def setUp(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")

    def test_wide_frame_is_center_cropped_left_right(self):
        frame = np.zeros((360, 800, 3), np.uint8)
        frame[:, 80:720] = 200          # the centered 640 columns
        out = hli.prepare_frame(frame, (640, 360))
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertTrue(np.all(out == 200))

    def test_tall_frame_is_center_cropped_top_bottom(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[60:420] = 200             # the centered 360 rows
        out = hli.prepare_frame(frame, (640, 360))
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertTrue(np.all(out == 200))

    def test_same_aspect_is_only_resized(self):
        out = hli.prepare_frame(np.full((1080, 1920, 3), 7, np.uint8), (640, 360))
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertTrue(np.all(out == 7))


class FakeSolver:
    def __init__(self, result, delay=0.0):
        self.result, self.delay, self.writeable_seen = result, delay, []

    def process(self, img):
        import time

        self.writeable_seen.append(img.flags.writeable)
        time.sleep(self.delay)
        return self.result


class ParallelDetectorTests(unittest.TestCase):
    def test_same_features_as_serial_and_frame_restored(self):
        det = types.SimpleNamespace(
            _pose=FakeSolver(pose_result(), delay=0.05),
            _hands=FakeSolver(hands_result(("Left", LM(21, 3))), delay=0.05),
            close=lambda: None,
        )
        par = hli.ParallelDetector(det)
        frame = np.zeros((360, 640, 3), np.uint8)
        import time

        t0 = time.perf_counter()
        kp = par.process(frame)
        elapsed = time.perf_counter() - t0
        par.close()
        serial = hli.extract_keypoints(det._pose.result, det._hands.result)
        np.testing.assert_array_equal(kp, serial)
        self.assertTrue(frame.flags.writeable)
        self.assertEqual(det._pose.writeable_seen + det._hands.writeable_seen, [False, False])
        self.assertLess(elapsed, 0.095)  # both 50 ms solvers ran at the same time


class ModelMetaTests(unittest.TestCase):
    def write(self, folder, meta):
        import json
        import os

        model = os.path.join(folder, "m.h5")
        open(model, "wb").close()
        if meta is not None:
            with open(os.path.join(folder, "model_meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
        return model

    def test_missing_meta_gives_empty_dict(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(hli.load_model_meta(self.write(d, None)), {})

    def test_other_feature_layout_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            model = self.write(d, {"feature_version": "holistic", "feature_dim": 1662})
            with self.assertRaises(RuntimeError):
                hli.load_model_meta(model)

    def test_settings_come_from_meta_and_env_overrides(self):
        import os
        import tempfile
        from unittest import mock

        meta = {"actions": ["a", "b"], "window": 20, "stable_n": 6, "threshold": 0.75,
                "min_hand_frac": 0.5, "frame_stride": 2, "frame_size": [640, 360],
                "feature_version": F.FEATURE_VERSION, "feature_dim": F.FEATURE_DIM}
        with tempfile.TemporaryDirectory() as d:
            loaded = hli.load_model_meta(self.write(d, meta))
        keys = ("ACTIONS", "SEQUENCE_LENGTH", "STABLE_N", "THRESHOLD", "MIN_HAND_FRAC",
                "FRAME_STRIDE", "MODEL_COMPLEXITY", "FRAME_SIZE")
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with mock.patch.dict(os.environ, clean, clear=True):
            s = jetson_main.model_settings(loaded)
            self.assertEqual(s["actions"], ["a", "b"])
            self.assertEqual((s["sequence_length"], s["stable_n"], s["threshold"]), (20, 6, 0.75))
            self.assertEqual((s["min_hand_frac"], s["frame_stride"], s["frame_size"]), (0.5, 2, (640, 360)))
            with mock.patch.dict(os.environ, {"THRESHOLD": "0.9", "STABLE_N": "9", "FRAME_SIZE": "0"}):
                s = jetson_main.model_settings(loaded)
                self.assertEqual((s["threshold"], s["stable_n"], s["frame_size"]), (0.9, 9, None))

    def test_defaults_without_meta_are_the_old_ones(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            s = jetson_main.model_settings({})
        self.assertEqual(s["actions"], ["hello", "thanks", "iloveyou"])
        self.assertEqual((s["sequence_length"], s["stable_n"], s["frame_size"]), (30, 10, None))


if __name__ == "__main__":
    unittest.main()
