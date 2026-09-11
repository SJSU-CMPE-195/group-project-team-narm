import importlib.util
from pathlib import Path
import queue
import sys
import unittest


JETSON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JETSON_DIR))
SPEC = importlib.util.spec_from_file_location("asl_pipeline_main", JETSON_DIR / "main.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


class QueuePolicyTests(unittest.TestCase):
    def test_latest_queue_evicts_oldest_decoded_frame(self):
        target = queue.Queue(maxsize=2)

        self.assertFalse(PIPELINE.put_latest(target, "frame-1"))
        self.assertFalse(PIPELINE.put_latest(target, "frame-2"))
        self.assertTrue(PIPELINE.put_latest(target, "frame-3"))

        self.assertEqual(target.get_nowait(), "frame-2")
        self.assertEqual(target.get_nowait(), "frame-3")

    def test_stats_updates_are_visible_in_snapshot(self):
        stats = PIPELINE.PipelineStats()
        stats.add("payloads")
        stats.add("payload_bytes", 4096)
        stats.set_latency(12.5)

        snapshot = stats.snapshot()
        self.assertEqual(snapshot["payloads"], 1)
        self.assertEqual(snapshot["payload_bytes"], 4096)
        self.assertEqual(snapshot["latency_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
