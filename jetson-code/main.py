from __future__ import annotations

from dataclasses import dataclass, field
import os
import queue
import threading
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

from decode_h264 import H264PayloadDecoder
from decode_jpeg import JpegPayloadDecoder
from holistic_lstm_infer import HolisticLSTMInfer
from tcp_ingest_server import TCPIngestServer


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _dbg(message: str, data: dict, hypothesis_id: str) -> None:
    # #region agent log
    import json

    try:
        with open("debug-9d06a1.log", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "9d06a1",
                        "runId": os.environ.get("DEBUG_RUN_ID", "run1"),
                        "hypothesisId": hypothesis_id,
                        "location": "jetson-code/main.py",
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion agent log


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


@dataclass
class LatestState:
    actions: Sequence[str]
    max_sentence: int = 5
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    latest_frame_bgr: Optional[np.ndarray] = None
    latest_sentence: List[str] = field(default_factory=list)
    latest_probs: Optional[np.ndarray] = None

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            # Store a copy so the server thread can safely read/overlay it.
            self.latest_frame_bgr = frame_bgr.copy()

    def push_gloss(self, gloss: str) -> None:
        with self._lock:
            if self.latest_sentence and self.latest_sentence[-1] == gloss:
                return
            self.latest_sentence.append(gloss)
            if len(self.latest_sentence) > self.max_sentence:
                self.latest_sentence = self.latest_sentence[-self.max_sentence :]

    def snapshot(self) -> tuple[Optional[np.ndarray], List[str], Optional[np.ndarray], Sequence[str]]:
        with self._lock:
            frame = None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy()
            sentence = list(self.latest_sentence)
            probs = None if self.latest_probs is None else np.array(self.latest_probs, copy=True)
            actions = list(self.actions)
        return frame, sentence, probs, actions


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = env_int("PORT", 5000)
    preview_http = env_bool("PREVIEW_HTTP", False)
    preview_port = env_int("PREVIEW_PORT", 8000)
    preview_max_sentence = env_int("PREVIEW_MAX_SENTENCE", 5)
    payload_format = env_str("PAYLOAD_FORMAT", "jpeg").strip().lower()

    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path:
        raise RuntimeError("Set MODEL_PATH to your trained LSTM model file (action.h5 or action.keras).")

    sequence_length = env_int("SEQUENCE_LENGTH", 30)
    stable_n = env_int("STABLE_N", 10)
    threshold = float(os.environ.get("THRESHOLD", "0.5"))
    actions_env = os.environ.get("ACTIONS", "")
    if actions_env:
        actions: List[str] = [a.strip() for a in actions_env.split(",") if a.strip()]
    else:
        # Default actions from the tutorial notebook; override via ACTIONS for your own model.
        actions = ["hello", "thanks", "iloveyou"]

    import tensorflow as tf

    _dbg(
        "startup_config",
        {
            "host": host,
            "port": port,
            "model_path": model_path,
            "sequence_length": sequence_length,
            "stable_n": stable_n,
            "threshold": threshold,
            "actions": actions,
        },
        "H1",
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    infer = HolisticLSTMInfer(
        model=model,
        actions=actions,
        sequence_length=sequence_length,
        threshold=threshold,
        stable_n=stable_n,
    )
    if payload_format == "jpeg":
        decoder = JpegPayloadDecoder()
    elif payload_format == "h264":
        decoder = H264PayloadDecoder()
    else:
        raise RuntimeError("PAYLOAD_FORMAT must be 'h264' or 'jpeg'")

    q: "queue.Queue[bytes]" = queue.Queue(maxsize=16)
    latest = LatestState(actions=actions, max_sentence=preview_max_sentence)

    if preview_http:
        try:
            from preview_http import PreviewHTTPServer

            preview_server = PreviewHTTPServer(latest_state=latest, host="0.0.0.0", port=preview_port)
            threading.Thread(target=preview_server.serve_forever, daemon=True).start()
            print(f"[preview] http listening on 0.0.0.0:{preview_port}")
        except Exception as e:
            print(f"[preview] disabled (failed to start): {e!r}")

    def on_frame(frame):
        try:
            q.put_nowait(frame.payload)
        except queue.Full:
            # Drop if we're behind to keep latency bounded.
            return

    server = TCPIngestServer(host=host, port=port, on_frame=on_frame)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    print(f"[main] listening {host}:{port}")
    _dbg("tcp_server_started", {"host": host, "port": port}, "H2")

    while True:
        payload = q.get()
        # Drain backlog: keep only the freshest frame so inference tracks live video under load.
        while True:
            try:
                payload = q.get_nowait()
            except queue.Empty:
                break
        _dbg("payload_received", {"bytes": len(payload)}, "H2")
        try:
            decoded_frames = decoder.decode_payload(payload)
        except Exception as e:
            _dbg("decode_exception", {"err": repr(e), "payload_bytes": len(payload)}, "H3")
            continue
        _dbg("payload_decoded", {"frames": len(decoded_frames)}, "H3")
        for df in decoded_frames:
            try:
                bgr = cv2.cvtColor(df.rgb, cv2.COLOR_RGB2BGR)
                latest.update_frame(bgr)
            except Exception:
                # Preview is best-effort; do not break inference if conversion fails.
                pass
            try:
                pred = infer.push_frame(df.rgb)
            except Exception as e:
                _dbg("infer_exception", {"err": repr(e)}, "H4")
                raise
            if pred is None:
                continue
            print(f"[gloss] {pred.gloss} (conf={pred.confidence:.3f})")
            _dbg("gloss_update", {"gloss": pred.gloss, "conf": pred.confidence}, "H4")
            latest.push_gloss(pred.gloss)

        time.sleep(0.001)


if __name__ == "__main__":
    main()

