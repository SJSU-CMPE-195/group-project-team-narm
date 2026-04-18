from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional

from decode_h264 import H264PayloadDecoder
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


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = env_int("PORT", 5000)

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
    decoder = H264PayloadDecoder()

    q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

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
        _dbg("payload_received", {"bytes": len(payload)}, "H2")
        try:
            decoded_frames = decoder.decode_payload(payload)
        except Exception as e:
            _dbg("decode_exception", {"err": repr(e), "payload_bytes": len(payload)}, "H3")
            continue
        _dbg("payload_decoded", {"frames": len(decoded_frames)}, "H3")
        for df in decoded_frames:
            try:
                pred = infer.push_frame(df.rgb)
            except Exception as e:
                _dbg("infer_exception", {"err": repr(e)}, "H4")
                raise
            if pred is None:
                continue
            print(f"[gloss] {pred.gloss} (conf={pred.confidence:.3f})")
            _dbg("gloss_update", {"gloss": pred.gloss, "conf": pred.confidence}, "H4")

        time.sleep(0.001)


if __name__ == "__main__":
    main()

