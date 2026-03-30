from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional

from decode_h264 import H264PayloadDecoder
from gloss_to_sentence import glosses_to_sentence
from i3d_sliding_window import I3DSlidingWindow
from tcp_ingest_server import TCPIngestServer
from tts import speak


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

    wlasl_root = os.environ.get("WLASL_ROOT", "")
    checkpoint_path = os.environ.get("I3D_CHECKPOINT_PATH", "")
    if not wlasl_root:
        raise RuntimeError("Set WLASL_ROOT to your WLASL checkout (contains code/).")
    if not checkpoint_path:
        raise RuntimeError("Set I3D_CHECKPOINT_PATH to your trained I3D checkpoint.")

    window_t = env_int("I3D_WINDOW_T", 64)
    _dbg(
        "startup_config",
        {"host": host, "port": port, "window_t": window_t, "wlasl_root": wlasl_root, "ckpt": checkpoint_path},
        "H1",
    )
    model = I3DSlidingWindow(wlasl_root=wlasl_root, checkpoint_path=checkpoint_path, window_t=window_t)
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

    glosses = []
    last_gloss: Optional[str] = None

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
                pred = model.push_frame(df.rgb)
            except Exception as e:
                _dbg("i3d_exception", {"err": repr(e)}, "H4")
                raise
            if pred is None:
                continue

            if last_gloss != pred.gloss:
                glosses.append(pred.gloss)
                last_gloss = pred.gloss
                print(f"[gloss] {pred.gloss} (conf={pred.confidence:.3f})")
                _dbg("gloss_update", {"gloss": pred.gloss, "conf": pred.confidence, "count": len(glosses)}, "H4")

            # Demo policy: every 5 glosses, produce a sentence.
            if len(glosses) >= 5:
                _dbg("ollama_request", {"glosses": glosses, "model": os.environ.get("OLLAMA_MODEL")}, "H5")
                sent = glosses_to_sentence(glosses)
                print(f"[sentence] {sent}")
                speak(sent)
                _dbg("sentence_ready", {"sentence": sent}, "H5")
                glosses.clear()
                last_gloss = None

        time.sleep(0.001)


if __name__ == "__main__":
    main()

