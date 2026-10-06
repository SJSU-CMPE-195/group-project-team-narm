from __future__ import annotations

from dataclasses import dataclass, field
import os
import queue
import threading
import time
from typing import List, Optional, Sequence

import numpy as np

from holistic_lstm_infer import HolisticLSTMInfer
from tcp_ingest_server import PayloadFrame, TCPIngestServer


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default

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
            self.latest_frame_bgr = frame_bgr.copy()

    def push_gloss(self, gloss: str) -> None:
        with self._lock:
            if self.latest_sentence and self.latest_sentence[-1] == gloss:
                return
            self.latest_sentence.append(gloss)
            if len(self.latest_sentence) > self.max_sentence:
                self.latest_sentence = self.latest_sentence[-self.max_sentence :]

    def snapshot(
        self,
    ) -> tuple[Optional[np.ndarray], List[str], Optional[np.ndarray], Sequence[str]]:
        with self._lock:
            frame = None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy()
            sentence = list(self.latest_sentence)
            probs = None if self.latest_probs is None else np.array(self.latest_probs, copy=True)
            actions = list(self.actions)
        return frame, sentence, probs, actions


@dataclass(frozen=True)
class DecodedWorkFrame:
    rgb: np.ndarray
    received_ts: float


@dataclass
class PipelineStats:
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    payloads: int = 0
    payload_bytes: int = 0
    payload_drops: int = 0
    decoded: int = 0
    decoded_drops: int = 0
    decode_errors: int = 0
    inferred: int = 0
    glosses: int = 0
    latency_ms: float = 0.0

    def add(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, int(getattr(self, field_name)) + amount)

    def set_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latency_ms = latency_ms

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                "payloads": self.payloads,
                "payload_bytes": self.payload_bytes,
                "payload_drops": self.payload_drops,
                "decoded": self.decoded,
                "decoded_drops": self.decoded_drops,
                "decode_errors": self.decode_errors,
                "inferred": self.inferred,
                "glosses": self.glosses,
                "latency_ms": self.latency_ms,
            }


def put_latest(target: queue.Queue, item: object) -> bool:
    """Put an independent frame, evicting the oldest item when full."""
    dropped = False
    while True:
        try:
            target.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                target.get_nowait()
                dropped = True
            except queue.Empty:
                continue


def start_stats_reporter(
    stats: PipelineStats,
    payload_q: queue.Queue,
    decoded_q: queue.Queue,
    interval: int,
) -> None:
    def report() -> None:
        previous = stats.snapshot()
        previous_ts = time.monotonic()
        while True:
            time.sleep(interval)
            current = stats.snapshot()
            now = time.monotonic()
            elapsed = max(now - previous_ts, 0.001)
            rx_fps = (current["payloads"] - previous["payloads"]) / elapsed
            decode_fps = (current["decoded"] - previous["decoded"]) / elapsed
            infer_fps = (current["inferred"] - previous["inferred"]) / elapsed
            mbps = (
                (current["payload_bytes"] - previous["payload_bytes"])
                * 8.0
                / elapsed
                / 1_000_000.0
            )
            print(
                "[stats] "
                f"rx={rx_fps:.1f}fps/{mbps:.2f}Mbps "
                f"decode={decode_fps:.1f}fps infer={infer_fps:.1f}fps "
                f"drops(payload={int(current['payload_drops'])},"
                f"decoded={int(current['decoded_drops'])}) "
                f"errors={int(current['decode_errors'])} "
                f"queues(payload={payload_q.qsize()},decoded={decoded_q.qsize()}) "
                f"latency={current['latency_ms']:.1f}ms"
            )
            previous = current
            previous_ts = now

    threading.Thread(target=report, name="pipeline-stats", daemon=True).start()


def main() -> None:
    host = env_str("HOST", "0.0.0.0")
    port = env_int("PORT", 5000)
    preview_http = env_bool("PREVIEW_HTTP", False)
    preview_port = env_int("PREVIEW_PORT", 8000)
    preview_max_sentence = env_int("PREVIEW_MAX_SENTENCE", 5)
    payload_format = env_str("PAYLOAD_FORMAT", "h264").strip().lower()
    stats_interval = env_int("STATS_INTERVAL_SEC", 5)

    model_path = env_str("MODEL_PATH", "")
    if not model_path:
        raise RuntimeError("Set MODEL_PATH to a trained LSTM model file.")

    sequence_length = env_int("SEQUENCE_LENGTH", 30)
    stable_n = env_int("STABLE_N", 10)
    threshold = float(os.environ.get("THRESHOLD", "0.5"))
    actions_env = env_str("ACTIONS", "")
    actions: List[str] = (
        [action.strip() for action in actions_env.split(",") if action.strip()]
        if actions_env
        else ["hello", "thanks", "iloveyou"]
    )

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    infer = HolisticLSTMInfer(
        model=model,
        actions=actions,
        sequence_length=sequence_length,
        threshold=threshold,
        stable_n=stable_n,
    )
    if payload_format == "jpeg":
        from decode_jpeg import JpegPayloadDecoder

        decoder = JpegPayloadDecoder()
    elif payload_format == "h264":
        from decode_h264 import H264PayloadDecoder

        decoder = H264PayloadDecoder()
    else:
        raise RuntimeError("PAYLOAD_FORMAT must be 'h264' or 'jpeg'")

    # H.264 access units must enter the decoder in order. This queue applies
    # TCP backpressure instead of discarding dependent compressed frames.
    payload_q: "queue.Queue[PayloadFrame]" = queue.Queue(
        maxsize=env_int("PAYLOAD_QUEUE_SIZE", 64)
    )
    # Decoded RGB frames are independent, so keeping only the freshest few is
    # safe when MediaPipe/inference is slower than the camera.
    decoded_q: "queue.Queue[DecodedWorkFrame]" = queue.Queue(
        maxsize=env_int("DECODED_QUEUE_SIZE", 2)
    )
    stats = PipelineStats()
    latest = LatestState(actions=actions, max_sentence=preview_max_sentence)
    cv2_module = None

    if preview_http:
        try:
            import cv2 as cv2_module
            from preview_http import PreviewHTTPServer

            preview_server = PreviewHTTPServer(
                latest_state=latest, host="0.0.0.0", port=preview_port
            )
            threading.Thread(
                target=preview_server.serve_forever,
                name="preview-http",
                daemon=True,
            ).start()
            print(f"[preview] http listening on 0.0.0.0:{preview_port}")
        except Exception as exc:
            print(f"[preview] disabled (failed to start): {exc!r}")
            preview_http = False

    def on_payload(frame: PayloadFrame) -> None:
        stats.add("payloads")
        stats.add("payload_bytes", len(frame.payload))
        if payload_format == "h264":
            payload_q.put(frame)
        elif put_latest(payload_q, frame):
            stats.add("payload_drops")

    def decode_worker() -> None:
        current_stream_id = -1
        last_error_log_ts = 0.0
        while True:
            payload = payload_q.get()
            if payload_format == "h264" and payload.stream_id != current_stream_id:
                decoder.reset()
                current_stream_id = payload.stream_id
            try:
                decoded_frames = decoder.decode_payload(payload.payload)
            except Exception as exc:
                stats.add("decode_errors")
                now = time.monotonic()
                if now - last_error_log_ts >= 5.0:
                    print(f"[decode] invalid payload (rate-limited): {exc!r}")
                    last_error_log_ts = now
                continue

            for decoded in decoded_frames:
                stats.add("decoded")
                work = DecodedWorkFrame(
                    rgb=decoded.rgb,
                    received_ts=payload.received_ts,
                )
                if put_latest(decoded_q, work):
                    stats.add("decoded_drops")

    threading.Thread(
        target=decode_worker,
        name="video-decode",
        daemon=True,
    ).start()

    server = TCPIngestServer(host=host, port=port, on_frame=on_payload)
    threading.Thread(
        target=server.serve_forever,
        name="tcp-ingest",
        daemon=True,
    ).start()
    start_stats_reporter(stats, payload_q, decoded_q, max(stats_interval, 1))

    print(
        f"[main] listening {host}:{port}, payload={payload_format}, "
        f"payload_queue={payload_q.maxsize}, decoded_queue={decoded_q.maxsize}"
    )

    try:
        while True:
            work = decoded_q.get()
            if preview_http:
                try:
                    latest.update_frame(
                        cv2_module.cvtColor(work.rgb, cv2_module.COLOR_RGB2BGR)
                    )
                except Exception:
                    pass

            try:
                prediction = infer.push_frame(work.rgb)
            except Exception as exc:
                print(f"[infer] frame failed: {exc!r}")
                continue

            stats.add("inferred")
            stats.set_latency((time.monotonic() - work.received_ts) * 1000.0)
            if prediction is not None:
                stats.add("glosses")
                print(
                    f"[gloss] {prediction.gloss} "
                    f"(conf={prediction.confidence:.3f})"
                )
                latest.push_gloss(prediction.gloss)
    finally:
        infer.close()
        server.stop()


if __name__ == "__main__":
    main()
