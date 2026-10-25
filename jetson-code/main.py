from __future__ import annotations

from dataclasses import dataclass, field
import os
import queue
import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from holistic_lstm_infer import HolisticLSTMInfer, validate_model
from tcp_ingest_server import PayloadFrame, TCPIngestServer


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


class FrameSampler:
    """Keep every Nth decoded frame.

    The model is trained on every 2nd frame of 30 fps video (15 fps). The glasses stream defaults
    to 15 fps, so FRAME_STRIDE=1 keeps every frame to match that timing. Use FRAME_STRIDE=2 for
    30 fps firmware. The stride is applied to arriving frames (before the queue that drops frames
    when inference is slow), so it does not depend on how fast inference runs.
    """

    def __init__(self, stride: int) -> None:
        self.stride = max(int(stride), 1)
        self._count = 0

    def keep(self) -> bool:
        keep = self._count % self.stride == 0
        self._count += 1
        return keep


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
    latest_frame_rgb: Optional[np.ndarray] = None
    latest_sentence: List[str] = field(default_factory=list)
    latest_probs: Optional[np.ndarray] = None
    latest_landmarks: Optional[dict] = None
    # Bumped on every change so the preview can skip re-encoding unchanged data.
    version: int = 0

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self.latest_frame_bgr = frame_bgr.copy()
            self.version += 1

    def update_frame_rgb(self, frame_rgb: np.ndarray) -> None:
        """Store the newest decoded frame without copying or converting it.

        Called from the decode thread for every frame, so the preview shows
        all decoded video and does not wait for MediaPipe/LSTM inference.
        """
        with self._lock:
            self.latest_frame_rgb = frame_rgb
            self.version += 1

    def set_inference(self, probs: Optional[np.ndarray], landmarks: Optional[dict]) -> None:
        with self._lock:
            self.latest_probs = None if probs is None else np.array(probs, copy=True)
            self.latest_landmarks = landmarks
            self.version += 1

    def push_gloss(self, gloss: str) -> None:
        with self._lock:
            if self.latest_sentence and self.latest_sentence[-1] == gloss:
                return
            self.latest_sentence.append(gloss)
            if len(self.latest_sentence) > self.max_sentence:
                self.latest_sentence = self.latest_sentence[-self.max_sentence :]
            self.version += 1

    def preview_snapshot(self) -> tuple:
        """(version, frame_rgb, sentence, probs, actions, landmarks) for the preview.

        The frame array is shared, not copied; callers must only read it.
        """
        with self._lock:
            return (
                self.version,
                self.latest_frame_rgb,
                list(self.latest_sentence),
                None if self.latest_probs is None else np.array(self.latest_probs, copy=True),
                list(self.actions),
                self.latest_landmarks,
            )

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
    extra: Optional[Callable[[], str]] = None,
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
            if extra is not None:
                timing = extra()
                if timing:
                    print(f"[timing] per inferred frame: {timing}")
            previous = current
            previous_ts = now

    threading.Thread(target=report, name="pipeline-stats", daemon=True).start()


def main() -> None:
    host = env_str("HOST", "0.0.0.0")
    port = env_int("PORT", 5000)
    preview_http = env_bool("PREVIEW_HTTP", False)
    preview_port = env_int("PREVIEW_PORT", 8000)
    preview_max_sentence = env_int("PREVIEW_MAX_SENTENCE", 5)
    preview_fps = env_int("PREVIEW_FPS", 30)
    preview_width = env_int("PREVIEW_WIDTH", 640)
    preview_quality = env_int("PREVIEW_JPEG_QUALITY", 75)
    preview_landmarks = env_bool("PREVIEW_LANDMARKS", True)
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
    # Stop early with a clear message if the model was trained on a different feature set or word list.
    validate_model(model, sequence_length, actions)
    frame_stride = env_int("FRAME_STRIDE", 1)
    min_hand_frac = env_float("MIN_HAND_FRAC", 0.6)
    sampler = FrameSampler(frame_stride)
    infer = HolisticLSTMInfer(
        model=model,
        actions=actions,
        sequence_length=sequence_length,
        threshold=threshold,
        stable_n=stable_n,
        model_complexity=env_int("MODEL_COMPLEXITY", 0),
        infer_width=env_int("INFER_WIDTH", 640),
        repeat_reset=env_int("REPEAT_RESET", 15),
        min_hand_frac=min_hand_frac,
    )
    print(
        f"[main] model={model_path} actions={','.join(actions)} window={sequence_length} "
        f"frame_stride={frame_stride} threshold={threshold} stable_n={stable_n} "
        f"min_hand_frac={min_hand_frac}"
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

    if preview_http:
        try:
            from preview_http import PreviewHTTPServer

            preview_server = PreviewHTTPServer(
                latest_state=latest,
                host="0.0.0.0",
                port=preview_port,
                max_fps=float(preview_fps),
                width=preview_width,
                quality=preview_quality,
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
                if preview_http:
                    # Show every decoded frame; do not wait for inference.
                    latest.update_frame_rgb(decoded.rgb)
                if not sampler.keep():
                    continue  # skipped by FRAME_STRIDE; the model only sees every Nth frame
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
    start_stats_reporter(
        stats,
        payload_q,
        decoded_q,
        max(stats_interval, 1),
        extra=infer.timer.summary_and_reset,
    )

    print(
        f"[main] listening {host}:{port}, payload={payload_format}, "
        f"payload_queue={payload_q.maxsize}, decoded_queue={decoded_q.maxsize}"
    )

    try:
        while True:
            work = decoded_q.get()

            try:
                prediction = infer.push_frame(work.rgb)
            except Exception as exc:
                print(f"[infer] frame failed: {exc!r}")
                continue

            if preview_http:
                try:
                    landmarks = infer.preview_landmarks() if preview_landmarks else None
                    latest.set_inference(infer.last_probs, landmarks)
                except Exception:
                    pass

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
