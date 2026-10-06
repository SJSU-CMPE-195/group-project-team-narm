#!/usr/bin/env python3
"""Inventory frame counts for ASL training videos under raw_videos/."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv"}
CSV_COLUMNS = [
    "label",
    "relative_path",
    "filename",
    "frames",
    "fps",
    "duration_sec",
    "error",
]


def find_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return videos


def label_for(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return ""


def count_frames_by_reading(cap: cv2.VideoCapture) -> int:
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    return count


def inspect_video(path: Path) -> tuple[int | None, float | None, float | None, str]:
    """Return (frames, fps, duration_sec, error)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None, "failed to open"

    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        if frames <= 0:
            frames = count_frames_by_reading(cap)

        if frames <= 0:
            return None, fps if fps > 0 else None, None, "no frames readable"

        duration = (frames / fps) if fps > 0 else None
        return frames, fps if fps > 0 else None, duration, ""
    except Exception as exc:  # noqa: BLE001 — report per-file and continue
        return None, None, None, str(exc)
    finally:
        cap.release()


def write_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    ok_rows = [r for r in rows if not r["error"]]
    err_rows = [r for r in rows if r["error"]]

    by_label: dict[str, int] = defaultdict(int)
    for r in rows:
        by_label[r["label"]] += 1

    print(f"Total videos: {len(rows)}")
    print(f"Succeeded:    {len(ok_rows)}")
    print(f"Failed:       {len(err_rows)}")

    if ok_rows:
        frame_counts = [int(r["frames"]) for r in ok_rows]
        mean_frames = sum(frame_counts) / len(frame_counts)
        print(
            f"Frames:       min={min(frame_counts)}  "
            f"max={max(frame_counts)}  mean={mean_frames:.1f}"
        )

    print("\nPer-label video counts:")
    for label in sorted(by_label):
        print(f"  {label}: {by_label[label]}")

    if err_rows:
        print("\nFailures:")
        for r in err_rows:
            print(f"  {r['relative_path']}: {r['error']}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Count frames for ASL training videos and write a CSV table."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir / "raw_videos",
        help="Root directory containing gloss-label folders of videos",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=script_dir / "video_frame_counts.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: root directory not found: {root}", file=sys.stderr)
        return 1

    videos = find_videos(root)
    if not videos:
        print(f"No videos found under {root}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for i, path in enumerate(videos, start=1):
        rel = path.relative_to(root).as_posix()
        print(f"[{i}/{len(videos)}] {rel}", flush=True)

        frames, fps, duration, error = inspect_video(path)
        rows.append(
            {
                "label": label_for(path, root),
                "relative_path": rel,
                "filename": path.name,
                "frames": frames if frames is not None else "",
                "fps": f"{fps:.6g}" if fps is not None else "",
                "duration_sec": f"{duration:.6g}" if duration is not None else "",
                "error": error,
            }
        )

    write_csv(args.out.resolve(), rows)
    print()
    print_summary(rows)
    print(f"\nWrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
