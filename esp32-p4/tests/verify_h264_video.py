"""Generate real baseline H.264, run the C muxer, and decode its MP4 output.

Test-only dependency: PyAV. No camera, Jetson, or inference packages are needed.
Run from esp32-p4 with build/test_h264_mp4.exe already compiled:
    python tests/verify_h264_video.py
"""
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "h264-test-deps"))
import av


def main():
    output_dir = ROOT / "build" / "h264-browser-test"
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height, fps, frame_count = 1920, 1080, 30, 90
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, fps)
    encoder.framerate = Fraction(fps, 1)
    encoder.bit_rate = 4_000_000
    encoder.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "profile": "baseline",
        "x264-params": "aud=1:repeat-headers=1:keyint=30:min-keyint=30:scenecut=0:bframes=0",
    }
    annexb = output_dir / "fixture.h264"
    with annexb.open("wb") as destination:
        for index in range(frame_count):
            frame = av.VideoFrame(width, height, "yuv420p")
            # Vary luminance so a decoder must actually produce new pictures.
            for plane_index, plane in enumerate(frame.planes):
                value = 32 + index % 180 if plane_index == 0 else 128
                plane.update(bytes([value]) * plane.buffer_size)
            frame.pts = index
            for packet in encoder.encode(frame):
                destination.write(bytes(packet))
        for packet in encoder.encode(None):
            destination.write(bytes(packet))

    mp4 = output_dir / "fixture.mp4"
    subprocess.run(
        [str(ROOT / "build" / "test_h264_mp4.exe"), str(annexb), str(mp4),
         str(width), str(height), str(fps)],
        check=True,
    )
    with av.open(str(mp4)) as container:
        video = container.streams.video[0]
        assert video.codec_context.name == "h264"
        assert (video.codec_context.width, video.codec_context.height) == (width, height)
        decoded = list(container.decode(video=0))
        assert len(decoded) == frame_count, (len(decoded), frame_count)
        assert all((frame.width, frame.height) == (width, height) for frame in decoded)
        assert all(frame.pts < following.pts for frame, following in zip(decoded, decoded[1:]))
        assert abs(float(decoded[-1].time) - (frame_count - 1) / fps) < 1 / fps
        assert sum(frame.key_frame for frame in decoded) >= 3
        codec = f"avc1.{video.codec_context.extradata[1:4].hex()}"
    (output_dir / "codec.txt").write_text(codec, encoding="utf-8")
    print(f"Decoded all {frame_count} frames: {width}x{height} @ {fps} FPS, {codec}.")
    print(f"Browser test fixture: {mp4}")


if __name__ == "__main__":
    main()
