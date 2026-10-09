"""Loop the real C-muxed video fixture for browser playback QA on localhost.

Run tests/verify_h264_video.py first. This server is not firmware or a relay
required for production; the ESP serves these same assets itself.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "build" / "h264-browser-test"


def load_fragments():
    data = (FIXTURE / "fixture.mp4").read_bytes()
    fragments = []
    init = bytearray()
    offset = 0
    pending = None
    while offset < len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        assert size >= 8 and offset + size <= len(data)
        box = data[offset:offset + size]
        kind = box[4:8]
        if kind in (b"ftyp", b"moov"):
            init.extend(box)
        elif kind == b"moof":
            pending = box
        elif kind == b"mdat":
            assert pending is not None
            fragments.append(pending + box)
            pending = None
        offset += size
    assert init and fragments and pending is None
    return bytes(init), fragments


def main():
    initialization, fragments = load_fragments()
    codec = (FIXTURE / "codec.txt").read_text(encoding="utf-8").strip()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path in ("/", "/player.js"):
                filename = "h264_player.html" if self.path == "/" else "h264_player.js"
                body = (ROOT / "main" / filename).read_bytes()
                mime = "text/html" if self.path == "/" else "application/javascript"
                self.send_response(200)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream.mp4":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", f'video/mp4; codecs="{codec}"')
            self.send_header("X-Video-Width", "1920")
            self.send_header("X-Video-Height", "1080")
            self.send_header("X-Video-FPS", "30")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def send_chunk(body):
                self.wfile.write(f"{len(body):x}\r\n".encode("ascii"))
                self.wfile.write(body)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            try:
                send_chunk(initialization)
                index = 0
                deadline = time.monotonic()
                while True:
                    fragment = bytearray(fragments[index % len(fragments)])
                    tfdt = fragment.index(b"tfdt")
                    mfhd = fragment.index(b"mfhd")
                    struct.pack_into(">Q", fragment, tfdt + 8, index * 3000)
                    struct.pack_into(">I", fragment, mfhd + 8, index + 1)
                    send_chunk(fragment)
                    index += 1
                    deadline += 1 / 30
                    time.sleep(max(0, deadline - time.monotonic()))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                print("Browser disconnected; fixture stream released.", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("H.264 browser fixture: http://127.0.0.1:8765/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
