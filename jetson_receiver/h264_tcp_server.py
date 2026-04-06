import argparse
import os
import socket
import struct
import time


def recvall(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise ConnectionError if the peer disconnects."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("TCP client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive H.264 AUs over TCP with 4-byte BE length prefix.")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5000, help="Listen port (default: 5000).")
    parser.add_argument("--outfile", default="received.h264", help="Output .h264 file (default: received.h264).")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    parser.add_argument("--log-every", type=int, default=30, help="Log every N frames (default: 30).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)) or ".", exist_ok=True)
    out_path = os.path.abspath(args.outfile)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"[h264_tcp_server] Listening on {args.host}:{args.port}")
    print(f"[h264_tcp_server] Writing to: {out_path}")

    conn, peer = server.accept()
    print(f"[h264_tcp_server] Client connected: {peer[0]}:{peer[1]}")

    # Try to reduce latency for small packets.
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    frame_idx = 0
    t0 = time.time()
    total_payload_bytes = 0

    with open(out_path, "wb") as f:
        while True:
            header = conn.recv(4)
            if not header:
                break
            # recv(4) may return fewer than 4 bytes; pull the remainder.
            if len(header) < 4:
                header = header + recvall(conn, 4 - len(header))

            (length,) = struct.unpack(">I", header)  # big-endian uint32
            if length == 0:
                continue

            payload = recvall(conn, length)
            f.write(payload)
            f.flush()

            frame_idx += 1
            total_payload_bytes += length

            if args.log_every > 0 and (frame_idx % args.log_every == 0):
                elapsed = time.time() - t0
                fps = (frame_idx / elapsed) if elapsed > 0 else 0.0
                mbps = (total_payload_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0.0
                print(
                    f"[h264_tcp_server] frames={frame_idx} "
                    f"last_au_size={length} bytes fps~{fps:.2f} payload~{mbps:.2f} MB/s"
                )

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"[h264_tcp_server] Reached max_frames={args.max_frames}. Stopping.")
                break

    conn.close()
    server.close()
    print(f"[h264_tcp_server] Done. frames={frame_idx} output={out_path}")


if __name__ == "__main__":
    main()

