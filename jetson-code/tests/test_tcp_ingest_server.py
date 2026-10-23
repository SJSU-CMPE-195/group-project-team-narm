from pathlib import Path
import socket
import struct
import sys
import time
import unittest
from unittest.mock import MagicMock, patch


JETSON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JETSON_DIR))

from tcp_ingest_server import H264Frame, PayloadFrame, TCPIngestServer


class TCPIngestServerTests(unittest.TestCase):
    def test_button_stop_and_restart_creates_new_stream(self):
        """A closed ESP connection must not prevent the next button press."""
        first_sender, first_receiver = socket.socketpair()
        second_sender, second_receiver = socket.socketpair()
        for connection in (first_sender, first_receiver, second_sender, second_receiver):
            self.addCleanup(connection.close)

        first_payload = b"\x00\x00\x00\x01first H.264 access unit"
        second_payload = b"\x00\x00\x00\x01new stream H.264 access unit"
        # Include a zero-length keepalive before the first access unit.
        first_sender.sendall(struct.pack(">II", 0, len(first_payload)) + first_payload)
        first_sender.shutdown(socket.SHUT_WR)
        second_sender.sendall(struct.pack(">I", len(second_payload)) + second_payload)

        frames = []

        def receive(frame):
            frames.append(frame)
            if len(frames) == 2:
                server.stop()

        server = TCPIngestServer(on_frame=receive)
        listener = MagicMock()
        listener.__enter__.return_value = listener
        listener.accept.side_effect = [
            (first_receiver, ("esp32", 10001)),
            (second_receiver, ("esp32", 10002)),
        ]

        started = time.monotonic()
        with patch("tcp_ingest_server.socket.socket", return_value=listener):
            server.serve_forever()

        self.assertEqual([frame.payload for frame in frames], [first_payload, second_payload])
        self.assertEqual([frame.stream_id for frame in frames], [1, 2])
        self.assertTrue(all(isinstance(frame, PayloadFrame) for frame in frames))
        self.assertTrue(all(started <= frame.received_ts <= time.monotonic() for frame in frames))
        self.assertIs(H264Frame, PayloadFrame)

    def test_incomplete_frame_is_discarded_before_reconnection(self):
        first_sender, first_receiver = socket.socketpair()
        second_sender, second_receiver = socket.socketpair()
        for connection in (first_sender, first_receiver, second_sender, second_receiver):
            self.addCleanup(connection.close)

        first_sender.sendall(struct.pack(">I", 10) + b"partial")
        first_sender.shutdown(socket.SHUT_WR)
        second_sender.sendall(struct.pack(">I", 4) + b"next")

        frames = []

        def receive(frame):
            frames.append(frame)
            server.stop()

        server = TCPIngestServer(on_frame=receive)
        listener = MagicMock()
        listener.__enter__.return_value = listener
        listener.accept.side_effect = [
            (first_receiver, ("esp32", 10001)),
            (second_receiver, ("esp32", 10002)),
        ]

        with patch("tcp_ingest_server.socket.socket", return_value=listener):
            server.serve_forever()

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].payload, b"next")
        self.assertEqual(frames[0].stream_id, 2)


if __name__ == "__main__":
    unittest.main()
