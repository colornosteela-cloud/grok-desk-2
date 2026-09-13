"""SSE slow subscribers are dropped with the socket closed so clients reconnect."""
import socket
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deskd"))

import deskd as d  # noqa: E402


class _FakeSseHandler:
    def __init__(self, sock: socket.socket) -> None:
        self.connection = sock


class SlowSubscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = list(d.subscribers)

    def tearDown(self) -> None:
        with d.lock:
            d.subscribers[:] = self.saved

    def _register(self) -> tuple[threading.Event, list, socket.socket]:
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        handler = _FakeSseHandler(b)
        wake = threading.Event()
        q: list = []
        with d.lock:
            d.subscribers.append((wake, q, handler))
        return wake, q, b

    def test_healthy_subscriber_keeps_receiving(self) -> None:
        wake, q, _sock = self._register()
        for i in range(10):
            d.emit({"type": "chat", "n": i})
        self.assertEqual(len(q), 10)
        self.assertTrue(wake.is_set())
        with d.lock:
            self.assertEqual(len(d.subscribers), 1)

    def test_slow_subscriber_is_dropped_and_closed(self) -> None:
        _wake, _q, sock = self._register()
        for i in range(501):
            d.emit({"type": "chat", "n": i})
        with d.lock:
            self.assertEqual(d.subscribers, [])
        self.assertEqual(sock.fileno(), -1, "socket must be closed so EventSource reconnects")

    def test_exactly_500_events_keeps_subscriber(self) -> None:
        _wake, q, _sock = self._register()
        for i in range(500):
            d.emit({"type": "chat", "n": i})
        self.assertEqual(len(q), 500)
        with d.lock:
            self.assertEqual(len(d.subscribers), 1)


if __name__ == "__main__":
    unittest.main()
