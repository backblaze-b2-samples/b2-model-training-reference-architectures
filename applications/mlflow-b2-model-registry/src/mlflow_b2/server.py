"""Webhook receiver: verify, enqueue, ack fast; a single worker registers models in order."""
from __future__ import annotations

import json
import queue
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import events

MAX_BODY = 1 << 20


def make_server(secret: str, handle: Callable[[dict], object], host: str = "127.0.0.1",
                port: int = 8080) -> ThreadingHTTPServer:
    jobs: queue.Queue[bytes] = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        timeout = 10

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._reply(411)
            if not 0 <= length <= MAX_BODY:
                return self._reply(413)
            body = self.rfile.read(length)
            if not events.verify_signature(secret, body, self.headers.get(events.SIGNATURE_HEADER)):
                return self._reply(401)
            try:
                if not isinstance(json.loads(body), dict):
                    return self._reply(400)
            except ValueError:
                return self._reply(400)
            jobs.put(body)
            self._reply(200)

        def _reply(self, code: int):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            if code == 413:
                self.send_header("Connection", "close")
            self.end_headers()

        def log_message(self, fmt, *args):
            print(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}", file=sys.stderr)

    def work():
        while True:
            body = jobs.get()
            try:
                handle(json.loads(body))
            except Exception as e:  # keep the worker alive; delivery is at least once
                print(f"error: event processing failed: {type(e).__name__}", file=sys.stderr)
            finally:
                jobs.task_done()

    server = ThreadingHTTPServer((host, port), Handler)
    server.jobs = jobs
    threading.Thread(target=work, name="b2-event-worker", daemon=True).start()
    return server
