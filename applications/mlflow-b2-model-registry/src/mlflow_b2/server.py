"""Webhook receiver: verify, persist to a spool, ack; a single worker registers models with retries."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import events

MAX_BODY = 1 << 20
MAX_RETRY_DELAY = 300.0
OK, RETRY, REJECTED = 0, 1, 2  # handler results, same as registry.handle_payload exit codes


class Spool:
    """Accepted deliveries live in pending/ until processed; rejected or exhausted ones move to failed/.

    Files are named by the body's sha256, so an at-least-once redelivery of the same body is stored once.
    """

    def __init__(self, root: Path):
        self.pending, self.failed = Path(root) / "pending", Path(root) / "failed"
        self.pending.mkdir(parents=True, exist_ok=True)
        self.failed.mkdir(parents=True, exist_ok=True)

    def put(self, body: bytes) -> Path:
        path = self.pending / f"{hashlib.sha256(body).hexdigest()}.json"
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        with open(tmp, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    def waiting(self) -> list[Path]:
        found = []
        for path in self.pending.glob("*.json"):
            try:
                found.append((path.stat().st_mtime, path))
            except FileNotFoundError:  # processed by the worker while listing
                pass
        return [path for _, path in sorted(found)]

    def fail(self, path: Path) -> None:
        os.replace(path, self.failed / path.name)


def make_server(secret: str, handle: Callable[[dict], int], host: str = "127.0.0.1", port: int = 8080,
                *, spool_dir: str | Path = "event-spool", max_attempts: int = 5,
                retry_delay: float = 2.0) -> ThreadingHTTPServer:
    spool = Spool(Path(spool_dir))
    jobs: queue.Queue[Path] = queue.Queue()
    queued: set[Path] = set()
    lock = threading.Lock()

    def enqueue(path: Path) -> None:
        with lock:
            if path in queued:
                return
            queued.add(path)
        jobs.put(path)

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
            try:
                path = spool.put(body)
            except OSError:
                return self._reply(503)  # not persisted, so let B2 redeliver
            enqueue(path)
            self._reply(200)

        def do_GET(self):
            if self.path != "/healthz":
                return self._reply(404)
            self._reply(200 if server.healthy else 503)

        def _reply(self, code: int):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            if code == 413:
                self.send_header("Connection", "close")
            self.end_headers()

        def log_message(self, fmt, *args):
            print(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}", file=sys.stderr)

    attempts: dict[Path, int] = {}

    def process(path: Path) -> None:
        try:
            result = handle(json.loads(path.read_bytes()))
        except Exception as e:  # temporary B2 or MLflow failures are retried
            print(f"error: event processing failed: {type(e).__name__}", file=sys.stderr)
            result = RETRY
        if result == OK:
            path.unlink(missing_ok=True)
        elif result == RETRY and attempts.get(path, 0) + 1 < max_attempts:
            attempts[path] = attempts.get(path, 0) + 1
            later(path, attempts[path] - 1)
            return
        else:
            spool.fail(path)
            print(f"event {path.stem[:12]} moved to {spool.failed}", file=sys.stderr)
        attempts.pop(path, None)

    def later(path: Path, attempt: int) -> None:
        timer = threading.Timer(min(retry_delay * 2 ** attempt, MAX_RETRY_DELAY), enqueue, [path])
        timer.daemon = True
        timer.start()

    def work():
        spool_failures = 0
        while True:
            path = jobs.get()
            with lock:
                queued.discard(path)
            try:
                if path.exists():
                    process(path)
                spool_failures = 0
            except Exception as e:  # spool disk trouble: keep the worker and the pending file, try again
                spool_failures += 1
                server.healthy = False
                print(f"error: event spool operation failed: {type(e).__name__}", file=sys.stderr)
                if path.exists():
                    later(path, spool_failures)
            else:
                server.healthy = True
            finally:
                jobs.task_done()

    server = ThreadingHTTPServer((host, port), Handler)
    server.jobs, server.spool, server.healthy = jobs, spool, True
    for path in spool.waiting():  # deliveries accepted before a restart
        enqueue(path)
    threading.Thread(target=work, name="b2-event-worker", daemon=True).start()
    return server
