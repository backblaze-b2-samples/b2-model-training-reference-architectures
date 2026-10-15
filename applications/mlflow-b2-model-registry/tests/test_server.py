import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest
from conftest import EVENT_SECRET, event_body, model_manifest, publish, sign

from mlflow_b2 import events, registry, server


@pytest.fixture
def serve(tmp_path):
    started = []

    def start(handle, **kw):
        kw = {"spool_dir": tmp_path / "spool", "retry_delay": 0.01, **kw}
        srv = server.make_server(EVENT_SECRET, handle, "127.0.0.1", 0, **kw)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return srv

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def _post(srv, body: bytes, signature: str | None) -> int:
    headers = {events.SIGNATURE_HEADER: signature} if signature else {}
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/", data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_acks_before_processing_and_rejects_bad_signature(serve):
    seen, release = [], threading.Event()

    def slow(payload):
        release.wait(5)
        seen.append(payload)
        return server.OK

    srv = serve(slow)
    body = event_body("models/m/v1/manifest.json")
    t = time.monotonic()
    assert _post(srv, body, sign(body)) == 200
    assert time.monotonic() - t < 1 and seen == []
    assert _post(srv, body, sign(body, "wrong-secret")) == 401
    assert _post(srv, body, None) == 401
    assert _post(srv, b"not json", sign(b"not json")) == 400
    assert _post(srv, b"[]", sign(b"[]")) == 400
    release.set()
    srv.jobs.join()
    assert len(seen) == 1


def test_oversized_body_rejected_before_reading(serve):
    srv = serve(lambda p: server.OK)
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    conn.putrequest("POST", "/")
    conn.putheader("Content-Length", str(server.MAX_BODY + 1))
    conn.endheaders()
    assert conn.getresponse().status == 413
    conn.close()


def test_worker_registers_model(serve, s3, cfg):
    client = registry.configure(cfg)
    srv = serve(lambda p: registry.handle_payload(cfg, client, s3, p))
    body = event_body(publish(s3, model_manifest()))
    assert _post(srv, body, sign(body)) == 200
    assert _post(srv, body, sign(body)) == 200  # at-least-once redelivery
    srv.jobs.join()
    assert [str(mv.version) for mv in client.search_model_versions("name = 'shapes-classifier'")] == ["1"]


def _until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_delivery_is_persisted_before_ack(serve):
    release = threading.Event()
    srv = serve(lambda p: server.OK if release.wait(5) else server.RETRY)
    body = event_body("models/m/v1/manifest.json")
    assert _post(srv, body, sign(body)) == 200
    assert _post(srv, body, sign(body)) == 200  # redelivery of the same body is stored once
    assert len(srv.spool.waiting()) == 1
    release.set()
    _until(lambda: not srv.spool.waiting())


def test_temporary_failure_is_retried(serve):
    results = [server.RETRY, server.RETRY, server.OK]
    calls = []

    def flaky(payload):
        calls.append(payload)
        result = results[len(calls) - 1]
        if len(calls) == 2:
            raise RuntimeError("B2 or MLflow briefly unavailable")
        return result

    srv = serve(flaky)
    body = event_body("models/m/v1/manifest.json")
    assert _post(srv, body, sign(body)) == 200
    _until(lambda: len(calls) == 3 and not srv.spool.waiting())
    assert list(srv.spool.failed.iterdir()) == []


def test_rejected_and_exhausted_events_are_kept_in_failed(serve):
    srv = serve(lambda p: server.REJECTED)
    body = event_body("models/m/v1/manifest.json")
    assert _post(srv, body, sign(body)) == 200
    _until(lambda: len(list(srv.spool.failed.iterdir())) == 1)
    calls = []
    srv2 = serve(lambda p: calls.append(p) or server.RETRY, spool_dir=srv.spool.pending.parent.parent / "s2",
                 max_attempts=3)
    body2 = event_body("models/m/v2/manifest.json")
    assert _post(srv2, body2, sign(body2)) == 200
    _until(lambda: len(list(srv2.spool.failed.iterdir())) == 1)
    assert len(calls) == 3 and not srv2.spool.waiting()


def test_pending_deliveries_replay_after_restart(serve, tmp_path):
    body = event_body("models/m/v1/manifest.json")
    server.Spool(tmp_path / "spool").put(body)  # accepted and acknowledged before a crash
    seen = []
    srv = serve(lambda p: seen.append(p) or server.OK)
    _until(lambda: not srv.spool.waiting())
    assert seen == [json.loads(body)]


def test_temporary_head_error_is_retried_through_the_receiver(serve, s3, cfg, monkeypatch):
    from test_events import _flaky_head

    client = registry.configure(cfg)
    srv = serve(lambda p: registry.handle_payload(cfg, client, s3, p))
    body = event_body(publish(s3, model_manifest()))
    calls = _flaky_head(s3, monkeypatch)
    assert _post(srv, body, sign(body)) == 200
    _until(lambda: not srv.spool.waiting())
    assert len(calls) >= 2 and list(srv.spool.failed.iterdir()) == []
    assert [str(mv.version) for mv in client.search_model_versions("name = 'shapes-classifier'")] == ["1"]


def _health(srv) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/healthz", timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_worker_survives_spool_filesystem_errors(serve):
    first, second = event_body("models/m/v1/manifest.json"), event_body("models/m/v2/manifest.json")
    seen = []

    def handle(payload):
        seen.append(payload)
        return server.REJECTED if payload == json.loads(first) else server.OK

    srv = serve(handle)
    srv.spool.failed.chmod(0o500)  # moving the rejected event into failed/ will raise PermissionError
    try:
        assert _post(srv, first, sign(first)) == 200
        _until(lambda: seen and _health(srv) == 503)
        assert _post(srv, second, sign(second)) == 200
        _until(lambda: json.loads(second) in seen)  # the worker is still alive
        assert len(srv.spool.waiting()) == 1  # the rejected event is kept, not lost
    finally:
        srv.spool.failed.chmod(0o700)
    _until(lambda: not srv.spool.waiting() and len(list(srv.spool.failed.iterdir())) == 1)
    _until(lambda: _health(srv) == 200)
