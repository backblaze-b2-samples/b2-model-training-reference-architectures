import http.client
import threading
import time
import urllib.error
import urllib.request

import pytest
from conftest import EVENT_SECRET, event_body, model_manifest, publish, sign

from mlflow_b2 import events, registry, server


@pytest.fixture
def serve():
    started = []

    def start(handle):
        srv = server.make_server(EVENT_SECRET, handle, "127.0.0.1", 0)
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
    srv = serve(lambda p: None)
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
