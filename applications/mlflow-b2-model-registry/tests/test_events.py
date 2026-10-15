import hashlib
import json

from botocore.exceptions import ClientError
from conftest import BUCKET, WEIGHTS, event_body, model_manifest, publish, sign

from mlflow_b2 import events, registry, storage


def test_signature_roundtrip():
    body = b'{"events":[]}'
    assert events.verify_signature("s" * 32, body, sign(body, "s" * 32))
    assert not events.verify_signature("s" * 32, body, sign(body, "x" * 32))
    assert not events.verify_signature("s" * 32, body, "")
    assert not events.verify_signature("s" * 32, body, None)


def test_manifest_keys_filter_type_pattern_bucket():
    payload = json.loads(event_body("models/m/v1/manifest.json", "models/m/v1/model.pt",
                                    "models/m/manifest.json", "checkpoints/r/step-1/manifest.json",
                                    "models/m/v1/manifest.json"))
    payload["events"] += json.loads(event_body("models/m/v2/manifest.json",
                                               event_type="b2:ObjectDeleted:Delete"))["events"]
    payload["events"] += json.loads(event_body("models/m/v3/manifest.json", bucket="other"))["events"]
    assert events.manifest_keys(payload, BUCKET) == ["models/m/v1/manifest.json"]


def _handle(cfg, s3, *keys):
    client = registry.configure(cfg)
    return registry.handle_payload(cfg, client, s3, json.loads(event_body(*keys))), client


def test_registers_version_pointing_at_b2_prefix(s3, cfg, capsys):
    key = publish(s3, model_manifest())
    rc, client = _handle(cfg, s3, key)
    assert rc == 0 and "registered shapes-classifier version 1" in capsys.readouterr().out
    mv = client.get_model_version("shapes-classifier", "1")
    assert mv.source == "b2://test-mlflow-b2-bucket@s3.us-east-1.backblazeb2.com/models/shapes-classifier/v2/"
    assert mv.tags["dataset.id"] == "synthetic-shapes@v1" and mv.tags["dataset.fingerprint"] == "b" * 64
    assert mv.tags["source.run_id"] == "run-0001" and mv.tags["source.step"] == "120"
    assert mv.tags["source.checkpoint_fingerprint"] == "a" * 64
    assert mv.tags[registry.FINGERPRINT_TAG] == model_manifest()["fingerprint"]


def test_replay_is_idempotent(s3, cfg, capsys):
    key = publish(s3, model_manifest())
    _handle(cfg, s3, key)
    rc, client = _handle(cfg, s3, key, key)
    assert rc == 0 and "skipped" in capsys.readouterr().out
    assert len(client.search_model_versions("name = 'shapes-classifier'")) == 1
    publish(s3, model_manifest(version="v3", weights=b"other weights"), weights=b"other weights")
    _handle(cfg, s3, "models/shapes-classifier/v3/manifest.json")
    assert len(client.search_model_versions("name = 'shapes-classifier'")) == 2


def test_non_matching_events_ignored(s3, cfg, capsys):
    rc, client = _handle(cfg, s3, "models/shapes-classifier/v2/model.pt", "datasets/x/v1/manifest.json")
    assert rc == 0 and "nothing to do" in capsys.readouterr().out
    assert client.search_registered_models() == []


def test_schema_invalid_manifest_rejected(s3, cfg, capsys):
    bad = model_manifest()
    del bad["source"]
    rc, client = _handle(cfg, s3, publish(s3, bad))
    assert rc == 2 and "source" in capsys.readouterr().err
    assert client.search_registered_models() == []


def test_key_mismatch_and_missing_files_rejected(s3, cfg, capsys):
    key = publish(s3, model_manifest(), key="models/shapes-classifier/v9/manifest.json")
    rc, client = _handle(cfg, s3, key)
    assert rc == 2 and "do not match" in capsys.readouterr().err
    m = model_manifest(version="v4")
    s3.put_object(Bucket=BUCKET, Key="models/shapes-classifier/v4/manifest.json", Body=json.dumps(m))
    rc, _ = _handle(cfg, s3, "models/shapes-classifier/v4/manifest.json")
    assert rc == 2 and "model.pt" in capsys.readouterr().err
    assert client.search_registered_models() == []


def test_stale_fingerprint_rejected(s3, cfg, capsys):
    """A copied fingerprint must not make a different model look already registered."""
    publish(s3, model_manifest())
    _, client = _handle(cfg, s3, "models/shapes-classifier/v2/manifest.json")
    stale = model_manifest(version="v3", weights=b"other weights",
                           fingerprint=model_manifest()["fingerprint"])
    rc, _ = _handle(cfg, s3, publish(s3, stale, weights=b"other weights"))
    assert rc == 2 and "fingerprint does not match" in capsys.readouterr().err
    assert len(client.search_model_versions("name = 'shapes-classifier'")) == 1


def test_changed_content_rejected(s3, cfg, capsys):
    """Same size, different bytes: caught by the stored sha256, which a size check alone misses."""
    key = publish(s3, model_manifest())
    swapped = b"X" + WEIGHTS[1:]
    s3.put_object(Bucket=BUCKET, Key="models/shapes-classifier/v2/model.pt", Body=swapped,
                  Metadata={"sha256": hashlib.sha256(swapped).hexdigest()})
    rc, client = _handle(cfg, s3, key)
    assert rc == 2 and "sha256" in capsys.readouterr().err
    s3.put_object(Bucket=BUCKET, Key="models/shapes-classifier/v2/model.pt", Body=WEIGHTS)
    rc, _ = _handle(cfg, s3, key)
    assert rc == 2 and "metadata missing" in capsys.readouterr().err
    assert client.search_registered_models() == []


def test_deleted_manifest_is_storage_error(s3, cfg, capsys):
    rc, _ = _handle(cfg, s3, "models/shapes-classifier/v2/manifest.json")
    assert rc == 1 and "not found" in capsys.readouterr().err


def test_get_bytes_caps_size(s3):
    s3.put_object(Bucket=BUCKET, Key="big", Body=b"x" * (storage.MAX_MANIFEST_BYTES + 1))
    try:
        storage.get_bytes(s3, BUCKET, "big")
    except storage.StorageError as e:
        assert "larger" in str(e)
    else:
        raise AssertionError("expected StorageError")


def _flaky_head(s3, monkeypatch, failures=1, code="503"):
    """HEAD fails with a temporary B2 error `failures` times, then behaves normally."""
    real, calls = s3.head_object, []

    def head(**kw):
        calls.append(kw["Key"])
        if len(calls) <= failures:
            raise ClientError({"Error": {"Code": code, "Message": "SlowDown"}}, "HeadObject")
        return real(**kw)
    monkeypatch.setattr(s3, "head_object", head)
    return calls


def test_temporary_head_error_is_retryable_not_rejected(s3, cfg, monkeypatch, capsys):
    key = publish(s3, model_manifest())
    _flaky_head(s3, monkeypatch)
    rc, client = _handle(cfg, s3, key)
    assert rc == 1 and "503" in capsys.readouterr().err
    assert client.search_registered_models() == []
    rc, _ = _handle(cfg, s3, key)
    assert rc == 0 and len(client.search_model_versions("name = 'shapes-classifier'")) == 1


def test_retryable_failure_wins_over_rejection_in_one_batch(s3, cfg, monkeypatch):
    bad = model_manifest(version="v5")
    del bad["source"]
    rejected = publish(s3, bad)
    ok = publish(s3, model_manifest(version="v6"))
    _flaky_head(s3, monkeypatch)
    rc, client = _handle(cfg, s3, rejected, ok)
    assert rc == 1  # v6 must be retried, not discarded with v5
    rc, _ = _handle(cfg, s3, rejected, ok)
    assert rc == 2 and len(client.search_model_versions("name = 'shapes-classifier'")) == 1
