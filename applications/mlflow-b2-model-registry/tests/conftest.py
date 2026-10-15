import hashlib
import hmac
import json
import os
from pathlib import Path

ENDPOINT = "https://s3.us-east-1.backblazeb2.com"
# moto reads this when its S3 URL table is first imported, so it must be set before moto loads.
# It routes the b2:// host through moto, so MLflow's real B2ArtifactRepository runs end to end.
os.environ.setdefault("MOTO_S3_CUSTOM_ENDPOINTS", ENDPOINT)

import boto3  # noqa: E402
import pytest  # noqa: E402
from botocore.config import Config as BotoConfig  # noqa: E402
from moto import mock_aws  # noqa: E402

from mlflow_b2 import cli, storage  # noqa: E402
from mlflow_b2.config import Config  # noqa: E402

BUCKET = "test-mlflow-b2-bucket"
SECRET = "SUPERSECRETKEYVALUE123"
EVENT_SECRET = "event-signing-secret-0123456789ab"
DATA = Path(__file__).resolve().parents[3] / "testdata" / "synthetic-shapes"
WEIGHTS = b"not really torch weights"


@pytest.fixture
def cfg(tmp_path):
    return Config(key_id="keyid123", key=SECRET, bucket=BUCKET, region="us-east-1", endpoint=ENDPOINT,
                  tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", event_secret=EVENT_SECRET)


@pytest.fixture
def s3(cfg, monkeypatch):
    """moto covers S3 data operations only; key scoping, Object Lock and notifications need live B2."""
    for n in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(n, "test")  # registry.configure overwrites these; monkeypatch restores
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1", endpoint_url=ENDPOINT,
                              config=BotoConfig(signature_version="s3v4"))
        client.create_bucket(Bucket=BUCKET)
        monkeypatch.setattr(storage, "make_client", lambda _cfg: client)
        yield client


@pytest.fixture
def run(cfg, monkeypatch):
    monkeypatch.setattr(cli, "load", lambda: cfg)
    return cli.main


def model_manifest(model="shapes-classifier", version="v2", fingerprint="c" * 64, **overrides):
    m = {
        "schema_version": "1.0.0", "id": f"{model}@{version}", "model": model, "version": version,
        "format": "torch.save", "framework": {"name": "torch", "version": "2.5.0"},
        "source": {"run_id": "run-0001", "step": 120, "checkpoint_fingerprint": "a" * 64},
        "dataset": {"id": "synthetic-shapes@v1", "fingerprint": "b" * 64},
        "metrics": {"val_accuracy": 0.9},
        "files": [{"path": "model.pt", "size": len(WEIGHTS),
                   "sha256": hashlib.sha256(WEIGHTS).hexdigest()}],
        "fingerprint": fingerprint, "retention_mode": "none",
    }
    return {**m, **overrides}


def publish(client, manifest, key=None):
    """Mimic the training app: files first, manifest last."""
    base = f"models/{manifest['model']}/{manifest['version']}/"
    client.put_object(Bucket=BUCKET, Key=base + "model.pt", Body=WEIGHTS)
    key = key or base + "manifest.json"
    client.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(manifest).encode())
    return key


def event_body(*keys, event_type="b2:ObjectCreated:Upload", bucket=BUCKET) -> bytes:
    return json.dumps({"events": [
        {"bucketName": bucket, "eventType": event_type, "objectName": k, "eventVersion": 1}
        for k in keys]}).encode()


def sign(body: bytes, secret: str = EVENT_SECRET) -> str:
    return "v1=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
