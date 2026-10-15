import json
import shutil
from pathlib import Path

import boto3
import pytest
from botocore.config import Config as BotoConfig
from jsonschema import Draft202012Validator
from moto import mock_aws

from b2_train import cli, storage
from b2_train.config import Config

BUCKET = "test-b2-train-bucket"
SECRET = "SUPERSECRETKEYVALUE123"
ROOT = Path(__file__).resolve().parents[3]
SHAPES = ROOT / "testdata" / "synthetic-shapes"


@pytest.fixture
def cfg():
    return Config(key_id="keyid123", key=SECRET, bucket=BUCKET, region="us-east-1",
                  endpoint="https://s3.us-east-1.backblazeb2.com")


@pytest.fixture
def s3(cfg, monkeypatch):
    """moto covers S3 data operations only; key scoping, Object Lock and lifecycle need live B2."""
    with mock_aws():
        # Pin SigV4 like make_client; the SDK default varies by environment (CI signs with SigV2).
        client = boto3.client("s3", region_name="us-east-1",
                              config=BotoConfig(signature_version="s3v4"))
        client.create_bucket(Bucket=BUCKET)
        monkeypatch.setattr(storage, "make_client", lambda _cfg: client)
        yield client


@pytest.fixture
def shapes(tmp_path):
    """Private copy of the shared dataset so corruption tests cannot touch testdata/."""
    return Path(shutil.copytree(SHAPES, tmp_path / "synthetic-shapes"))


@pytest.fixture
def run(cfg, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # staging/ lands in tmp
    monkeypatch.setattr(cli, "load", lambda: cfg)
    return cli.main


@pytest.fixture
def published(s3, run, shapes):
    assert run(["publish-dataset", "--src", str(shapes)]) == 0
    return json.loads((shapes / "manifest.json").read_text())


@pytest.fixture
def calls(s3):
    """(operation, key) for every request the client sends, in order."""
    seen = []
    s3.meta.events.register("before-parameter-build.s3.*", lambda params, model, **kw:
                            seen.append((model.name, params.get("Key"))))
    return seen


@pytest.fixture
def validate():
    def check(schema: str, doc: dict) -> dict:
        schema_doc = json.loads((ROOT / "schemas" / f"{schema}.schema.json").read_text())
        errors = [e.message for e in Draft202012Validator(schema_doc).iter_errors(doc)]
        assert not errors, errors
        return doc
    return check


def get_json(s3, key: str, bucket: str = BUCKET) -> dict:
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
