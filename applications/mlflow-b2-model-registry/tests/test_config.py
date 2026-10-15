import pytest

from mlflow_b2 import config, storage
from mlflow_b2.config import ConfigError

OPTIONAL = ("B2_ENDPOINT", "MLFLOW_TRACKING_URI", "B2_MLFLOW_PREFIX", "B2_EVENT_SIGNING_SECRET")


def _set(monkeypatch, region="eu-central-003", **extra):
    for n in (*config.REQUIRED, *OPTIONAL):
        monkeypatch.delenv(n, raising=False)
    for n, v in zip(config.REQUIRED, ("id", "topsecret", "bkt", region)):
        monkeypatch.setenv(n, v)
    for n, v in extra.items():
        monkeypatch.setenv(n, v)


def test_missing_env_lists_names_only(monkeypatch, tmp_path):
    _set(monkeypatch)
    monkeypatch.delenv("B2_BUCKET_NAME")
    with pytest.raises(ConfigError) as e:
        config.load(str(tmp_path / "none.env"))
    assert "B2_BUCKET_NAME" in str(e.value) and "topsecret" not in str(e.value)


def test_defaults_endpoint_and_artifact_root(monkeypatch, tmp_path):
    _set(monkeypatch, B2_EVENT_SIGNING_SECRET="hooksecret")
    cfg = config.load(str(tmp_path / "none.env"))
    assert cfg.endpoint == "https://s3.eu-central-003.backblazeb2.com"
    assert cfg.tracking_uri == "sqlite:///mlflow.db"
    assert cfg.artifact_root == "b2://bkt@s3.eu-central-003.backblazeb2.com/mlflow"
    assert "topsecret" not in repr(cfg) and "hooksecret" not in repr(cfg)


def test_prefix_and_tracking_overrides(monkeypatch, tmp_path):
    _set(monkeypatch, B2_MLFLOW_PREFIX="/team/mlflow/", MLFLOW_TRACKING_URI="http://mlflow:5000")
    cfg = config.load(str(tmp_path / "none.env"))
    assert cfg.artifact_root == "b2://bkt@s3.eu-central-003.backblazeb2.com/team/mlflow"
    assert cfg.tracking_uri == "http://mlflow:5000"
    assert cfg.b2_uri("models/m/v1/") == "b2://bkt@s3.eu-central-003.backblazeb2.com/models/m/v1/"


def test_endpoint_override_must_be_b2_form(monkeypatch, tmp_path):
    _set(monkeypatch, B2_ENDPOINT="https://s3.us-west-004.backblazeb2.com")
    assert config.load(str(tmp_path / "none.env")).artifact_root.startswith(
        "b2://bkt@s3.us-west-004.backblazeb2.com/")
    _set(monkeypatch, B2_ENDPOINT="https://example.test")
    with pytest.raises(ConfigError):
        config.load(str(tmp_path / "none.env")).artifact_root


def test_own_client_uses_sigv4_and_user_agent(cfg):
    c = storage.make_client(cfg).meta.config
    assert c.signature_version == "s3v4" and c.user_agent_extra == storage.USER_AGENT
    assert c.retries["mode"] == "standard" and c.request_checksum_calculation == "when_required"
