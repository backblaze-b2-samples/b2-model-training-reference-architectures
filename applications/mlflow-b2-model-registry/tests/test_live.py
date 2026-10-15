"""Live B2 round trip through MLflow's b2:// store. Opt-in: RUN_B2_LIVE_TESTS=1 plus a .env."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_B2_LIVE_TESTS") != "1",
                                reason="set RUN_B2_LIVE_TESTS=1 with B2 credentials")


def test_live_train_promote(tmp_path, monkeypatch):
    from conftest import DATA

    from mlflow_b2 import cli
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp = "live-test-" + os.urandom(4).hex()  # artifacts land under <prefix>/<exp>/; delete after
    assert cli.main(["artifact-uri"]) == 0
    assert cli.main(["train", "--data", str(DATA), "--experiment", exp, "--epochs", "1",
                     "--register-as", exp]) == 0
    assert cli.main(["promote", "--model", exp, "--version", "1"]) == 0
