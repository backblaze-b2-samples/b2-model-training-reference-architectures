"""Live B2 round trip (key scoping, real streaming). Opt-in: RUN_B2_LIVE_TESTS=1 plus a .env."""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_B2_LIVE_TESTS") != "1",
                                reason="set RUN_B2_LIVE_TESTS=1 with B2 credentials")

APP = Path(__file__).resolve().parents[1]
SHAPES = APP.parents[1] / "testdata" / "synthetic-shapes"


def test_live_train_resume_promote_prune(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    from b2_train import cli
    run_id = "live-" + os.urandom(4).hex()  # unique run and model version; locked versions cannot be reused
    for name, value in dotenv_values(APP / ".env").items():  # read before chdir; cli loads ./.env
        if value and name not in os.environ:
            monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["publish-dataset", "--src", str(SHAPES)]) == 0  # idempotent if already published
    train = ["train", "--run-id", run_id, "--checkpoint-every", "3", "--cache-dir", str(tmp_path / "cache")]
    assert cli.main(train + ["--max-steps", "4"]) == 0
    assert cli.main(train) == 0
    # B2_LIVE_LOCK_DAYS=1 tests governance retention; needs an Object-Lock bucket and writeFileRetentions,
    # and the promoted objects then cannot be deleted until the retention date passes.
    lock = os.environ.get("B2_LIVE_LOCK_DAYS")
    retention = ["--lock-days", lock] if lock else ["--no-lock"]
    assert cli.main(["promote", "--run-id", run_id, "--model", "live-test", "--model-version", run_id,
                     *retention]) == 0
    assert cli.main(["prune", "--run-id", run_id, "--keep", "1"]) == 0
