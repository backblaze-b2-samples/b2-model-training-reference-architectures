import hashlib
import json
import shutil

import mlflow
import numpy as np
import pytest
from conftest import BUCKET, DATA
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.b2_artifact_repo import B2ArtifactRepository

from mlflow_b2 import registry, train


def test_read_split_counts():
    m = train.load_manifest(DATA)
    x, y = train.read_split(DATA, m, "train")
    assert x.shape == (80, 64) and set(y) == {0, 1, 2, 3}
    assert train.read_split(DATA, m, "val")[0].shape[0] == 16


def test_tampered_shard_rejected(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    with open(data / "shards" / "val-000000.tar", "ab") as f:
        f.write(b"x")
    with pytest.raises(train.DatasetError, match="checksum"):
        train.load_manifest(data)


def test_changed_shards_under_original_fingerprint_rejected(tmp_path):
    """Shard hashes updated to match new content, top-level pin left alone: not a valid dataset."""
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    shard = data / "shards" / "val-000000.tar"
    with open(shard, "ab") as f:
        f.write(b"\0" * 512)
    m = json.loads((data / "manifest.json").read_text())
    entry = next(f for f in m["files"] if f["path"] == "shards/val-000000.tar")
    entry["size"], entry["sha256"] = shard.stat().st_size, hashlib.sha256(shard.read_bytes()).hexdigest()
    (data / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(train.DatasetError, match="fingerprint does not match"):
        train.load_manifest(data)


def test_train_logs_to_b2_and_registers(s3, cfg):
    client = registry.configure(cfg)
    run_id, version = train.run(cfg, DATA, experiment="synthetic-shapes", epochs=3, lr=0.05,
                                register_as="shapes-classifier")
    assert version == "1"
    location = client.get_experiment_by_name("synthetic-shapes").artifact_location
    assert location == f"{cfg.artifact_root}/synthetic-shapes"
    assert isinstance(get_artifact_repository(location), B2ArtifactRepository)
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert all(k.startswith("mlflow/synthetic-shapes/") for k in keys)
    assert f"mlflow/synthetic-shapes/{run_id}/artifacts/dataset/manifest.json" in keys
    assert any(k.endswith("/artifacts/MLmodel") for k in keys)
    assert any(k.endswith("/artifacts/weights.npz") for k in keys)

    run = client.get_run(run_id)
    assert run.data.tags["dataset.id"] == "synthetic-shapes@v1"
    assert len(run.data.tags["dataset.fingerprint"]) == 64
    assert run.data.params["epochs"] == "3"
    assert [m.step for m in client.get_metric_history(run_id, "val_accuracy")] == [0, 1, 2]
    mv = client.get_model_version("shapes-classifier", "1")
    assert mv.tags["dataset.id"] == "synthetic-shapes@v1" and mv.tags["source.run_id"] == run_id

    model = mlflow.pyfunc.load_model("models:/shapes-classifier/1")  # downloads through b2://
    m = train.load_manifest(DATA)
    xv, yv = train.read_split(DATA, m, "val")
    pred = model.predict(xv)
    assert (pred == np.array(m["classes"])[yv]).mean() > 0.9
