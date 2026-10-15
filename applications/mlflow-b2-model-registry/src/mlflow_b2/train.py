"""Train on the local WebDataset shards and log the run, dataset pin and model to MLflow."""
from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import cloudpickle
import mlflow
import numpy as np
from mlflow.models import infer_signature

from . import model, registry, schemas
from .config import Config


class DatasetError(Exception):
    pass


def load_manifest(data: Path, schema_dir=schemas.SCHEMA_DIR) -> dict:
    path = data / "manifest.json"
    if not path.exists():
        raise DatasetError(f"{path} not found; --data must point at a shard manifest directory")
    m = json.loads(path.read_text())
    errs = schemas.errors(m, "shard-manifest", schema_dir)
    for f in [] if errs else m["files"]:
        shard = data / f["path"]
        if not shard.exists() or hashlib.sha256(shard.read_bytes()).hexdigest() != f["sha256"]:
            errs.append(f"{f['path']}: missing or checksum differs from manifest")
    if not errs and m["fingerprint"] != registry.fingerprint(m["files"]):
        errs.append("fingerprint does not match the listed files")  # it becomes the run's dataset pin
    if errs:
        raise DatasetError("shard manifest invalid:\n  " + "\n  ".join(errs))
    return m


def read_split(data: Path, m: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Group tar members by WebDataset key (path up to the first dot of the basename)."""
    image, label = m["sample_keys"]["image"], m["sample_keys"]["label"]
    samples: dict[str, dict[str, bytes]] = {}
    for shard in m["splits"][split]["shards"]:
        with tarfile.open(data / shard) as tar:
            for member in tar:
                head, _, name = member.name.rpartition("/")
                stem, _, ext = name.partition(".")
                if member.isfile() and ext in (image, label):
                    samples.setdefault(f"{shard}:{head}/{stem}", {})[ext] = tar.extractfile(member).read()
    incomplete = [k for k, s in samples.items() if image not in s or label not in s]
    if incomplete or not samples:
        raise DatasetError(f"split {split}: {len(incomplete)} incomplete samples of {len(samples)}")
    keys = sorted(samples)
    x = np.stack([model.features(samples[k][image]) for k in keys])
    return x, np.array([int(samples[k][label]) for k in keys])


def run(cfg: Config, data: Path, *, experiment: str, epochs: int, lr: float, batch_size: int = 16,
        seed: int = 0, register_as: str | None = None, schema_dir=schemas.SCHEMA_DIR):
    m = load_manifest(data, schema_dir)
    x, y = read_split(data, m, "train")
    xv, yv = read_split(data, m, "val")
    params = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed,
              "image_size": model.SIZE, "train_samples": len(y), "val_samples": len(yv)}
    dataset_tags = {"dataset.id": m["id"], "dataset.fingerprint": m["fingerprint"]}
    # Serialize the pyfunc class by value so serving needs only mlflow + numpy, not this package.
    cloudpickle.register_pickle_by_value(model)
    with tempfile.TemporaryDirectory() as tmp, \
            mlflow.start_run(experiment_id=registry.experiment_id(cfg, experiment)) as active:
        mlflow.set_tags(dataset_tags)
        mlflow.log_params(params)
        mlflow.log_artifact(str(data / "manifest.json"), "dataset")
        w = model.fit(x, y, xv, yv, m["classes"], epochs=epochs, lr=lr, batch_size=batch_size,
                      seed=seed, on_epoch=lambda e, metrics: mlflow.log_metrics(metrics, step=e))
        weights = Path(tmp) / "weights.npz"
        np.savez(weights, **w)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model.ShapesClassifier(),
            artifacts={"weights": str(weights)},
            signature=infer_signature(xv[:2], w["classes"][yv[:2]]),
            pip_requirements=[f"mlflow=={mlflow.__version__}", f"numpy=={np.__version__}"],
            registered_model_name=register_as,
        )
    version = info.registered_model_version and str(info.registered_model_version)
    if register_as and version:
        client = mlflow.MlflowClient()
        for k, v in {**dataset_tags, "source.run_id": active.info.run_id}.items():
            client.set_model_version_tag(register_as, version, k, v)
    return active.info.run_id, version
