"""MLflow tracking/registry setup, alias promotion and event-driven model registration."""
from __future__ import annotations

import json
import os
import re
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from . import events, schemas, storage
from .config import Config, ConfigError

NAME = re.compile(r"^[A-Za-z0-9._-]+$")
FINGERPRINT_TAG = "manifest.fingerprint"


class InvalidManifest(Exception):
    pass


def configure(cfg: Config) -> MlflowClient:
    """Point MLflow at the tracking store and hand it the B2 key via the boto3 env chain."""
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.key
    os.environ.pop("AWS_SESSION_TOKEN", None)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    return MlflowClient(cfg.tracking_uri)


def experiment_id(cfg: Config, name: str) -> str:
    if not NAME.match(name):
        raise ConfigError("experiment names may contain only letters, digits, '.', '_' and '-'")
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name, artifact_location=f"{cfg.artifact_root}/{name}")
    if not exp.artifact_location.startswith(cfg.artifact_root + "/"):
        print(f"warning: experiment '{name}' stores artifacts outside {cfg.artifact_root}",
              file=sys.stderr)
    return exp.experiment_id


def promote(client: MlflowClient, model: str, version: str, alias: str) -> None:
    client.get_model_version(model, version)
    client.set_registered_model_alias(model, alias, version)


def version_with_fingerprint(client: MlflowClient, model: str, fingerprint: str) -> str | None:
    token = None
    while True:
        page = client.search_model_versions(f"name = '{model}'", max_results=200, page_token=token)
        for mv in page:
            if mv.tags.get(FINGERPRINT_TAG) == fingerprint:
                return mv.version
        token = page.token
        if not token:
            return None


def register_manifest(cfg: Config, client: MlflowClient, s3, key: str,
                      schema_dir=schemas.SCHEMA_DIR) -> str:
    """Register models/<model>/<version>/ from its manifest; skip if already registered."""
    model, version = events.MANIFEST_KEY.match(key).groups()
    try:
        m = json.loads(storage.get_bytes(s3, cfg.bucket, key))
    except ValueError as e:
        raise InvalidManifest("not valid JSON") from e
    errs = schemas.errors(m, "model-manifest", schema_dir)
    if not errs and (m["model"], m["version"]) != (model, version):
        errs.append("model/version do not match the object key")
    prefix = f"models/{model}/{version}/"
    if not errs:
        errs = storage.missing_files(s3, cfg.bucket, prefix, m["files"])
    if errs:
        raise InvalidManifest("; ".join(errs))
    existing = version_with_fingerprint(client, model, m["fingerprint"])
    if existing:
        return f"skipped {model}: version {existing} already has manifest {m['fingerprint'][:12]}"
    try:
        client.create_registered_model(model)
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    src = m["source"]
    mv = client.create_model_version(
        model, cfg.b2_uri(prefix),
        tags={"dataset.id": m["dataset"]["id"], "dataset.fingerprint": m["dataset"]["fingerprint"],
              "source.run_id": src["run_id"], "source.step": str(src["step"]),
              "source.checkpoint_fingerprint": src["checkpoint_fingerprint"],
              "model.format": m["format"], FINGERPRINT_TAG: m["fingerprint"]},
        description=f"{m['id']} promoted from run {src['run_id']} step {src['step']}",
    )
    return f"registered {model} version {mv.version} from {m['id']}"


def handle_payload(cfg: Config, client: MlflowClient, s3, payload: dict,
                   schema_dir=schemas.SCHEMA_DIR) -> int:
    """Exit-code style: 0 ok, 1 storage error, 2 a manifest was rejected."""
    keys = events.manifest_keys(payload, cfg.bucket)
    if not keys:
        print("no model manifests in payload; nothing to do")
    rc = 0
    for key in keys:
        try:
            print(register_manifest(cfg, client, s3, key, schema_dir))
        except InvalidManifest as e:
            print(f"rejected {key}: {e}", file=sys.stderr)
            rc = 2
        except storage.StorageError as e:
            print(f"error: {e}", file=sys.stderr)
            rc = max(rc, 1)
    return rc
