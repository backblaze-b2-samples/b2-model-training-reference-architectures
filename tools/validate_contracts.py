#!/usr/bin/env python3
"""Validate catalog, shared dataset, manifest examples, schemas, and metadata files."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_testdata import croissant, fingerprint  # noqa: E402

DATASET = ROOT / "testdata" / "synthetic-shapes"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def validate(instance: dict, schema_path: str) -> None:
    schema = load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(instance)


def check_dataset() -> dict:
    shard = json.loads((DATASET / "manifest.json").read_text())
    validate(shard, "schemas/shard-manifest.schema.json")
    for f in shard["files"]:
        data = (DATASET / f["path"]).read_bytes()
        assert len(data) == f["size"], f"{f['path']}: size differs from manifest"
        assert hashlib.sha256(data).hexdigest() == f["sha256"], f"{f['path']}: sha256 differs from manifest"
    assert shard["fingerprint"] == fingerprint(shard["files"])
    assert shard["statistics"]["file_count"] == len(shard["files"])
    assert shard["statistics"]["total_bytes"] == sum(f["size"] for f in shard["files"])
    assert shard["statistics"]["sample_count"] == sum(f["samples"] for f in shard["files"])
    listed = sorted(p for split in shard["splits"].values() for p in split["shards"])
    assert listed == sorted(f["path"] for f in shard["files"]), "splits and files disagree"
    assert json.loads((DATASET / "croissant.json").read_text()) == croissant(shard), "croissant.json is stale"
    return shard


def check_run_manifests(shard: dict) -> None:
    pin = {"id": shard["id"], "fingerprint": shard["fingerprint"]}
    checkpoint = load_json("manifests/checkpoint-manifest.example.json")
    validate(checkpoint, "schemas/checkpoint-manifest.schema.json")
    model = load_json("manifests/model-manifest.example.json")
    validate(model, "schemas/model-manifest.schema.json")
    for name, m in (("checkpoint", checkpoint), ("model", model)):
        assert m["dataset"] == pin, f"{name} example is not pinned to the shared dataset"
        assert m["fingerprint"] == fingerprint(m["files"]), f"{name} example fingerprint is stale"
    assert model["source"]["run_id"] == checkpoint["run_id"]
    assert model["source"]["step"] == checkpoint["step"]
    assert model["source"]["checkpoint_fingerprint"] == checkpoint["fingerprint"]


def main() -> int:
    catalog = yaml.safe_load((ROOT / "catalog" / "architectures.yaml").read_text())
    validate(catalog, "schemas/architecture-catalog.schema.json")
    check_run_manifests(check_dataset())
    load_json("codemeta.json")
    print("catalog, shared dataset, and run contracts valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
