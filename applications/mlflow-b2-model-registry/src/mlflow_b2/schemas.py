"""Validation against the repository's shared JSON Schemas."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from .config import ConfigError

SCHEMA_DIR = Path(__file__).resolve().parents[4] / "schemas"


def errors(doc, name: str, schema_dir: Path | str = SCHEMA_DIR) -> list[str]:
    path = Path(schema_dir) / f"{name}.schema.json"
    if not path.exists():
        raise ConfigError(f"schema not found: {path}; pass --schema-dir")
    validator = Draft202012Validator(json.loads(path.read_text()))
    return sorted(f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
                  for e in validator.iter_errors(doc))
