"""Checksums, fingerprints and manifest builders for the shared contracts in ../../schemas."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
NAME_RE = re.compile(r"^[a-z0-9._-]+$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(files: list[dict]) -> str:
    """sha256 over sorted "path:sha256" lines; same rule as tools/build_testdata.py."""
    lines = "\n".join(f"{f['path']}:{f['sha256']}" for f in sorted(files, key=lambda f: f["path"]))
    return sha256_bytes(lines.encode())


def describe(root: Path) -> list[dict]:
    return [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(root.rglob("*")) if p.is_file()]


def safe_path(path: str) -> bool:
    """Relative POSIX path with no empty, '.' or '..' segments: it cannot escape its prefix or folder."""
    return (isinstance(path, str) and bool(path) and not path.startswith("/") and "\\" not in path
            and all(seg not in ("", ".", "..") for seg in path.split("/")))


def file_errors(m: dict) -> list[str]:
    """Paths are safe and the fingerprint is recomputed from the listed files, never trusted."""
    errors = [f"unsafe path {f.get('path')!r}" for f in m.get("files", []) if not safe_path(f.get("path"))]
    if not errors and fingerprint(m["files"]) != m["fingerprint"]:
        errors.append("fingerprint does not match the listed files")
    return errors


def dataset_errors(m: dict) -> list[str]:
    """Structure of a shard manifest, whether read from disk or from B2."""
    errors = [f"manifest missing {k}" for k in ("id", "dataset", "version", "files", "splits", "fingerprint",
                                                  "sample_keys") if k not in m]
    if errors:
        return errors
    errors = file_errors(m)
    listed = {f["path"] for f in m["files"]}
    errors += [f"split {s}: {p} not in files" for s, v in m["splits"].items() for p in v["shards"]
               if p not in listed]
    return errors


def check_dataset(src: Path, m: dict) -> list[str]:
    """Local shard files must match the manifest byte for byte before anything is uploaded."""
    errors = dataset_errors(m)
    if errors:
        return errors
    for f in m["files"]:
        p = src / f["path"]
        if not p.is_file():
            errors.append(f"{f['path']}: missing")
        elif p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]:
            errors.append(f"{f['path']}: size or sha256 differs from manifest")
    return errors


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def checkpoint(*, run_id, step, epoch, fmt, torch_version, dataset, world_size, metrics, files) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "step": step,
        "epoch": epoch,
        "format": fmt,
        "framework": {"name": "pytorch", "version": torch_version},
        "dataset": dataset,
        "world_size": world_size,
        "metrics": metrics,
        "files": files,
        "fingerprint": fingerprint(files),
        "created_at": now(),
    }


def model(ckpt: dict, name: str, version: str, retention_mode: str) -> dict:
    # No timestamps: promoting the same checkpoint again yields the same bytes.
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"{name}@{version}",
        "model": name,
        "version": version,
        "format": ckpt["format"],
        "framework": ckpt["framework"],
        "source": {"run_id": ckpt["run_id"], "step": ckpt["step"],
                   "checkpoint_fingerprint": ckpt["fingerprint"]},
        "dataset": ckpt["dataset"],
        "metrics": ckpt["metrics"],
        "files": ckpt["files"],
        "fingerprint": fingerprint(ckpt["files"]),
        "retention_mode": retention_mode,
    }


def dumps(m: dict) -> bytes:
    return json.dumps(m, indent=1).encode()
