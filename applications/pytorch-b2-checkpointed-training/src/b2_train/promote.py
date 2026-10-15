"""Promote a committed checkpoint to models/<model>/<version>/ with server-side copies."""
from __future__ import annotations

from . import checkpoint, layout, manifest, storage


def promote(client, bucket, run: str, step: int | None, model: str, version: str, *,
            lock_days: int | None = None) -> tuple[dict, str]:
    if step is None:
        step = checkpoint.latest_complete(client, bucket, run)
        if step is None:
            raise storage.StorageError(f"run {run} has no complete checkpoint")
    ck = checkpoint.load_manifest(client, bucket, run, step)
    m = manifest.model(ck, model, version, "governance" if lock_days else "none")
    src, dst = layout.checkpoint_prefix(run, step), layout.model_prefix(model, version)
    data = manifest.dumps(m)
    existing = storage.head(client, bucket, dst + layout.MANIFEST_NAME)
    if existing:
        if existing["Metadata"].get("sha256") != manifest.sha256_bytes(data):
            raise storage.Collision(f"model {model}@{version} already exists with different content")
        return m, "skipped"
    # Check every destination before copying anything, so a refusal leaves no partial version.
    todo = []
    for f in ck["files"]:
        sha = storage.stored_sha(client, bucket, dst + f["path"])
        if sha is not None and sha != f["sha256"]:
            raise storage.Collision(f"{dst}{f['path']} already exists with different content")
        if sha is None:
            todo.append(f)
    for f in todo:
        storage.copy(client, bucket, src + f["path"], dst + f["path"], f["sha256"], lock_days=lock_days)
    errors = storage.verify(client, bucket, dst, ck["files"])
    if errors:
        raise storage.StorageError("model verification failed: " + "; ".join(errors))
    storage.put_bytes(client, bucket, dst + layout.MANIFEST_NAME, data, overwrite=False, lock_days=lock_days)
    return m, "promoted"
