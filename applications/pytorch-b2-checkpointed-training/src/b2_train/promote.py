"""Promote a committed checkpoint to models/<model>/<version>/ with verified server-side copies."""
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
    until = storage.lock_until(lock_days)
    # Everything is checked before anything is written, so a refusal leaves no new objects behind.
    etags = {f["path"]: storage.verify_object(client, bucket, src + f["path"], f) for f in ck["files"]}
    todo, reused = [], []
    for f in ck["files"]:
        h = storage.head(client, bucket, dst + f["path"])
        if h is None:
            todo.append(f)
            continue
        if h["Metadata"].get("sha256") != f["sha256"]:
            raise storage.Collision(f"{dst}{f['path']} already exists with different content")
        if until is None and storage.retention(h)[0]:
            raise storage.Collision(f"{dst}{f['path']} is already under retention from an earlier attempt; "
                                    "promote with --lock-days or to a new version")
        storage.verify_object(client, bucket, dst + f["path"], f)  # leftover of an interrupted promotion
        reused.append(f)
    for f in todo:
        storage.copy(client, bucket, src + f["path"], dst + f["path"], f["sha256"], etag=etags[f["path"]],
                     until=until)
    if until:
        for f in reused:
            storage.ensure_retention(client, bucket, dst + f["path"], until)
    errors = storage.verify(client, bucket, dst, ck["files"])
    if until:
        errors += storage.retention_errors(client, bucket, [dst + f["path"] for f in ck["files"]], until)
    if errors:
        raise storage.StorageError("model verification failed: " + "; ".join(errors))
    storage.put_bytes(client, bucket, dst + layout.MANIFEST_NAME, data, overwrite=False, until=until)
    if until:
        errors = storage.retention_errors(client, bucket, [dst + layout.MANIFEST_NAME], until)
        if errors:
            raise storage.StorageError("model verification failed: " + "; ".join(errors))
    return m, "promoted"
