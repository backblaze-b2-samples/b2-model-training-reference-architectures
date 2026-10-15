"""Checkpoint commit protocol: stage locally, upload files, verify, write the manifest last."""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from . import layout, manifest, storage

DCP = "torch.distributed.checkpoint"


def state(model, opt, progress: torch.Tensor) -> dict:
    msd, osd = get_state_dict(model, opt)
    return {"model": msd, "optim": osd, "progress": progress}


def save_local(model, opt, progress: torch.Tensor, path: Path) -> str:
    """DCP writes .metadata plus one .distcp file per rank; a single process needs no process group."""
    shutil.rmtree(path, ignore_errors=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*single process.*")
        dcp.save(state(model, opt, progress), checkpoint_id=str(path))
    return DCP


def load_local(model, opt, path: Path, fmt: str) -> torch.Tensor:
    if fmt != DCP:
        raise storage.StorageError(f"unsupported checkpoint format {fmt}")
    sd = state(model, opt, torch.zeros(3, dtype=torch.int64))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*single process.*")
        dcp.load(sd, checkpoint_id=str(path))
    set_state_dict(model, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
    return sd["progress"]


def scan(client, bucket, run: str) -> dict[int, dict]:
    """step -> {"keys": [...], "complete": manifest present}. Only the manifest makes a step complete."""
    root, steps = layout.run_prefix(run), {}
    for key in storage.list_keys(client, bucket, root):
        m = layout.STEP_RE.match(key[len(root):])
        if m:
            s = steps.setdefault(int(m.group(1)), {"keys": [], "complete": False})
            s["keys"].append(key)
            s["complete"] |= key.endswith("/" + layout.MANIFEST_NAME)
    return steps


def latest_complete(client, bucket, run: str) -> int | None:
    done = [s for s, v in scan(client, bucket, run).items() if v["complete"]]
    return max(done) if done else None


def commit(client, bucket, run: str, step: int, staged: Path, **fields) -> dict:
    prefix = layout.checkpoint_prefix(run, step)
    if storage.head(client, bucket, prefix + layout.MANIFEST_NAME):
        raise storage.Collision(f"checkpoint {run}/step-{step:08d} is already committed")
    files = manifest.describe(staged)
    # Files of an uncommitted step may be leftovers of a crashed attempt; overwrite them.
    storage.upload_tree(client, bucket, [(prefix + f["path"], staged / f["path"]) for f in files])
    errors = storage.verify(client, bucket, prefix, files)
    if errors:
        raise storage.StorageError("checkpoint verification failed: " + "; ".join(errors))
    m = manifest.checkpoint(run_id=run, step=step, files=files, **fields)
    storage.put_bytes(client, bucket, prefix + layout.MANIFEST_NAME, manifest.dumps(m), overwrite=False)
    return m


def load_manifest(client, bucket, run: str, step: int) -> dict:
    what = f"checkpoint {run}/step-{step:08d}"
    m = storage.load_json(client, bucket, layout.checkpoint_prefix(run, step) + layout.MANIFEST_NAME, what)
    errors = manifest.file_errors(m)
    if errors:
        raise storage.StorageError(f"{what} manifest is inconsistent: " + "; ".join(errors))
    return m


def fetch(client, bucket, run: str, step: int, dest: Path) -> dict:
    m = load_manifest(client, bucket, run, step)
    shutil.rmtree(dest, ignore_errors=True)
    prefix, root = layout.checkpoint_prefix(run, step), dest.resolve()
    for f in m["files"]:
        target = (dest / f["path"]).resolve()
        if root not in target.parents:  # second guard behind safe_path: never write outside dest
            raise storage.StorageError(f"{f['path']}: resolves outside {dest}")
        storage.download(client, bucket, prefix + f["path"], target, f["size"], f["sha256"])
    return m


def prune(client, bucket, run: str, keep: int) -> dict[str, int]:
    """Keep the newest `keep` complete steps and anything newer than the newest complete one."""
    steps = scan(client, bucket, run)
    done = sorted((s for s, v in steps.items() if v["complete"]), reverse=True)
    newest = done[0] if done else None
    doomed = set(done[keep:]) | {s for s, v in steps.items()
                                 if not v["complete"] and newest is not None and s < newest}
    # Manifests go first so an interrupted prune leaves incomplete (ignored) steps, never half-complete ones.
    manifests = [k for s in doomed for k in steps[s]["keys"] if k.endswith("/" + layout.MANIFEST_NAME)]
    rest = [k for s in doomed for k in steps[s]["keys"] if not k.endswith("/" + layout.MANIFEST_NAME)]
    deleted = storage.delete_keys(client, bucket, manifests) + storage.delete_keys(client, bucket, rest)
    return {"checkpoints_deleted": len(doomed), "objects_deleted": deleted,
            "checkpoints_kept": len(steps) - len(doomed)}
