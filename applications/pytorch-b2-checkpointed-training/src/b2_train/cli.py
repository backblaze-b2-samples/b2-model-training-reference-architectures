"""b2-train: publish-dataset -> train (checkpoint + resume) -> promote -> prune."""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

from botocore.exceptions import ClientError

from . import checkpoint, layout, manifest, promote, storage
from .config import ConfigError, load


class Refused(Exception):
    """Input or contract problem: exit 2 and change nothing."""


def _name(value: str, flag: str) -> str:
    if not manifest.NAME_RE.match(value):
        raise Refused(f"{flag} must match [a-z0-9._-]+")
    return value


def cmd_publish_dataset(a, cfg) -> int:
    """Verify local shards, upload them, verify remote, then write the shard manifest last."""
    src = Path(a.src)
    m = json.loads((src / layout.MANIFEST_NAME).read_text())
    errors = manifest.check_dataset(src, m)
    if errors:
        print("local dataset does not match its manifest:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2
    c = storage.make_client(cfg)
    storage.check_bucket(c, cfg.bucket)
    base = layout.dataset_prefix(_name(m["dataset"], "dataset"), _name(m["version"], "version"))
    items = [(base + f["path"], src / f["path"]) for f in m["files"]]
    print(storage.upload_tree(c, cfg.bucket, items, overwrite=False, multipart_mb=a.multipart_mb))
    remote_errors = storage.verify(c, cfg.bucket, base, m["files"])
    if remote_errors:
        print("remote verification failed:\n  " + "\n  ".join(remote_errors), file=sys.stderr)
        return 3
    storage.put_bytes(c, cfg.bucket, base + layout.MANIFEST_NAME,
                      (src / layout.MANIFEST_NAME).read_bytes(), overwrite=False)
    print(f"published {m['id']} fingerprint={m['fingerprint'][:12]}")
    return 0


def cmd_train(a, cfg) -> int:
    from . import train  # torch import is slow; keep it off the other subcommands

    _name(a.run_id, "--run-id")
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        raise Refused("this sample trains in one process; see infrastructure/kubernetes/README.md")
    c = storage.make_client(cfg)
    ds = storage.load_json(c, cfg.bucket, layout.dataset_prefix(a.dataset, a.version) + layout.MANIFEST_NAME,
                           f"dataset {a.dataset}/{a.version}")
    if a.expect_fingerprint and a.expect_fingerprint != ds["fingerprint"]:
        raise Refused(f"dataset {ds['id']} fingerprint {ds['fingerprint'][:12]} != --expect-fingerprint")
    print(f"dataset {ds['id']} fingerprint={ds['fingerprint'][:12]}")
    try:
        train.fit(c, functools.partial(storage.make_client, cfg), cfg.bucket, ds, run_id=a.run_id,
                  epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, checkpoint_every=a.checkpoint_every,
                  max_steps=a.max_steps, cache_dir=a.cache_dir, staging_dir=a.staging_dir,
                  resume=not a.no_resume, seed=a.seed, workers=a.workers)
    except train.DatasetMismatch as e:
        raise Refused(str(e)) from e
    return 0


def cmd_promote(a, cfg) -> int:
    c = storage.make_client(cfg)
    m, status = promote.promote(c, cfg.bucket, _name(a.run_id, "--run-id"), a.step,
                                _name(a.model, "--model"), _name(a.model_version, "--model-version"),
                                lock_days=a.lock_days)
    print(f"{status} {m['id']} from {a.run_id}/step-{m['source']['step']:08d} "
          f"fingerprint={m['fingerprint'][:12]} retention={m['retention_mode']}")
    return 0


def cmd_prune(a, cfg) -> int:
    if a.keep < 1:
        raise Refused("--keep must be at least 1")
    print(checkpoint.prune(storage.make_client(cfg), cfg.bucket, _name(a.run_id, "--run-id"), a.keep))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="b2-train")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("publish-dataset"); d.set_defaults(fn=cmd_publish_dataset)
    d.add_argument("--src", required=True, help="directory with manifest.json and shards/")
    d.add_argument("--multipart-mb", type=int, default=64)
    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--dataset", default="synthetic-shapes"); t.add_argument("--version", default="v1")
    t.add_argument("--run-id", required=True)
    t.add_argument("--epochs", type=int, default=2); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-2); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--checkpoint-every", type=int, default=0, help="steps; 0 = only at the end")
    t.add_argument("--max-steps", type=int, default=0,
                   help="stop without a final checkpoint (simulated preemption)")
    t.add_argument("--cache-dir", default=None, help="local shard cache, keyed by sha256")
    t.add_argument("--staging-dir", default="staging", help="local checkpoint staging")
    t.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    t.add_argument("--expect-fingerprint", default=None)
    t.add_argument("--no-resume", action="store_true")
    r = sub.add_parser("promote"); r.set_defaults(fn=cmd_promote)
    r.add_argument("--run-id", required=True)
    r.add_argument("--step", type=int, default=None, help="default: latest complete checkpoint")
    r.add_argument("--model", required=True); r.add_argument("--model-version", required=True)
    r.add_argument("--lock-days", type=int, default=None,
                   help="GOVERNANCE retention; bucket must have Object Lock enabled")
    k = sub.add_parser("prune"); k.set_defaults(fn=cmd_prune)
    k.add_argument("--run-id", required=True); k.add_argument("--keep", type=int, required=True)
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    try:
        return a.fn(a, load())
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except (ConfigError, storage.StorageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ClientError as e:  # code only; never echo request details
        print(f"error: B2 request failed: {e.response.get('Error', {}).get('Code')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
