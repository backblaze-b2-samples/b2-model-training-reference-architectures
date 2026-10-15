"""Tiny CPU training loop with periodic checkpoints to B2 and resume from the latest complete one."""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import checkpoint, layout
from .data import ShardStream


class DatasetMismatch(Exception):
    pass


class TinyCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 8 * 8, classes),
        )

    def forward(self, x):
        return self.net(x)


def _split(client_factory, bucket, ds: dict, split: str, **kw) -> ShardStream:
    files = {f["path"]: f for f in ds["files"]}
    return ShardStream(client_factory, bucket, layout.dataset_prefix(ds["dataset"], ds["version"]),
                       [files[p] for p in ds["splits"][split]["shards"]], ds["sample_keys"], **kw)


@torch.no_grad()
def evaluate(model, loader) -> dict[str, float]:
    model.eval()
    loss = correct = n = 0
    for batch in loader:
        out = model(batch["image"])
        loss += nn.functional.cross_entropy(out, batch["label"], reduction="sum").item()
        correct += (out.argmax(1) == batch["label"]).sum().item()
        n += len(batch["label"])
    model.train()
    return {"val_loss": loss / max(n, 1), "val_accuracy": correct / max(n, 1)}


def fit(client, client_factory, bucket: str, ds: dict, *, run_id: str, epochs=2, batch_size=16, lr=1e-2,
        checkpoint_every=0, max_steps=0, cache_dir=None, staging_dir="staging", resume=True, seed=0,
        workers=0, log=print) -> dict:
    torch.manual_seed(seed)
    model = TinyCNN(len(ds["classes"]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pin = {"id": ds["id"], "fingerprint": ds["fingerprint"]}
    staging = Path(staging_dir) / run_id
    step = epoch = skip = 0
    resumed = checkpoint.latest_complete(client, bucket, run_id) if resume else None
    if resumed is not None:
        ck = checkpoint.fetch(client, bucket, run_id, resumed, staging / "restore")
        if ck["dataset"] != pin:
            raise DatasetMismatch(f"checkpoint step {resumed} was trained on {ck['dataset']['id']} "
                                  "with a different fingerprint; start a new run")
        step, epoch, skip = checkpoint.load_local(model, opt, staging / "restore", ck["format"]).tolist()
        shutil.rmtree(staging / "restore", ignore_errors=True)
        log(f"resumed {run_id} from step {step} (epoch {epoch})")
    saved, last, losses = step, None, []
    kw = dict(cache_dir=cache_dir, seed=seed)
    train = _split(client_factory, bucket, ds, "train", **kw)
    val = DataLoader(_split(client_factory, bucket, ds, "val", **kw), batch_size=batch_size)
    loader = DataLoader(train, batch_size=batch_size, num_workers=workers)

    def save(e: int, b: int) -> dict:
        """(e, b) is where training continues: epoch e after skipping b batches."""
        nonlocal saved, losses
        path = staging / f"step-{step:08d}"
        fmt = checkpoint.save_local(model, opt, torch.tensor([step, e, b]), path)
        metrics = {"train_loss": sum(losses) / max(len(losses), 1), **evaluate(model, val)}
        m = checkpoint.commit(client, bucket, run_id, step, path, epoch=min(e, epochs - 1), fmt=fmt,
                              torch_version=torch.__version__, dataset=pin, world_size=train.world_size,
                              metrics=metrics)
        shutil.rmtree(path, ignore_errors=True)
        saved, losses = step, []
        log(f"checkpoint step {step} committed fingerprint={m['fingerprint'][:12]}")
        return m

    for epoch in range(epoch, epochs):
        train.set_epoch(epoch)
        for b, batch in enumerate(loader):
            if b < skip:  # fast-forward to where the checkpoint left off; shard order is seeded per epoch
                continue
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(batch["image"]), batch["label"])
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if checkpoint_every and step % checkpoint_every == 0:
                last = save(epoch, b + 1)
            if max_steps and step >= max_steps:
                log(f"stopped at step {step} (--max-steps); resume continues from step {saved}")
                return {"step": step, "resumed_from": resumed, "interrupted": True, "checkpoint": last}
        skip = 0
    if saved != step:
        last = save(epochs, 0)
    log(f"trained {run_id} to step {step}")
    return {"step": step, "resumed_from": resumed, "interrupted": False, "checkpoint": last}

