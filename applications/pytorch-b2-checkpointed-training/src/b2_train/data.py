"""Stream WebDataset shards from B2 into a torch IterableDataset, split by rank and worker."""
from __future__ import annotations

import hashlib
import io
import os
import random
import tarfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .storage import StorageError


def assign(files: list, rank: int, world_size: int, worker: int, workers: int, seed: int) -> list:
    """Shuffle shard order identically on every rank, then take every Nth shard for this slot."""
    order = list(files)
    random.Random(seed).shuffle(order)
    return order[rank * workers + worker::world_size * workers]


class _Verified(io.RawIOBase):
    """Hash bytes as tarfile reads them; optionally tee them into a cache file."""

    def __init__(self, body, sink=None):
        self.body, self.sink, self.h, self.n = body, sink, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.body.read(len(b))
        self.h.update(data)
        self.n += len(data)
        if self.sink:
            self.sink.write(data)
        b[:len(data)] = data
        return len(data)

    def finish(self, f: dict) -> None:
        while self.readinto(bytearray(1 << 16)):  # tar end padding the reader stopped before
            pass
        if self.n != f["size"] or self.h.hexdigest() != f["sha256"]:
            raise StorageError(f"{f['path']}: streamed content differs from manifest")


def decode(name: str, data: bytes, keys: dict):
    if name == keys["image"]:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        pixels = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(h, w, 3)
        return pixels.permute(2, 0, 1).float() / 255
    if name == keys["label"]:
        return int(data.decode().strip())
    return None


class ShardStream(IterableDataset):
    """Each shard is one GET, read as a tar stream ("r|"): no full download, no seeking."""

    def __init__(self, client_factory, bucket: str, prefix: str, files: list[dict], sample_keys: dict,
                 *, cache_dir: str | None = None, seed: int = 0, rank=None, world_size=None):
        self.client_factory, self.bucket, self.prefix = client_factory, bucket, prefix
        self.files, self.keys, self.seed, self.epoch = files, sample_keys, seed, 0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rank = int(os.environ.get("RANK", 0)) if rank is None else rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else world_size
        self._client = None

    def __getstate__(self):  # boto3 clients do not pickle; each worker builds its own
        return {**self.__dict__, "_client": None}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def my_shards(self) -> list[dict]:
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        return assign(self.files, self.rank, self.world_size, worker, workers, self.seed + self.epoch)

    def __iter__(self):
        for f in self.my_shards():
            yield from self._samples(f)

    def _samples(self, f: dict):
        cached = self.cache_dir / f"{f['sha256']}.tar" if self.cache_dir else None
        if cached and cached.exists():  # verified when written
            with cached.open("rb") as fh:
                yield from self._read(fh)
            return
        if self._client is None:
            self._client = self.client_factory()
        body = self._client.get_object(Bucket=self.bucket, Key=self.prefix + f["path"])["Body"]
        tmp = cached.with_suffix(f".{os.getpid()}.part") if cached else None
        if tmp:
            tmp.parent.mkdir(parents=True, exist_ok=True)
        sink = tmp.open("wb") if tmp else None
        try:
            stream = _Verified(body, sink)
            yield from self._read(io.BufferedReader(stream, 1 << 16))
            stream.finish(f)
            if sink:
                sink.close()
                tmp.replace(cached)
        finally:
            body.close()
            if sink and not sink.closed:
                sink.close()
            if tmp and tmp.exists():
                tmp.unlink()

    def _read(self, fh):
        sample, current = {}, None
        with tarfile.open(fileobj=fh, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key, _, ext = member.name.rpartition("/")[2].partition(".")
                if key != current and sample:
                    yield self._finish(current, sample)
                    sample = {}
                current = key
                value = decode(ext, tar.extractfile(member).read(), self.keys)
                if value is not None:
                    sample[ext] = value
        if sample:
            yield self._finish(current, sample)

    def _finish(self, key: str, sample: dict) -> dict:
        try:
            return {"__key__": key, "image": sample[self.keys["image"]], "label": sample[self.keys["label"]]}
        except KeyError as e:
            raise StorageError(f"sample {key} is missing {e.args[0]}") from e
