import itertools
from types import SimpleNamespace

import pytest
from conftest import BUCKET

from b2_train import data, layout, storage, train


def _stream(s3, ds, split="train", **kw):
    return train._split(lambda: s3, BUCKET, ds, split, **kw)


def _gets(calls):
    return [k for op, k in calls if op == "GetObject"]


def test_stream_yields_every_train_sample_once(published, s3, calls):
    samples = list(_stream(s3, published))
    keys = [s["__key__"] for s in samples]
    assert len(keys) == 80 and set(keys) == {f"train-{i:05d}" for i in range(80)}
    assert samples[0]["image"].shape == (3, 32, 32) and 0.0 <= samples[0]["image"].max() <= 1.0
    assert {s["label"] for s in samples} == {0, 1, 2, 3}
    assert len(_gets(calls)) == 4  # one streaming GET per shard, nothing else


def test_assign_is_disjoint_and_complete():
    files = list(range(13))
    for world, workers in itertools.product((1, 2, 3), (1, 2, 4)):
        parts = [data.assign(files, r, world, w, workers, seed=7)
                 for r in range(world) for w in range(workers)]
        flat = [f for p in parts for f in p]
        assert sorted(flat) == files and len(flat) == len(set(flat))


def test_rank_and_worker_splits_cover_dataset(published, s3, monkeypatch):
    seen = []
    for rank, worker in itertools.product(range(2), range(2)):
        monkeypatch.setattr(data, "get_worker_info", lambda w=worker: SimpleNamespace(id=w, num_workers=2))
        seen.append({s["__key__"] for s in _stream(s3, published, rank=rank, world_size=2)})
    assert all(len(part) == 20 for part in seen)  # 4 shards over 4 slots
    assert set().union(*seen) == {f"train-{i:05d}" for i in range(80)}


def test_rank_from_env(published, s3, monkeypatch):
    monkeypatch.setenv("RANK", "1"); monkeypatch.setenv("WORLD_SIZE", "2")
    st = _stream(s3, published)
    assert (st.rank, st.world_size) == (1, 2) and len(list(st)) == 40


def test_cache_avoids_second_get(published, s3, calls, tmp_path):
    cache = tmp_path / "cache"
    st = _stream(s3, published, cache_dir=str(cache))
    first = list(st)
    st.set_epoch(1)
    second = list(st)
    assert len(_gets(calls)) == 4 and len(first) == len(second) == 80
    assert sorted(p.name for p in cache.iterdir()) == sorted(
        f"{f['sha256']}.tar" for f in published["files"] if f["path"].startswith("shards/train"))


def test_tampered_remote_shard_fails_stream_and_is_not_cached(published, s3, tmp_path):
    key = layout.dataset_prefix("synthetic-shapes", "v1") + "shards/train-000001.tar"
    body = bytearray(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()); body[-1] ^= 0xFF
    s3.put_object(Bucket=BUCKET, Key=key, Body=bytes(body))
    files = [f for f in published["files"] if f["path"] == "shards/train-000001.tar"]
    st = data.ShardStream(lambda: s3, BUCKET, layout.dataset_prefix("synthetic-shapes", "v1"), files,
                          published["sample_keys"], cache_dir=str(tmp_path / "c"))
    with pytest.raises(storage.StorageError, match="streamed content differs"):
        list(st)
    assert not any((tmp_path / "c").iterdir())
