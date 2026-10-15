import json

import pytest
import torch
from conftest import BUCKET, ROOT, get_json
from jsonschema import Draft202012Validator

from b2_train import checkpoint, layout, manifest, storage, train

TRAIN = ["train", "--run-id", "r1", "--checkpoint-every", "3"]


def _manifest(s3, run, step):
    return get_json(s3, layout.checkpoint_prefix(run, step) + layout.MANIFEST_NAME)


def _weights(s3, run, step, tmp_path):
    model = train.TinyCNN(4)
    opt = torch.optim.Adam(model.parameters())
    ck = checkpoint.fetch(s3, BUCKET, run, step, tmp_path / f"{run}-{step}")
    checkpoint.load_local(model, opt, tmp_path / f"{run}-{step}", ck["format"])
    return model.state_dict()


def test_train_writes_schema_valid_checkpoints_manifest_last(published, s3, run, calls, validate):
    assert run(TRAIN) == 0
    assert sorted(checkpoint.scan(s3, BUCKET, "r1")) == [3, 6, 9, 10]  # 80/16 = 5 steps x 2 epochs
    prefix = layout.checkpoint_prefix("r1", 10)
    writes = [k for op, k in calls if op == "PutObject" and k.startswith(prefix)]
    assert writes[-1] == prefix + "manifest.json" and len(writes) == 3
    m = validate("checkpoint-manifest", _manifest(s3, "r1", 10))
    assert m["format"] == "torch.distributed.checkpoint" and m["epoch"] == 1 and m["world_size"] == 1
    assert m["dataset"] == {"id": "synthetic-shapes@v1", "fingerprint": published["fingerprint"]}
    assert {f["path"] for f in m["files"]} == {".metadata", "__0_0.distcp"}
    assert {"train_loss", "val_loss", "val_accuracy"} <= m["metrics"].keys()


def test_interrupted_run_resumes_from_latest_complete_checkpoint(published, s3, run, capsys, tmp_path):
    assert run(TRAIN + ["--max-steps", "7"]) == 0
    assert sorted(checkpoint.scan(s3, BUCKET, "r1")) == [3, 6]
    capsys.readouterr()
    assert run(TRAIN) == 0
    assert "resumed r1 from step 6" in capsys.readouterr().out
    assert sorted(checkpoint.scan(s3, BUCKET, "r1")) == [3, 6, 9, 10]
    assert _manifest(s3, "r1", 9)["epoch"] == 1
    # The resumed run ends bit-identical to one that was never interrupted.
    assert run(["train", "--run-id", "ref", "--checkpoint-every", "3"]) == 0
    a, b = _weights(s3, "r1", 10, tmp_path), _weights(s3, "ref", 10, tmp_path)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_incomplete_checkpoint_is_ignored(published, s3, run, capsys):
    assert run(TRAIN + ["--max-steps", "3"]) == 0
    s3.put_object(Bucket=BUCKET, Key=layout.checkpoint_prefix("r1", 8) + "__0_0.distcp", Body=b"torn")
    capsys.readouterr()
    assert run(TRAIN + ["--max-steps", "4"]) == 0
    assert "resumed r1 from step 3" in capsys.readouterr().out
    assert checkpoint.latest_complete(s3, BUCKET, "r1") == 3


def test_resume_complete_run_is_noop(published, s3, run, capsys):
    assert run(TRAIN) == 0
    capsys.readouterr()
    assert run(TRAIN) == 0
    out = capsys.readouterr().out
    assert "resumed r1 from step 10" in out and "committed" not in out


def test_no_resume_refuses_to_overwrite_committed_step(published, s3, run, capsys):
    assert run(TRAIN + ["--max-steps", "3"]) == 0
    assert run(TRAIN + ["--max-steps", "3", "--no-resume"]) == 1
    assert "already committed" in capsys.readouterr().err


def test_fingerprint_mismatch_refused(published, s3, run, capsys):
    assert run(TRAIN + ["--expect-fingerprint", "0" * 64]) == 2
    assert "refused" in capsys.readouterr().err
    assert checkpoint.scan(s3, BUCKET, "r1") == {}
    assert run(TRAIN + ["--max-steps", "1", "--expect-fingerprint", published["fingerprint"]]) == 0


def test_resume_refuses_checkpoint_from_other_dataset(published, s3, run, capsys):
    assert run(TRAIN + ["--max-steps", "3"]) == 0
    key = layout.checkpoint_prefix("r1", 3) + layout.MANIFEST_NAME
    m = get_json(s3, key); m["dataset"]["fingerprint"] = "a" * 64
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(m).encode())
    assert run(TRAIN) == 2
    assert "different fingerprint" in capsys.readouterr().err


def test_corrupt_checkpoint_file_fails_resume(published, s3, run, capsys):
    assert run(TRAIN + ["--max-steps", "3"]) == 0
    s3.put_object(Bucket=BUCKET, Key=layout.checkpoint_prefix("r1", 3) + "__0_0.distcp", Body=b"bitrot")
    assert run(TRAIN) == 1
    assert "downloaded content differs" in capsys.readouterr().err


def test_train_needs_published_dataset_and_single_process(s3, run, capsys, monkeypatch):
    assert run(TRAIN) == 1
    assert "dataset synthetic-shapes/v1 not found" in capsys.readouterr().err
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert run(TRAIN) == 2
    assert run(["train", "--run-id", "Bad/Run"]) == 2


def test_train_refuses_manifest_whose_fingerprint_does_not_match_shards(published, s3, run, capsys):
    """Changed shards under an unchanged top-level fingerprint must not train under the original pin."""
    key = layout.dataset_prefix("synthetic-shapes", "v1") + layout.MANIFEST_NAME
    m = get_json(s3, key)
    m["files"][0]["sha256"] = "0" * 64  # content changed, pin left alone
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(m).encode())
    assert run(TRAIN + ["--expect-fingerprint", published["fingerprint"]]) == 2
    assert "fingerprint does not match" in capsys.readouterr().err
    assert checkpoint.scan(s3, BUCKET, "r1") == {}


def test_checkpoint_paths_cannot_escape_restore_directory(published, s3, run, tmp_path):
    assert run(TRAIN + ["--max-steps", "3"]) == 0
    key = layout.checkpoint_prefix("r1", 3) + layout.MANIFEST_NAME
    m = get_json(s3, key)
    m["files"].append({"path": "../escaped.txt", "size": 1, "sha256": "1" * 64})
    m["fingerprint"] = manifest.fingerprint(m["files"])
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(m).encode())
    restore = tmp_path / "scratch" / "restore"
    with pytest.raises(storage.StorageError, match="unsafe path"):
        checkpoint.fetch(s3, BUCKET, "r1", 3, restore)
    assert not (restore.parent / "escaped.txt").exists()


PATH_RULES = [("shard-manifest", ("$defs", "path")),
              ("checkpoint-manifest", ("$defs", "files", "items", "properties", "path")),
              ("model-manifest", ("$defs", "files", "items", "properties", "path"))]


@pytest.mark.parametrize("path", ["../x", "a/../b", "./x", "/x", "a//b", "a/..", "a/./b", ".."])
def test_unsafe_paths_rejected_by_schemas_and_code(path):
    for name, where in PATH_RULES:
        rule = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text())
        for part in where:
            rule = rule[part]
        assert not Draft202012Validator(rule).is_valid(path), (name, path)
    assert not manifest.safe_path(path)


@pytest.mark.parametrize("path", ["shards/train-000000.tar", ".metadata", "__0_0.distcp", "a/b.c"])
def test_ordinary_paths_still_valid(path):
    for name, where in PATH_RULES:
        rule = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text())
        for part in where:
            rule = rule[part]
        assert Draft202012Validator(rule).is_valid(path), (name, path)
    assert manifest.safe_path(path)
