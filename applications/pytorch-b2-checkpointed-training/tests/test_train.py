import json

import torch
from conftest import BUCKET, get_json

from b2_train import checkpoint, layout, train

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
