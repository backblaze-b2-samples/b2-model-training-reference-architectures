import dataclasses

import pytest
from conftest import BUCKET, get_json

from b2_train import checkpoint, cli, layout

TRAIN = ["train", "--run-id", "r1", "--checkpoint-every", "3"]
PROMOTE = ["promote", "--run-id", "r1", "--model", "shapes-cnn", "--model-version", "v1"]
UNLOCKED = PROMOTE + ["--no-lock"]
MODEL = layout.model_prefix("shapes-cnn", "v1")


def test_promote_latest_copies_and_writes_schema_valid_manifest(published, s3, run, calls, validate):
    assert run(TRAIN) == 0
    calls.clear()
    assert run(UNLOCKED) == 0
    m = validate("model-manifest", get_json(s3, MODEL + layout.MANIFEST_NAME))
    ck = get_json(s3, layout.checkpoint_prefix("r1", 10) + layout.MANIFEST_NAME)
    assert m["source"] == {"run_id": "r1", "step": 10, "checkpoint_fingerprint": ck["fingerprint"]}
    assert m["files"] == ck["files"] and m["retention_mode"] == "none" and m["dataset"] == ck["dataset"]
    for f in m["files"]:
        assert s3.head_object(Bucket=BUCKET, Key=MODEL + f["path"])["Metadata"]["sha256"] == f["sha256"]
    ops = [op for op, _ in calls]
    assert "CopyObject" in ops and ops.count("PutObject") == 1  # only the manifest is uploaded
    assert [k for op, k in calls if op in ("CopyObject", "PutObject")][-1] == MODEL + layout.MANIFEST_NAME


def test_promote_rerun_is_idempotent_and_other_content_refused(published, s3, run, capsys):
    assert run(TRAIN) == 0
    assert run(UNLOCKED + ["--step", "6"]) == 0
    capsys.readouterr()
    assert run(UNLOCKED + ["--step", "6"]) == 0
    assert capsys.readouterr().out.startswith("skipped")
    assert run(UNLOCKED + ["--step", "9"]) == 1
    assert "already exists with different content" in capsys.readouterr().err
    assert get_json(s3, MODEL + layout.MANIFEST_NAME)["source"]["step"] == 6


def test_promote_refuses_when_destination_file_differs(published, s3, run, capsys):
    assert run(TRAIN + ["--max-steps", "3"]) == 0
    s3.put_object(Bucket=BUCKET, Key=MODEL + ".metadata", Body=b"x", Metadata={"sha256": "1" * 64})
    assert run(UNLOCKED) == 1
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=MODEL)["Contents"]}
    assert keys == {MODEL + ".metadata"}


def test_promote_without_checkpoint_or_bad_name(s3, run):
    assert run(UNLOCKED) == 1
    assert run(["promote", "--run-id", "r1", "--model", "Shapes", "--model-version", "v1", "--no-lock"]) == 2


def test_promote_with_lock_days_sets_governance(s3, cfg, shapes, monkeypatch, validate, tmp_path):
    monkeypatch.chdir(tmp_path)
    s3.create_bucket(Bucket="locked-bucket", ObjectLockEnabledForBucket=True)
    # moto wants an integrity header even on body-less CopyObject with lock headers; S3 does not.
    s3.meta.events.register("before-sign.s3.CopyObject", lambda request, **kw:
                            request.headers.__setitem__("x-amz-sdk-checksum-algorithm", "SHA256"))
    locked = dataclasses.replace(cfg, bucket="locked-bucket")
    monkeypatch.setattr(cli, "load", lambda: locked)
    assert cli.main(["publish-dataset", "--src", str(shapes)]) == 0
    assert cli.main(TRAIN + ["--max-steps", "3"]) == 0
    assert cli.main(PROMOTE + ["--lock-days", "1"]) == 0
    m = validate("model-manifest", get_json(s3, MODEL + layout.MANIFEST_NAME, "locked-bucket"))
    assert m["retention_mode"] == "governance"
    for key in [MODEL + layout.MANIFEST_NAME] + [MODEL + f["path"] for f in m["files"]]:
        assert s3.head_object(Bucket="locked-bucket", Key=key)["ObjectLockMode"] == "GOVERNANCE"


def test_prune_keeps_newest_complete_and_newer_incomplete(published, s3, run, capsys):
    assert run(TRAIN) == 0  # complete: 3, 6, 9, 10
    for step in (2, 11):
        s3.put_object(Bucket=BUCKET, Key=layout.checkpoint_prefix("r1", step) + "__0_0.distcp",
                      Body=b"partial")
    s3.put_object(Bucket=BUCKET, Key="checkpoints/r2/step-00000001/manifest.json", Body=b"{}")
    capsys.readouterr()
    assert run(["prune", "--run-id", "r1", "--keep", "2"]) == 0
    out = capsys.readouterr().out
    assert "'checkpoints_deleted': 3" in out and "'checkpoints_kept': 3" in out
    steps = checkpoint.scan(s3, BUCKET, "r1")
    assert sorted(steps) == [9, 10, 11] and not steps[11]["complete"]
    assert checkpoint.scan(s3, BUCKET, "r2")
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix="datasets/")["KeyCount"] == 6
    assert run(["prune", "--run-id", "r1", "--keep", "0"]) == 2


@pytest.mark.parametrize("extra", [[], ["--lock-days", "0"], ["--lock-days", "1", "--no-lock"]])
def test_promote_requires_explicit_retention_choice(run, extra, capsys):
    with pytest.raises(SystemExit) as exc:
        run(PROMOTE + extra)
    assert exc.value.code == 2
