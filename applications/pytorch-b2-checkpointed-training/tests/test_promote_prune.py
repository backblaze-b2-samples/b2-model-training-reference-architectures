import dataclasses

import pytest
from botocore.exceptions import ClientError
from conftest import BUCKET, get_json

from b2_train import checkpoint, cli, layout, storage

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


LOCKED = "locked-bucket"


@pytest.fixture
def locked_run(s3, cfg, shapes, monkeypatch, tmp_path):
    """CLI against an Object-Lock bucket with a 3-step checkpoint committed."""
    monkeypatch.chdir(tmp_path)
    s3.create_bucket(Bucket=LOCKED, ObjectLockEnabledForBucket=True)
    # moto wants an integrity header even on body-less CopyObject with lock headers; S3 does not.
    s3.meta.events.register("before-sign.s3.CopyObject", lambda request, **kw:
                            request.headers.__setitem__("x-amz-sdk-checksum-algorithm", "SHA256"))
    monkeypatch.setattr(cli, "load", lambda: dataclasses.replace(cfg, bucket=LOCKED))
    assert cli.main(["publish-dataset", "--src", str(shapes)]) == 0
    assert cli.main(TRAIN + ["--max-steps", "3"]) == 0
    return cli.main


def _model_keys(m):
    return [MODEL + layout.MANIFEST_NAME] + [MODEL + f["path"] for f in m["files"]]


def test_promote_with_lock_days_sets_governance(s3, locked_run, validate):
    assert locked_run(PROMOTE + ["--lock-days", "1"]) == 0
    m = validate("model-manifest", get_json(s3, MODEL + layout.MANIFEST_NAME, LOCKED))
    assert m["retention_mode"] == "governance"
    for key in _model_keys(m):
        assert s3.head_object(Bucket=LOCKED, Key=key)["ObjectLockMode"] == "GOVERNANCE"


def test_locked_retry_locks_files_left_by_an_unlocked_attempt(s3, locked_run):
    """An interrupted unlocked promotion must not leave unlocked payload under a governance manifest."""
    ck = get_json(s3, layout.checkpoint_prefix("r1", 3) + layout.MANIFEST_NAME, LOCKED)
    for f in ck["files"]:  # copied without retention, manifest never written
        s3.copy_object(Bucket=LOCKED, Key=MODEL + f["path"], MetadataDirective="REPLACE",
                       Metadata={"sha256": f["sha256"]},
                       CopySource={"Bucket": LOCKED, "Key": layout.checkpoint_prefix("r1", 3) + f["path"]})
        assert "ObjectLockMode" not in s3.head_object(Bucket=LOCKED, Key=MODEL + f["path"])
    assert locked_run(PROMOTE + ["--lock-days", "365"]) == 0
    m = get_json(s3, MODEL + layout.MANIFEST_NAME, LOCKED)
    for key in _model_keys(m):
        assert s3.head_object(Bucket=LOCKED, Key=key)["ObjectLockMode"] == "GOVERNANCE", key


def test_unlocked_retry_refuses_files_already_under_retention(s3, locked_run, capsys):
    ck = get_json(s3, layout.checkpoint_prefix("r1", 3) + layout.MANIFEST_NAME, LOCKED)
    f = ck["files"][0]
    body = s3.get_object(Bucket=LOCKED, Key=layout.checkpoint_prefix("r1", 3) + f["path"])["Body"].read()
    s3.put_object(Bucket=LOCKED, Key=MODEL + f["path"], Body=body, Metadata={"sha256": f["sha256"]},
                  ObjectLockMode="GOVERNANCE", ObjectLockRetainUntilDate=storage.lock_until(1),
                  ChecksumAlgorithm="SHA256")
    assert locked_run(UNLOCKED) == 1
    assert "already under retention" in capsys.readouterr().err
    assert "Contents" not in s3.list_objects_v2(Bucket=LOCKED, Prefix=MODEL + layout.MANIFEST_NAME)


@pytest.mark.parametrize("forge_metadata", [False, True])
def test_promote_refuses_corrupted_source_checkpoint(published, s3, run, capsys, forge_metadata):
    """Same-size different bytes: never copied, and never labelled with the expected checksum."""
    assert run(TRAIN + ["--max-steps", "3"]) == 0
    ck = get_json(s3, layout.checkpoint_prefix("r1", 3) + layout.MANIFEST_NAME)
    f = next(f for f in ck["files"] if f["path"] != ".metadata")
    key = layout.checkpoint_prefix("r1", 3) + f["path"]
    corrupted = bytes(b ^ 0xFF for b in s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    s3.put_object(Bucket=BUCKET, Key=key, Body=corrupted,
                  Metadata={"sha256": f["sha256"]} if forge_metadata else {})
    assert run(UNLOCKED) == 1
    assert "differs from manifest" in capsys.readouterr().err
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=MODEL)["KeyCount"] == 0


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


def _hide_lock_headers(s3, monkeypatch, retention_api_denied: bool):
    """What B2 returns to a key without readFileRetentions: HEAD carries no Object Lock fields."""
    real_head = s3.head_object

    def head(**kw):
        h = real_head(**kw)
        return {k: v for k, v in h.items() if not k.startswith("ObjectLock")}

    def get_retention(**kw):  # stands in for moto's GetObjectRetention, which returns 500 in moto 5.2
        if retention_api_denied:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObjectRetention")
        h = real_head(**kw)
        return {"Retention": {"Mode": h["ObjectLockMode"], "RetainUntilDate": h["ObjectLockRetainUntilDate"]}}
    monkeypatch.setattr(s3, "head_object", head)
    monkeypatch.setattr(s3, "get_object_retention", get_retention)


def test_missing_read_retention_capability_is_reported_as_such(s3, locked_run, monkeypatch, capsys):
    _hide_lock_headers(s3, monkeypatch, retention_api_denied=True)
    assert locked_run(PROMOTE + ["--lock-days", "1"]) == 1
    err = capsys.readouterr().err
    assert "needs readFileRetentions" in err and "retention is none" not in err


def test_retention_api_is_used_when_head_omits_lock_fields(s3, locked_run, monkeypatch):
    _hide_lock_headers(s3, monkeypatch, retention_api_denied=False)
    assert locked_run(PROMOTE + ["--lock-days", "1"]) == 0
