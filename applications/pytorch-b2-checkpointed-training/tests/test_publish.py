import json

from conftest import BUCKET, get_json

from b2_train import layout

BASE = layout.dataset_prefix("synthetic-shapes", "v1")
ARGS = ["publish-dataset", "--src"]


def test_publish_manifest_last_and_schema_valid(s3, run, shapes, calls, validate):
    assert run(ARGS + [str(shapes)]) == 0
    writes = [k for op, k in calls if op in ("PutObject", "CreateMultipartUpload")]
    assert len(writes) == 6 and writes[-1] == BASE + "manifest.json"
    m = validate("shard-manifest", get_json(s3, BASE + "manifest.json"))
    head = s3.head_object(Bucket=BUCKET, Key=BASE + m["files"][0]["path"])
    assert head["Metadata"]["sha256"] == m["files"][0]["sha256"]


def test_publish_rerun_is_idempotent(s3, run, shapes, capsys):
    s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
    assert run(ARGS + [str(shapes)]) == 0
    capsys.readouterr()
    assert run(ARGS + [str(shapes)]) == 0
    assert "'uploaded': 0" in capsys.readouterr().out
    assert len(s3.list_object_versions(Bucket=BUCKET)["Versions"]) == 6


def test_publish_collision_on_different_remote_content(s3, run, shapes, capsys):
    s3.put_object(Bucket=BUCKET, Key=BASE + "shards/train-000000.tar", Body=b"other",
                  Metadata={"sha256": "0" * 64})
    assert run(ARGS + [str(shapes)]) == 1
    assert "already exists with different content" in capsys.readouterr().err
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert BASE + "manifest.json" not in keys


def test_publish_refuses_local_corruption(s3, run, shapes, capsys):
    shard = shapes / "shards" / "train-000002.tar"
    data = bytearray(shard.read_bytes()); data[600] ^= 0xFF
    shard.write_bytes(bytes(data))
    assert run(ARGS + [str(shapes)]) == 2
    assert "train-000002.tar: size or sha256 differs" in capsys.readouterr().err
    assert "Contents" not in s3.list_objects_v2(Bucket=BUCKET)


def test_publish_refuses_fingerprint_drift(s3, run, shapes):
    p = shapes / "manifest.json"
    m = json.loads(p.read_text()); m["fingerprint"] = "f" * 64
    p.write_text(json.dumps(m))
    assert run(ARGS + [str(shapes)]) == 2
