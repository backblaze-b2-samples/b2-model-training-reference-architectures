from conftest import BUCKET, SECRET

from b2_train import cli


def test_end_to_end_prints_no_secrets(s3, run, shapes, tmp_path, capsys):
    cache = str(tmp_path / "cache")
    assert run(["publish-dataset", "--src", str(shapes)]) == 0
    assert run(["train", "--run-id", "e2e", "--checkpoint-every", "4", "--cache-dir", cache]) == 0
    assert run(["promote", "--run-id", "e2e", "--model", "shapes-cnn", "--model-version", "v1"]) == 0
    assert run(["prune", "--run-id", "e2e", "--keep", "1"]) == 0
    out = capsys.readouterr()
    text = out.out + out.err
    assert SECRET not in text and "keyid123" not in text and "https://" not in text
    assert "published synthetic-shapes@v1" in text and "promoted shapes-cnn@v1" in text
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert {k.split("/")[0] for k in keys} == {"datasets", "checkpoints", "models"}
    assert {k for k in keys if k.startswith("checkpoints/")} == {
        "checkpoints/e2e/step-00000010/" + n for n in (".metadata", "__0_0.distcp", "manifest.json")}
    assert not (tmp_path / "staging" / "e2e").exists() or not any((tmp_path / "staging" / "e2e").iterdir())


def test_missing_env_exits_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    for n in ("B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET_NAME", "B2_REGION"):
        monkeypatch.delenv(n, raising=False)
    assert cli.main(["prune", "--run-id", "a", "--keep", "1"]) == 1
    assert "missing required env vars" in capsys.readouterr().err
