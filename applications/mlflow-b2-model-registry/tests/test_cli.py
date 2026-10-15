from conftest import DATA, SECRET, event_body, model_manifest, publish, sign

from mlflow_b2 import cli, registry, storage


def test_end_to_end(s3, run, cfg, tmp_path, capsys):
    assert run(["artifact-uri"]) == 0
    assert capsys.readouterr().out.strip() == \
        "b2://test-mlflow-b2-bucket@s3.us-east-1.backblazeb2.com/mlflow"
    assert run(["train", "--data", str(DATA), "--epochs", "2"]) == 0
    out = capsys.readouterr().out
    assert "run_id=" in out and "registered shapes-classifier version 1" in out
    assert run(["promote", "--model", "shapes-classifier", "--version", "1"]) == 0
    client = registry.configure(cfg)
    assert str(client.get_model_version_by_alias("shapes-classifier", "champion").version) == "1"

    body = event_body(publish(s3, model_manifest()))
    payload = tmp_path / "event.json"
    payload.write_bytes(body)
    assert run(["handle-event", "--payload", str(payload), "--signature", sign(body)]) == 0
    assert run(["handle-event", "--payload", str(payload), "--signature", sign(body)]) == 0
    assert run(["promote", "--model", "shapes-classifier", "--version", "2"]) == 0
    mv = client.get_model_version_by_alias("shapes-classifier", "champion")
    assert str(mv.version) == "2" and mv.source.endswith("/models/shapes-classifier/v2/")
    printed = capsys.readouterr()
    assert SECRET not in printed.out + printed.err and "https://" not in printed.out + printed.err


def test_bad_signature_touches_nothing(run, tmp_path, monkeypatch, capsys):
    def boom(*_):
        raise AssertionError("must not touch B2 or MLflow")
    monkeypatch.setattr(storage, "make_client", boom)
    monkeypatch.setattr(registry, "configure", boom)
    body = event_body("models/m/v1/manifest.json")
    payload = tmp_path / "event.json"
    payload.write_bytes(body)
    assert run(["handle-event", "--payload", str(payload), "--signature", sign(body, "nope")]) == 2
    assert "signature" in capsys.readouterr().err
    assert not (tmp_path / "mlflow.db").exists()


def test_errors_exit_cleanly(s3, run, tmp_path, capsys):
    assert run(["promote", "--model", "nope", "--version", "1"]) == 1
    assert "error:" in capsys.readouterr().err
    assert run(["train", "--data", str(tmp_path)]) == 2
    assert run(["train", "--data", str(DATA), "--experiment", "bad name/x"]) == 1


def test_missing_event_secret(run, cfg, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "load", lambda: cfg.__class__(**{**cfg.__dict__, "event_secret": ""}))
    (tmp_path / "e.json").write_text("{}")
    assert cli.main(["handle-event", "--payload", str(tmp_path / "e.json"), "--signature", "v1=0"]) == 1
    assert "B2_EVENT_SIGNING_SECRET" in capsys.readouterr().err


def test_missing_env_exits_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    for n in ("B2_APPLICATION_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET_NAME", "B2_REGION"):
        monkeypatch.delenv(n, raising=False)
    assert cli.main(["artifact-uri"]) == 1
    assert "missing required env vars" in capsys.readouterr().err
