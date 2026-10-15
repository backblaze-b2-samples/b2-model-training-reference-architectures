"""mlflow-b2: artifact-uri -> train -> promote; handle-event / serve-events register promoted models."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError
from mlflow.exceptions import MlflowException

from . import events, registry, schemas, server, storage, train
from .config import ConfigError, load

DEFAULT_DATA = Path(__file__).resolve().parents[4] / "testdata" / "synthetic-shapes"


def _safe(msg: str) -> str:
    return re.sub(r"https?://\S+", "<url>", msg)


def _require_secret(cfg) -> None:
    if not cfg.event_secret:
        raise ConfigError("missing required env var: B2_EVENT_SIGNING_SECRET")


def cmd_artifact_uri(a, cfg) -> int:
    print(cfg.artifact_root)
    return 0


def cmd_train(a, cfg) -> int:
    registry.configure(cfg)
    run_id, version = train.run(cfg, Path(a.data), experiment=a.experiment, epochs=a.epochs,
                                lr=a.lr, seed=a.seed, register_as=a.register_as,
                                schema_dir=a.schema_dir)
    print(f"run_id={run_id}")
    if version:
        print(f"registered {a.register_as} version {version}")
    return 0


def cmd_promote(a, cfg) -> int:
    registry.promote(registry.configure(cfg), a.model, a.version, a.alias)
    print(f"{a.model}@{a.alias} -> version {a.version}")
    return 0


def cmd_handle_event(a, cfg) -> int:
    """Verify before parsing or touching B2/MLflow; a bad signature changes nothing."""
    _require_secret(cfg)
    body = Path(a.payload).read_bytes()
    if not events.verify_signature(cfg.event_secret, body, a.signature):
        print("error: event signature does not match", file=sys.stderr)
        return 2
    try:
        payload = json.loads(body)
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        print("error: payload is not a JSON object", file=sys.stderr)
        return 2
    return registry.handle_payload(cfg, registry.configure(cfg), storage.make_client(cfg), payload,
                                   a.schema_dir)


def cmd_serve_events(a, cfg) -> int:
    _require_secret(cfg)
    client, s3 = registry.configure(cfg), storage.make_client(cfg)
    srv = server.make_server(
        cfg.event_secret, lambda p: registry.handle_payload(cfg, client, s3, p, a.schema_dir),
        a.host, a.port)
    print(f"listening on http://{a.host}:{srv.server_port}/ (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlflow-b2")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("artifact-uri").set_defaults(fn=cmd_artifact_uri)
    t = sub.add_parser("train"); t.set_defaults(fn=cmd_train)
    t.add_argument("--data", default=str(DEFAULT_DATA), help="directory holding a shard manifest.json")
    t.add_argument("--experiment", default="synthetic-shapes")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--register-as", default="shapes-classifier")
    pr = sub.add_parser("promote"); pr.set_defaults(fn=cmd_promote)
    pr.add_argument("--model", required=True); pr.add_argument("--version", required=True)
    pr.add_argument("--alias", default="champion")
    h = sub.add_parser("handle-event"); h.set_defaults(fn=cmd_handle_event)
    h.add_argument("--payload", required=True, help="raw webhook body as received")
    h.add_argument("--signature", required=True, help=f"{events.SIGNATURE_HEADER} header value")
    s = sub.add_parser("serve-events"); s.set_defaults(fn=cmd_serve_events)
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8080)
    for sp in (t, h, s):
        sp.add_argument("--schema-dir", default=str(schemas.SCHEMA_DIR))
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    try:
        return a.fn(a, load())
    except train.DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (ConfigError, storage.StorageError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ClientError as e:  # code only; never echo request details
        print(f"error: B2 request failed: {e.response.get('Error', {}).get('Code')}", file=sys.stderr)
        return 1
    except (MlflowException, S3UploadFailedError) as e:
        print(f"error: {_safe(str(e))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
