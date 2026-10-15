# mlflow-b2-model-registry

CLI for the [track experiments and register models](../../architectures/track-experiments-and-register-models/architecture.md) architecture:
train a tiny classifier on the shared WebDataset shards -> log params, metrics, the dataset pin and a pyfunc model to MLflow with artifacts on B2 -> register and promote it with an alias.
A webhook receiver turns B2 event notifications for promoted-model manifests (written by the [PyTorch training app](../pytorch-b2-checkpointed-training/), see the [training run contract](../../docs/training-run-contract.md)) into MLflow model versions, which is the hand-off step in [orchestrate training pipelines](../../architectures/orchestrate-training-pipelines/architecture.md).
Uses MLflow's native `b2://` artifact repository (MLflow >= 3.11.1) and boto3 over the B2 S3-compatible API. No GPU or PyTorch needed; training takes seconds on CPU.

Capabilities demonstrated: S3 API via MLflow's `b2://` artifact store, scoped keys, per-experiment artifact locations under one prefix, dataset id/fingerprint lineage tags, schema-validated model manifests, idempotent event-driven registration, event-signature verification and a fast-ack webhook receiver.

## B2's role

Durable store for every MLflow artifact (run files, logged models, the dataset manifest each run trained on) and for promoted models written by other jobs. MLflow keeps metadata (runs, metrics, registry, aliases) in its tracking database, which stays off B2. B2 event notifications tell the registry when a promoted model is complete, so nothing polls the bucket.

## Setup

```bash
cd applications/mlflow-b2-model-registry
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt -e .
python -m pytest            # moto-backed; no credentials needed
```

Create a private bucket and application keys scoped to it (bucket names are globally unique):

| Role | Commands | Capabilities | Name prefix |
| --- | --- | --- | --- |
| Training client | `train` | `listFiles,readFiles,writeFiles` | `mlflow/` |
| Event receiver | `handle-event`, `serve-events` | `readFiles` | `models/` |
| Registry admin | `promote`, `artifact-uri` | none used; any key satisfies config | |

`promote` only touches the tracking database. Deleting artifacts (for example `mlflow gc`) needs `deleteFiles`; keep that on an operator key.

```bash
cp .env.example .env && chmod 600 .env     # key, bucket, B2_REGION; endpoint and artifact root are derived
mlflow-b2 artifact-uri                     # b2://<bucket>@s3.<region>.backblazeb2.com/mlflow
mlflow-b2 train --data ../../testdata/synthetic-shapes --epochs 5
mlflow-b2 promote --model shapes-classifier --version 1 --alias champion
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`train` creates the experiment with `artifact_location=<artifact root>/<experiment>`, tags the run and the registered version with `dataset.id` and `dataset.fingerprint` from the shard manifest, logs that manifest under `dataset/`, and logs a pyfunc model (weights as an artifact; it loads with only `mlflow` and `numpy`). The B2 key is handed to MLflow as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` inside the process only. For a shared tracking server, set `MLFLOW_TRACKING_URI` to the server from [infrastructure/mlflow](../../infrastructure/mlflow/); this app uses its direct artifact mode.

Event-driven registration: add an event notification rule on the bucket for `b2:ObjectCreated:*` with prefix `models/`, pointing at a public HTTPS URL that forwards to `mlflow-b2 serve-events`, and put the rule's signing secret in `B2_EVENT_SIGNING_SECRET`. A sample payload is in [examples/event-notifications.json](../../examples/event-notifications.json).

```bash
mlflow-b2 serve-events --host 127.0.0.1 --port 8080     # behind a TLS reverse proxy or tunnel
mlflow-b2 handle-event --payload body.json --signature "v1=<hex>"   # replay one delivery by hand
```

Behavior: the receiver checks `X-Bz-Event-Notification-Signature` (HMAC-SHA256 of the raw body, constant-time compare) before parsing, returns 401 on a mismatch and 413 above 1 MiB, writes the body to `--spool-dir` (`pending/<sha256>.json`, fsynced) and only then acks 200; if the write fails it returns 503 so B2 redelivers. One worker registers models in order. Temporary B2 or MLflow failures are retried with backoff up to `--max-attempts`; rejected manifests and exhausted retries move to `failed/` for inspection, and pending deliveries are replayed after a restart. Keep the spool on durable disk; deleting it loses accepted events. Only `models/<model>/<version>/manifest.json` keys matter. Each manifest is fetched, validated against the [model manifest schema](../../schemas/model-manifest.schema.json), and checked against the object key; its fingerprint is recomputed from the listed files, and every file must exist on B2 with the listed size and matching `sha256` user metadata (written by the training app). Hashing is not repeated on download, so the stored metadata is the integrity anchor. The new version's source is `b2://<bucket>@<host>/models/<model>/<version>/`, tagged with dataset id/fingerprint, source run/step, checkpoint fingerprint and `manifest.fingerprint`. Redeliveries are skipped when a version already carries that fingerprint. Exit codes: `0` ok, `1` config/B2/MLflow error, `2` bad signature, invalid dataset or a rejected manifest. Nothing prints keys or URLs.

Versions registered from events point at whatever format the training job wrote (for example `torch.save`); they are registry records with lineage, not pyfunc models. Versions from `train` are pyfunc models and load with `mlflow.pyfunc.load_model("models:/shapes-classifier@champion")`.

## Tests

`python -m pytest`: config (missing env, endpoint derivation, secret not in repr, artifact URI), dataset checksum gate, training end to end through MLflow's real `B2ArtifactRepository` (artifacts land under the prefix in the mocked bucket, model registered, tags present, model reloads from `b2://`), alias promotion, event signatures, non-matching keys, schema-invalid and mismatched manifests, idempotent replay, stale-fingerprint and changed-content rejection, webhook 200/401/413, spool persistence before the ack, retries, `failed/` retention, replay after restart and worker registration, CLI end to end.
moto serves the `s3.us-east-1.backblazeb2.com` host via `MOTO_S3_CUSTOM_ENDPOINTS`, so the `b2://` code path runs without network access.
`RUN_B2_LIVE_TESTS=1 python -m pytest tests/test_live.py` trains and promotes against your `.env` bucket with a throwaway experiment (artifacts stay under `<prefix>/live-test-*`; delete them afterwards).
moto covers S3 data operations only; key scoping and event notification delivery need live B2.
