# MLflow tracking server with B2 artifacts

Docker Compose for a shared MLflow tracking server: PostgreSQL holds runs, metrics and the model registry; B2 holds artifacts through MLflow's native `b2://` artifact repository.
Used by the [track experiments and register models](../../architectures/track-experiments-and-register-models/architecture.md) architecture and the [mlflow-b2-model-registry](../../applications/mlflow-b2-model-registry/) sample.

Image: `ghcr.io/mlflow/mlflow:v3.17.1-full`. The `-full` variant installs `mlflow[extras,db,...]`, which brings boto3 and psycopg2; the plain tag has neither. `b2://` support first shipped in MLflow 3.11.1, so do not pin older images. The image is large; build a slim one (`pip install "mlflow==3.17.1" boto3 psycopg2-binary`) if pull size matters.

## Artifact modes

The artifact location is fixed per experiment when it is created, so choose a mode before teams start logging.

| | (a) direct: `--default-artifact-root b2://...` | (b) proxied: `--artifacts-destination b2://...` + `--serve-artifacts` |
| --- | --- | --- |
| Profile | `COMPOSE_PROFILES=direct` | `COMPOSE_PROFILES=proxy` (default) |
| Experiment location | `b2://<bucket>@s3.<region>.backblazeb2.com/<prefix>/<id>` | `mlflow-artifacts:/<id>`; the server maps it to B2 |
| Who holds B2 keys | every client (training jobs, notebooks, serving) plus the server for the UI's artifact viewer | the server only |
| Artifact bandwidth | clients to B2 directly, in parallel; the server carries metadata only | every upload and download streams through the server |
| Scaling | add clients freely; B2 absorbs the throughput | size server CPU, network and workers for peak checkpoint and model traffic |
| Access control | B2 key scoping per role and prefix | MLflow server auth (for example `--app-name basic-auth`) in front of one B2 key |
| Good fit | training clusters moving large checkpoints and models | many lightweight users, notebooks, or clients you cannot hand keys to |

Mixed setups work: a server in mode (b) still honors experiments created with an explicit `b2://` location (the sample app's `train` does this), and those clients need their own keys. Model versions registered from B2 event notifications point at `b2://.../models/<model>/<version>/`, so downloading them always needs a B2 read key on the client.

## Keys per role

Create one key per role, scoped to the bucket and, where possible, a name prefix. Never use the master key.

| Role | Mode | Capabilities | Name prefix |
| --- | --- | --- | --- |
| Tracking server | (b) proxied | `listFiles,readFiles,writeFiles` (`writeFiles` also covers artifact deletion by name, which hides files) | `mlflow/` |
| Tracking server | (a) direct | `listFiles,readFiles` (UI artifact viewer) | `mlflow/` |
| Training client | (a) direct | `listFiles,readFiles,writeFiles` | `mlflow/` |
| Serving / inference | any | `listFiles,readFiles` | `mlflow/` or `models/` |
| Event receiver | any | `readFiles` | `models/` |
| Operator (`mlflow gc`, cleanup) | any | `listFiles,writeFiles` to hide deleted artifacts; add `deleteFiles` only to remove specific versions | `mlflow/` |

The server receives its key as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` inside the container; B2 keys are not shared with PostgreSQL.

## Setup

```bash
cd infrastructure/mlflow
cp .env.example .env && chmod 600 .env     # bucket, region, server key, Postgres password, profile
docker compose config >/dev/null           # fails fast on any missing value
docker compose up -d
open http://localhost:5000
```

Clients: `export MLFLOW_TRACKING_URI=http://localhost:5000`. In mode (b) that is all they need. In mode (a) they also need a training key in the standard boto3 variables (the sample app maps `B2_APPLICATION_KEY_ID` / `B2_APPLICATION_KEY` for you).

The database stays off B2: PostgreSQL uses the `pgdata` volume and is not published outside the Compose network. Back it up with `pg_dump` on its own schedule; the registry, aliases and run metadata live only there, while B2 holds the bytes they point to. The server binds `127.0.0.1:5000`; to expose it, put TLS and authentication in front and add your hostname to `MLFLOW_ALLOWED_HOSTS`.
