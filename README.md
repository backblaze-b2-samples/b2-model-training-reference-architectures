# B2 reference architectures for model training

How training teams use Backblaze B2 to stream versioned training data, checkpoint and resume distributed training, track experiments and register models, pin dataset versions for reproducibility, and orchestrate retraining. Documentation first, with the smallest runnable code that proves it.

Start with the [architecture index](docs/architecture-index.md), the [agent-readable summary](docs/model-training.md), or the portable [agent skill](skill/SKILL.md). Current B2 product behavior is documented at [backblaze.com/docs](https://www.backblaze.com/docs/cloud-storage).

## Where does Backblaze B2 fit in model training?

Backblaze B2 is the durable store for training data, checkpoints, and models, wherever the GPUs run.

- **Store in B2:** training-ready dataset versions (shards and manifests), checkpoints, run artifacts, promoted models.
- **Read from B2:** stream shards straight into data loaders; resume from the newest complete checkpoint on any compute provider.
- **Protect in B2:** Object Lock on promoted models, versions and lifecycle rules for checkpoints, checksummed manifests, Cloud Replication for disaster recovery.
- **Automate with B2:** event notifications to register promoted models and to retrain when a new dataset version lands.
- **Keep elsewhere:** tracking and orchestrator databases, hot sample caches (local NVMe), compute (bring your own or use a compute partner).

Pricing, egress terms, and limits change: see [B2 pricing](https://www.backblaze.com/cloud-storage/pricing) and the [B2 docs](https://www.backblaze.com/docs/cloud-storage).

## Architectures

<!-- catalog:architectures:start -->
| Architecture | Purpose | Resources |
|---|---|---|
| [Stream versioned training shards from B2 to GPU workers](architectures/stream-training-data/architecture.md) | Package a released dataset as immutable WebDataset shards on B2, then stream them into data loaders on every worker with an optional local cache keyed by checksum. | [app](applications/pytorch-b2-checkpointed-training/README.md) · [deployment](infrastructure/kubernetes/README.md) |
| [Checkpoint distributed training to B2 and resume after interruption](architectures/checkpoint-and-resume-training/architecture.md) | Save sharded checkpoints to local disk, upload them to B2 with a manifest written last, resume from the newest complete checkpoint, and lock promoted checkpoints. | [app](applications/pytorch-b2-checkpointed-training/README.md) · [deployment](infrastructure/kubernetes/README.md) |
| [Track experiments and register models with a B2 artifact store](architectures/track-experiments-and-register-models/architecture.md) | Keep tracking metadata in a database, store run artifacts and registered model files on B2, and register promoted models automatically from B2 event notifications. | [app](applications/mlflow-b2-model-registry/README.md) · [deployment](infrastructure/mlflow/README.md) |
| [Pin dataset versions for reproducible training runs](architectures/pin-dataset-versions/architecture.md) | Version training data on B2 with a data-versioning tool or immutable release prefixes, and pin every run to an exact version and fingerprint. | [deployment](https://docs.lakefs.io/latest/howto/deploy/) |
| [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) | Run preprocessing, training, evaluation, and registration as pipeline steps that exchange artifacts through B2, and trigger retraining from B2 event notifications. | [deployment](https://argo-workflows.readthedocs.io/en/latest/configure-artifact-repository/) |
<!-- catalog:architectures:end -->

## Third-party integrations

"Native B2 backend" means the tool ships B2-specific code. "S3-compatible endpoint" means the tool's S3 client accepts a custom endpoint, verified in its documentation or source. "Candidate" means the path is plausible but has not been tested here.

<!-- catalog:integrations:start -->
| Tool | Role | Support | How it connects to B2 | Architectures |
|---|---|---|---|---|
| [PyTorch](https://pytorch.org/docs/stable/data.html) | Training framework; IterableDataset streaming and distributed checkpointing | S3-compatible endpoint | boto3 or fsspec/s3fs clients configured with the regional B2 endpoint; no PyTorch-specific B2 code is required | [Stream versioned training shards from B2 to GPU workers](architectures/stream-training-data/architecture.md)<br>[Checkpoint distributed training to B2 and resume after interruption](architectures/checkpoint-and-resume-training/architecture.md) |
| [WebDataset](https://github.com/webdataset/webdataset) | Sequential tar-shard format and loader for large-scale training | S3-compatible endpoint | Shards read through fsspec URLs or a pipe command whose S3 client uses the B2 endpoint | [Stream versioned training shards from B2 to GPU workers](architectures/stream-training-data/architecture.md) |
| [Ray Train, Tune, and Data](https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html) | Distributed training, hyperparameter tuning, and parallel data loading | S3-compatible endpoint | pyarrow.fs.S3FileSystem with endpoint_override passed as storage_filesystem or to Ray Data readers | [Stream versioned training shards from B2 to GPU workers](architectures/stream-training-data/architecture.md)<br>[Checkpoint distributed training to B2 and resume after interruption](architectures/checkpoint-and-resume-training/architecture.md) |
| [TensorFlow with tensorflow-io](https://github.com/tensorflow/io) | tf.data input pipelines over TFRecord files | S3-compatible endpoint | tensorflow-io S3 filesystem with the S3_ENDPOINT environment variable set to the B2 endpoint | [Stream versioned training shards from B2 to GPU workers](architectures/stream-training-data/architecture.md) |
| [Hugging Face Datasets](https://huggingface.co/docs/datasets/filesystems) | Dataset loading and preprocessing for transformer training | S3-compatible endpoint | fsspec storage_options with client_kwargs.endpoint_url set to the B2 endpoint | [Stream versioned training shards from B2 to GPU workers](architectures/stream-training-data/architecture.md)<br>[Pin dataset versions for reproducible training runs](architectures/pin-dataset-versions/architecture.md) |
| [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/common/checkpointing_advanced.html) | Training loop with ModelCheckpoint callbacks | Candidate, needs acceptance test | fsspec-backed checkpoint paths with storage options for the B2 endpoint | [Checkpoint distributed training to B2 and resume after interruption](architectures/checkpoint-and-resume-training/architecture.md) |
| [JAX with TensorStore](https://google.github.io/tensorstore/kvstore/s3/index.html) | Sharded array checkpoints for JAX training through TensorStore | Candidate, needs acceptance test | TensorStore s3 key-value store driver with its endpoint option set to the B2 endpoint; TensorStore documents that S3-compatible stores may not give the atomicity it expects, so test the checkpoint commit protocol before adopting | [Checkpoint distributed training to B2 and resume after interruption](architectures/checkpoint-and-resume-training/architecture.md) |
| [MLflow](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/) | Experiment tracking, artifact store, and model registry | Native B2 backend | Native b2://<bucket>@s3.<region>.backblazeb2.com/<path> artifact URIs; s3:// with MLFLOW_S3_ENDPOINT_URL also works | [Track experiments and register models with a B2 artifact store](architectures/track-experiments-and-register-models/architecture.md) |
| [Weights & Biases](https://docs.wandb.ai/guides/artifacts/track-external-files/) | Experiment tracking with reference artifacts in customer storage | S3-compatible endpoint | Reference artifacts on s3:// URIs with AWS_S3_ENDPOINT_URL set to the B2 endpoint | [Track experiments and register models with a B2 artifact store](architectures/track-experiments-and-register-models/architecture.md) |
| [ClearML](https://clear.ml/docs/latest/docs/integrations/storage/) | Experiment tracking, model repository, and pipelines | S3-compatible endpoint | output_uri on the B2 endpoint with host-specific credentials in clearml.conf | [Track experiments and register models with a B2 artifact store](architectures/track-experiments-and-register-models/architecture.md) |
| [Aim](https://aimstack.readthedocs.io/en/latest/using/artifacts.html) | Experiment tracking with logged artifacts in object storage | S3-compatible endpoint | S3 artifacts backend with endpoint_url passed to its boto3 client; metrics and run metadata stay in Aim's local RocksDB and SQLite stores | [Track experiments and register models with a B2 artifact store](architectures/track-experiments-and-register-models/architecture.md) |
| [lakeFS](https://docs.lakefs.io/latest/howto/deploy/) | Git-like branches, commits, and tags over object storage | S3-compatible endpoint | S3 blockstore with blockstore.s3.endpoint set to the B2 endpoint and path-style addressing | [Pin dataset versions for reproducible training runs](architectures/pin-dataset-versions/architecture.md) |
| [DVC](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3) | Dataset and model versioning tied to Git commits | S3-compatible endpoint | s3 remote with endpointurl set to the B2 endpoint | [Pin dataset versions for reproducible training runs](architectures/pin-dataset-versions/architecture.md) |
| [Pachyderm](https://github.com/pachyderm/pachyderm/blob/master/src/internal/obj/bucket.go) | Data versioning and pipeline provenance | Candidate, needs acceptance test | S3 object storage through gocloud.dev blob URLs with an explicit endpoint parameter; deployment configuration and current maintenance status need confirmation on a pinned release | [Pin dataset versions for reproducible training runs](architectures/pin-dataset-versions/architecture.md) |
| [ZenML](https://sdkdocs.zenml.io/0.96.1/integration_code_docs/integrations-b2.html) | ML pipeline framework with pluggable stacks | Native B2 backend | B2 artifact store flavor with a regional endpoint, plus the S3 flavor with a custom endpoint_url | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) | Kubernetes-native ML pipelines | S3-compatible endpoint | S3-compatible object store configuration with endpoint, region, and path-style settings for the API server and launcher | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [Metaflow](https://docs.metaflow.org/getting-started/infrastructure) | ML workflow framework with a content-addressed datastore | S3-compatible endpoint | S3 datastore with METAFLOW_S3_ENDPOINT_URL set to the B2 endpoint | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [Flyte](https://docs.flyte.org/) | Kubernetes-native workflow orchestration for ML | S3-compatible endpoint | storage.type s3 with storage.connection.endpoint set to the B2 endpoint and access-key authentication | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [Kedro](https://docs.kedro.org/en/stable/catalog-data/data_catalog_yaml_examples.html) | Data and ML pipeline project framework | S3-compatible endpoint | Catalog credentials with client_kwargs.endpoint_url passed to s3fs | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [Argo Workflows](https://argo-workflows.readthedocs.io/en/latest/configure-artifact-repository/) | Kubernetes workflow engine for training jobs | S3-compatible endpoint | S3 artifact repository with an explicit endpoint, bucket, and key secrets | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [Dagster](https://docs.dagster.io/api/libraries/dagster-aws) | Asset-based data and ML pipeline orchestration | S3-compatible endpoint | dagster-aws S3Resource and S3 IO managers with endpoint_url set to the B2 endpoint | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [Prefect](https://github.com/PrefectHQ/prefect/blob/main/src/integrations/prefect-aws/prefect_aws/client_parameters.py) | Python workflow orchestration with storage blocks | S3-compatible endpoint | prefect-aws S3Bucket with AwsClientParameters(endpoint_url=...) set to the B2 endpoint | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
| [MLRun](https://docs.mlrun.org/en/stable/store/datastore.html) | ML orchestration with artifact, model, and dataset stores | S3-compatible endpoint | s3 datastore with AWS_ENDPOINT_URL_S3 or a DatastoreProfileS3 endpoint_url set to the B2 endpoint | [Orchestrate training pipelines and retrain when new data lands](architectures/orchestrate-training-pipelines/architecture.md) |
<!-- catalog:integrations:end -->

## Sample applications

Both samples use the shared [synthetic-shapes dataset](testdata/synthetic-shapes/README.md) by default and run their tests without credentials.

| Name | Capabilities demonstrated | Path |
|---|---|---|
| pytorch-b2-checkpointed-training | S3 API, multipart, scoped keys, manifest-last commits, fingerprint pinning, shard streaming, local cache, DCP checkpoints, resume, server-side copy, Object Lock retention, pruning | [applications/pytorch-b2-checkpointed-training](applications/pytorch-b2-checkpointed-training/README.md) |
| mlflow-b2-model-registry | MLflow native `b2://` artifact store, model registry, aliases, signed event notifications, idempotent event handling | [applications/mlflow-b2-model-registry](applications/mlflow-b2-model-registry/README.md) |

Deployment templates: [Kubernetes training job](infrastructure/kubernetes/README.md) and [MLflow tracking server](infrastructure/mlflow/README.md).

### Related Backblaze B2 samples

| Sample | Demonstrates |
|---|---|
| [WebDataset Streaming PyTorch Training](https://github.com/backblaze-b2-samples/webdataset-streaming-pytorch-training) | Manifested shards streamed from B2 into distributed PyTorch training |
| [B2 AI and data notebooks](https://github.com/backblaze-b2-samples/notebooks) | Ray Train and Tune checkpoints, PyTorch image classification, and lakeFS dataset versioning |
| [Unsloth LoRA Checkpoint Versioning](https://github.com/backblaze-b2-samples/unsloth-lora-checkpoint-versioning) | Fine-tuning checkpoints and experiment tracking on B2 |
| [lakeFS on B2 Quickstart](https://github.com/backblaze-b2-samples/lakefs-on-b2-quickstart) | Versioned data lake with B2 as the lakeFS blockstore |
| [LeRobot B2 Streaming](https://github.com/backblaze-b2-samples/lerobot-b2-streaming) | Robot datasets streamed from B2 for training without a full download |
| [B2 Event Broker](https://github.com/backblaze-b2-samples/b2-event-broker) | Signed event validation, prompt acknowledgement, fan-out, and retries |

See the [full sample cross-reference](docs/related-b2-samples.md) for fine-tuning, caching, feature-store, evaluation, and Kubernetes storage examples.

## B2 capability map

| Area | Capabilities | Selection notes |
|---|---|---|
| Access | S3 API, multipart, server-side copy, scoped keys (one bucket + one prefix + capabilities), presigned URLs | Default to the S3 API; use the native API only for notifications, lock, and replication setup |
| Change and lifecycle | Event notifications, versions, lifecycle rules (hide, then delete) | Events trigger registration and retraining; receivers must acknowledge fast and be idempotent |
| Protection | SSE-B2, SSE-C, Object Lock, replication | Lock promoted models; replication is bucket-to-bucket disaster recovery |
| Data movement | Included egress, compute partners | Stream to wherever the GPUs are; check pricing for terms |
| Performance | Standard B2, [Overdrive](https://www.backblaze.com/cloud-storage/b2-overdrive), local cache | Standard by default; cache immutable shards locally for repeated epochs; Overdrive when sustained aggregate throughput is the measured bottleneck |

## Repository layout

`catalog/` machine-readable architecture metadata, `schemas/` validated contracts, `architectures/` designs and diagrams, `applications/` runnable samples, `examples/` B2 configurations, `infrastructure/` deployment templates, `testdata/` the shared category dataset, `manifests/` example run manifests, `skill/` portable agent skill, `docs/` agent-readable pages, `tools/` generators and validators, `.github/` CI and templates. See [CONTRIBUTING](.github/CONTRIBUTING.md).

## Documentation and machine discovery

- [Backblaze B2 documentation](https://www.backblaze.com/docs/cloud-storage) is canonical for product behavior; this repository provides workflow-specific designs and runnable examples.
- [catalog/architectures.yaml](catalog/architectures.yaml) is the source of truth for the architecture and integration tables and is validated by [JSON Schema](schemas/architecture-catalog.schema.json).
- [llms.txt](llms.txt) is a curated agent index. All narrative pages are maintained as clean Markdown.
- The canonical `backblaze.com` deployment controls crawler access; [metadata/robots.txt.example](metadata/robots.txt.example) documents the intended policy without pretending that a repository file changes the domain's robots configuration.
- The [training run contract](docs/training-run-contract.md) defines shard, checkpoint, and model manifests and the lineage between them.
- Reuse and attribution are covered by the [MIT license](LICENSE) and [citation metadata](CITATION.cff); report sensitive issues through the [security policy](.github/SECURITY.md).

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-catalog.txt
python3 tools/generate_catalog.py --check
python3 tools/validate_contracts.py
cd applications/pytorch-b2-checkpointed-training && pip install -r requirements-dev.txt -e . && python -m pytest
```
