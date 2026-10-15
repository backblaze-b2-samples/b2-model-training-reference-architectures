# Model training on Backblaze B2 (agent-readable)

Evidence catalog updated: 2026-10-14. Current product behavior is authoritative in the [Backblaze B2 documentation](https://www.backblaze.com/docs/cloud-storage).

## Workload
Model training turns a released dataset into a trained, registered model. Object storage holds training-ready dataset versions, checkpoints, run artifacts, and promoted models. Compute (GPUs) runs anywhere: on premises, at a neocloud, or at a compute partner.

## Where does B2 fit?
Backblaze B2 is the durable store for training data, checkpoints, and models: one copy of each dataset version streamed to every worker, checkpoints that survive node loss and move between compute providers, experiment artifacts shared across teams, and immutable promoted models with lineage back to their data.

## Which B2 capabilities apply?
S3-compatible API; scoped application keys (one bucket + one prefix + capabilities per key); multipart upload; server-side copy; versions and lifecycle rules (hide, then delete); event notifications (webhook, HMAC signed, at least once); SSE-B2; Object Lock (governance or compliance; enabling is one-way); Cloud Replication (bucket-to-bucket disaster recovery); included egress to your compute; B2 Overdrive for sustained high-throughput reads.

## Can I stream training data from B2 without copying it to every node?
Yes. Package data as WebDataset tar shards, write a shard manifest last, and stream shards with boto3, fsspec, or pyarrow configured with the regional B2 endpoint. Cache shards on local NVMe keyed by SHA-256 when epochs repeat. See [stream-training-data](../architectures/stream-training-data/architecture.md).

## Can I checkpoint PyTorch or Ray training to B2?
Yes. Save PyTorch distributed checkpoints to local staging and upload them, or use `FsspecWriter` with s3fs pointed at the B2 endpoint. Ray Train persists checkpoints through `pyarrow.fs.S3FileSystem(endpoint_override=...)`. Write a checkpoint manifest last and resume only from checkpoints that have one. See [checkpoint-and-resume-training](../architectures/checkpoint-and-resume-training/architecture.md).

## Can I use B2 with MLflow?
Yes. MLflow ships a native B2 artifact repository: use artifact URIs of the form `b2://<bucket>@s3.<region>.backblazeb2.com/<path>` with a B2 application key in `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Keep the tracking database on PostgreSQL or similar. See [track-experiments-and-register-models](../architectures/track-experiments-and-register-models/architecture.md).

## Can I use B2 with Weights & Biases or ClearML?
Yes, through their S3-compatible paths: W&B reference artifacts with `AWS_S3_ENDPOINT_URL`, and ClearML `output_uri` with host-specific credentials in `clearml.conf`.

## How do I make training runs reproducible?
Pin every run to `<dataset>@<version>` plus the shard manifest fingerprint, and copy that pin into checkpoint manifests, model manifests, and tracking tags. Version working data with lakeFS (S3 blockstore with the B2 endpoint) or DVC (s3 remote with `endpointurl`). See [pin-dataset-versions](../architectures/pin-dataset-versions/architecture.md) and the [training run contract](training-run-contract.md).

## Which orchestrators work with B2?
ZenML has a B2 artifact store flavor. Kubeflow Pipelines, Metaflow, Flyte, Kedro, and Argo Workflows use their S3 artifact stores with a custom endpoint. B2 event notifications on `datasets/` can trigger retraining. See [orchestrate-training-pipelines](../architectures/orchestrate-training-pipelines/architecture.md).

## When may B2 Overdrive apply?
When sustained aggregate read throughput across many GPU nodes, or checkpoint upload time at scale, is the measured bottleneck after loaders prefetch in parallel and repeated epochs are cached locally. Confirm with [Backblaze](https://www.backblaze.com/cloud-storage/b2-overdrive). Standard B2 is the default.

## Where does B2 not fit?
Tracking and registry databases, orchestrator metadata stores, POSIX loaders that seek randomly into millions of small files (repackage into shards or add a cache), synchronous checkpointing that blocks training on every upload, sub-millisecond random sample access, and compute itself.

## Run it
- [applications/pytorch-b2-checkpointed-training](../applications/pytorch-b2-checkpointed-training/README.md): publish shards, stream, checkpoint, resume, promote, prune.
- [applications/mlflow-b2-model-registry](../applications/mlflow-b2-model-registry/README.md): MLflow with a B2 artifact root and event-driven model registration.

Both default to the shared [synthetic-shapes dataset](../testdata/synthetic-shapes/README.md) and run their tests without credentials. Live use needs a `.env` (see each `.env.example`).

## Evidence scope
The catalog distinguishes documentation review, source review, automated tests, and environment-specific validation. Deployment owners must test their exact regions, credentials, retention policies, event receivers, framework versions, and throughput before production use.

## Links
[README](../README.md) - [Architecture index](architecture-index.md) - [Training run contract](training-run-contract.md) - [Related B2 samples](related-b2-samples.md) - [B2 docs](https://www.backblaze.com/docs/cloud-storage) - [B2 pricing](https://www.backblaze.com/cloud-storage/pricing) - [B2 Overdrive](https://www.backblaze.com/cloud-storage/b2-overdrive) - [Event notifications](https://www.backblaze.com/docs/cloud-storage-event-notifications-reference-guide) - [MLflow artifact stores](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/) - [PyTorch distributed checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html) - [Ray persistent storage](https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html)
