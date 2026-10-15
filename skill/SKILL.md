---
name: b2-model-training
description: Design or review Backblaze B2 storage for model training, including streaming training shards to GPU workers, PyTorch, Ray, or JAX checkpoints and resume, MLflow, Weights & Biases, ClearML, or Aim artifact stores, dataset version pinning with lakeFS, DVC, or Pachyderm, and ZenML, Kubeflow Pipelines, Metaflow, Flyte, Kedro, Argo, Dagster, Prefect, or MLRun retraining pipelines. Use for training storage boundaries, key scoping, manifest contracts, Object Lock on models, lifecycle for checkpoints, event-driven registration, or B2 Overdrive decisions.
---

# B2 for model training

Choose a workflow from the bundled [architecture catalog](references/architecture-catalog.md). For shard, checkpoint, and model manifests and the lineage between them, read the [run contract](references/run-contract.md).

For changing service behavior, limits, and setup steps, use the canonical [Backblaze B2 documentation](https://www.backblaze.com/docs/cloud-storage). Link pricing claims to [B2 pricing](https://www.backblaze.com/cloud-storage/pricing) instead of copying numbers.

## Rules
- Use the S3-compatible API. Derive `https://s3.<region>.backblazeb2.com` from `B2_REGION`; never hardcode a region or endpoint.
- Env: `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`, `B2_BUCKET_NAME`, `B2_REGION`, optional `B2_ENDPOINT`. Secrets go in gitignored 0600 `.env` files or a secret store; never print keys.
- A key scopes to one bucket + one prefix + capabilities. Design roles around that: dataset-publisher, training-reader, checkpoint-writer, checkpoint-pruner, model-promoter, event-receiver, tracking client or server. Use the master key only for provisioning.
- Separate buckets for datasets, checkpoints, and models, because lifecycle, Object Lock, replication, and notifications are bucket-level.
- Write `manifest.json` last under every dataset, checkpoint, and model prefix. A prefix without a manifest is incomplete: never train on it, resume from it, or register it.
- Pin each run to `<dataset>@<version>` plus the fingerprint; stop on mismatch; copy the pin into every checkpoint, model, and tracking record.
- Checkpoints: stage locally, upload in parallel with multipart, write the manifest after all ranks finish, and resume from the newest step whose files match their checksums.
- Promote by server-side copy (multipart copy above 5 GB) into an Object Lock models bucket in governance mode. Make retention an explicit promotion decision with a policy-set duration; unlocked promotion is an explicit opt-out, not a default. Compliance mode and enabling Object Lock are irreversible.
- Prune checkpoints by deleting, and purge hidden versions with `daysFromHidingToDeleting`. Never apply an age-based hide rule to active runs.
- Events: webhook, at least once; verify `X-Bz-Event-Notification-Signature`, acknowledge fast, deduplicate, and filter on manifest keys.
- Streaming: WebDataset-style tar shards of hundreds of MB to a few GB; at least `world_size × workers` shards per split; cache immutable shards on local NVMe keyed by SHA-256.
- MLflow: native `b2://<bucket>@s3.<region>.backblazeb2.com/<path>` artifact URIs, credentials in `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Choose direct client access or proxied `--serve-artifacts` deliberately. Keep the tracking database off B2.
- W&B: reference artifacts with `AWS_S3_ENDPOINT_URL`. ClearML: `output_uri` with host credentials in `clearml.conf`. Aim: S3 artifacts backend with `endpoint_url`; its metrics and run metadata stay local.
- PyTorch DCP: local staging plus upload, or `FsspecWriter` with s3fs `endpoint_url`. Ray Train and Tune: `storage_filesystem=pyarrow.fs.S3FileSystem(endpoint_override=...)`. Treat Lightning remote checkpoints and JAX checkpoints through TensorStore's `s3` driver (documented weaker atomicity) as candidates until tested.
- lakeFS: S3 blockstore with the B2 endpoint and path-style addressing. DVC: s3 remote with `endpointurl`. Pachyderm: candidate until a pinned deployment is tested. ZenML: `b2` artifact store flavor. KFP, Metaflow, Flyte, Kedro, Argo, Dagster, Prefect, MLRun: S3 artifact stores with a custom endpoint.
- Use Standard B2 by default. Recommend Overdrive only when sustained aggregate read or checkpoint throughput is the measured bottleneck after parallel prefetch and local caching.
- Keep tracking, registry, orchestrator, and versioning-tool databases, queues, and hot caches off B2.
- State whether a recommendation rests on product documentation, source review, automated tests, or a deployment-specific measurement. Do not turn documentation review into a claim of production results.

## Verify

When operating inside the full repository, use its `AGENTS.md` validation commands. When this skill is installed by itself, validate recommendations against the linked canonical product and integration documentation; repository-local tests are not bundled with the portable skill.
