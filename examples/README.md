# B2 configuration examples

Templates for the buckets in the [checkpointing](../architectures/checkpoint-and-resume-training/architecture.md), [streaming](../architectures/stream-training-data/architecture.md), [tracking](../architectures/track-experiments-and-register-models/architecture.md), and [orchestration](../architectures/orchestrate-training-pipelines/architecture.md) architectures. Placeholders are in `<angle brackets>`. The CLI flags follow the B2 CLI v4 shapes used by the sibling AI data provider repository, and the JSON shapes follow the [B2 API docs](https://www.backblaze.com/apidocs). These are templates: review placeholders, capabilities, retention periods, and deletion behavior before applying them to an account.

| File | Applies to | Purpose |
|---|---|---|
| `sse.json` | all buckets | SSE-B2 default encryption |
| `lifecycle-checkpoints.json` | checkpoints bucket | Purge pruned (hidden) checkpoint versions 7 days after deletion |
| `lifecycle-abandoned-runs.json` | checkpoints bucket | Hide and purge scratch runs after 30 days; do not apply to long production runs |
| `lifecycle-staging.json` | staging or pipeline-artifact bucket | Expire packing output and intermediate pipeline artifacts |
| `object-lock.json` | models bucket | Default governance retention on promoted models |
| `event-notifications.json` | models bucket | Webhook on new model manifests (registration trigger) |
| `event-notifications-retrain.json` | datasets bucket | Webhook on new dataset manifests (retraining trigger) |
| `replication.json` | models bucket | Bucket-to-bucket copy to a DR bucket |

Retention periods are placeholders. Set them from your audit and cost requirements; lifecycle rules delete data.

```bash
# Buckets (names are globally unique; Object Lock can be enabled at creation or later, but never disabled)
b2 bucket create "$ORG-train-datasets" allPrivate
b2 bucket create "$ORG-train-checkpoints" allPrivate
b2 bucket create --file-lock-enabled "$ORG-train-models" allPrivate
b2 bucket create "$ORG-mlflow-artifacts" allPrivate

# Default encryption, lifecycle, Object Lock
for b in datasets checkpoints models; do b2 bucket update --default-server-side-encryption SSE-B2 "$ORG-train-$b"; done
b2 bucket update --default-server-side-encryption SSE-B2 "$ORG-mlflow-artifacts"
b2 bucket update --lifecycle-rules "$(cat lifecycle-checkpoints.json)" "$ORG-train-checkpoints"
b2 bucket update --default-retention-mode governance --default-retention-period "365 days" "$ORG-train-models"

# Replication (the destination bucket must also have Object Lock enabled)
b2 bucket update --replication "$(cat replication.json)" "$ORG-train-models"

# Event notifications use the native API b2_set_bucket_notification_rules
# (key needs writeBucketNotifications); the bodies are the event-notifications*.json files.
```

## Scoped application keys (one bucket + one prefix per key)

```bash
b2 key create --bucket "$ORG-train-datasets"    --name-prefix datasets/           dataset-publisher listFiles,readFiles,writeFiles
b2 key create --bucket "$ORG-train-datasets"    --name-prefix "datasets/$DATASET/" training-reader   listFiles,readFiles
b2 key create --bucket "$ORG-train-checkpoints" --name-prefix "checkpoints/$RUN/"  checkpoint-writer listFiles,readFiles,writeFiles
b2 key create --bucket "$ORG-train-checkpoints" --name-prefix checkpoints/        checkpoint-pruner listFiles,readFiles,deleteFiles
b2 key create --bucket "$ORG-train-models"      --name-prefix models/             model-promoter    listFiles,readFiles,writeFiles,writeFileRetentions
b2 key create --bucket "$ORG-train-models"      --name-prefix models/             event-receiver    listFiles,readFiles
b2 key create --bucket "$ORG-mlflow-artifacts"  --name-prefix mlflow/             mlflow-client     listFiles,readFiles,writeFiles
```

A training job that both reads datasets and writes checkpoints in separate buckets needs two keys, one per bucket. `b2 key create` prints the secret once; store it in a gitignored `0600` file or a secret manager, never in shell history, logs, or container images. Bucket, lifecycle, retention, replication, and notification changes need an administrative key (or the master key), so run them once from provisioning and keep that key out of runtime services.

## Caveats

- Lifecycle rules use B2 semantics: `daysFromUploadingToHiding` hides a file, and `daysFromHidingToDeleting` purges hidden versions. A rule that hides by age also hides the newest checkpoint of a long run, so scope it to scratch prefixes only.
- Object Lock in governance mode can be bypassed only by a key with `bypassGovernance`. Compliance mode cannot be shortened by anyone.
- Replication copies within your account for disaster recovery. Replicated data cannot be replicated again, and deletes and hide markers are not replicated ([B2 docs](https://www.backblaze.com/docs/cloud-storage-cloud-replication)).
- Event notifications are delivered at least once and must be acknowledged quickly. Receivers verify `X-Bz-Event-Notification-Signature` and deduplicate ([B2 docs](https://www.backblaze.com/docs/cloud-storage-event-notifications-reference-guide)).
