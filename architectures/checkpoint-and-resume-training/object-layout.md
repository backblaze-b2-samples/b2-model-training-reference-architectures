# Object layout and data-flow spec

Shared by [streaming](../stream-training-data/architecture.md), [checkpointing](architecture.md), and [experiment tracking](../track-experiments-and-register-models/architecture.md). Manifest contracts: [training run contract](../../docs/training-run-contract.md).

```
<org>-train-datasets/
  datasets/<dataset>/<version>/shards/<split>-<n:06d>.tar   WebDataset shards
  datasets/<dataset>/<version>/manifest.json                shard manifest, written last
<org>-train-checkpoints/
  checkpoints/<run>/step-<step:08d>/<dcp files>             .metadata plus per-rank __<rank>_<n>.distcp
  checkpoints/<run>/step-<step:08d>/manifest.json           checkpoint manifest, written last
<org>-train-models/
  models/<model>/<version>/<files>                          copied from one checkpoint
  models/<model>/<version>/manifest.json                    model manifest, written last
<org>-mlflow-artifacts/
  mlflow/<experiment>/<run-id>/artifacts/...                tracking server artifact root
```

Rules:

- No leading `/` and no `//` in keys. Names and versions use `[a-z0-9._-]`.
- Zero-padded step numbers make lexical listing order match numeric order.
- A dataset version, a promoted model version, and a checkpoint step are immutable. Rerunning identical content is skipped; different content at an existing key is refused. Publish a new version instead.
- Consumers trust a prefix only when its `manifest.json` exists and its listed files match.
- Every object stores `sha256` in user metadata. Keep metadata small: Object Lock buckets have a lower per-version metadata limit.

## Stages

| Stage | Input | Output | Gate |
|---|---|---|---|
| Package | Released dataset | `datasets/<dataset>/<version>/` | Local checksums match; manifest last |
| Pin | Shard manifest | Run configuration | Fingerprint equals the pinned value |
| Train | Shards | Model state | Data-loader assignment by rank and worker |
| Checkpoint | Model and optimizer state | `checkpoints/<run>/step-<step>/` | All ranks uploaded; remote verify; manifest last |
| Resume | Newest complete checkpoint | Restored state | Manifest present; checksums match |
| Promote | One checkpoint | `models/<model>/<version>/` | Copy verified; manifest last; retention set |
| Register | Model manifest created event | Registry version | Signature verified; idempotent by fingerprint |
| Prune | Old checkpoints | Hidden versions | Keep newest K complete; never touch `models/` |

Examples: [checkpoint manifest](../../manifests/checkpoint-manifest.example.json), [model manifest](../../manifests/model-manifest.example.json), [shard manifest](../../testdata/synthetic-shapes/manifest.json).
