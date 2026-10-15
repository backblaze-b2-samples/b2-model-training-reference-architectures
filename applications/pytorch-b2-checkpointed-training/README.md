# pytorch-b2-checkpointed-training

PyTorch CLI for [streaming training data](../../architectures/stream-training-data/architecture.md) and [checkpoint-and-resume training](../../architectures/checkpoint-and-resume-training/architecture.md):
publish WebDataset shards -> stream them from B2 into a tiny CNN -> commit `torch.distributed.checkpoint` checkpoints to B2 -> resume after interruption -> promote a checkpoint to a model version -> prune old checkpoints.
Uses the B2 S3-compatible API via boto3. Runs on CPU in seconds against the shared [`synthetic-shapes`](../../testdata/synthetic-shapes/manifest.json) dataset.

Capabilities demonstrated: S3 API, streaming GETs (`tarfile` in `r|` mode, no full download), rank- and worker-aware shard splitting, sha256-keyed local shard cache, multipart uploads (`--multipart-mb`), manifest-last commit markers, resume from the latest complete checkpoint, server-side copy for promotion, Object Lock retention (`--lock-days`), scoped keys, and dataset fingerprint pinning across dataset, checkpoint and model.

## B2's role

Durable store for training shards, checkpoints and promoted models. B2 provides the objects, sha256 metadata, server-side copy and retention; the CLI provides the manifests that make each step atomic and the lineage that ties a model back to its checkpoint and dataset version.
Local disk is only staging and cache: a new machine can resume a run from B2 alone.

## Object layout

```
datasets/<dataset>/<version>/shards/*.tar         WebDataset shards (png, cls, json per sample)
datasets/<dataset>/<version>/manifest.json        shard manifest, written last
checkpoints/<run>/step-<step:08d>/<files>         DCP files (.metadata, __0_0.distcp)
checkpoints/<run>/step-<step:08d>/manifest.json   checkpoint manifest, written last = commit marker
models/<model>/<version>/<files>                  server-side copies of one checkpoint
models/<model>/<version>/manifest.json            model manifest, written last
```

This sample uses one bucket with three prefixes. In production, put datasets, checkpoints and models in separate buckets: Object Lock, lifecycle rules and replication are bucket-level settings, and they differ per stage (locked models, short-lived checkpoints, replicated datasets).

## Setup

```bash
cd applications/pytorch-b2-checkpointed-training
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt -e .
python -m pytest            # moto-backed; no credentials needed
```

Create a bucket (private) and an application key scoped to it. Bucket names are globally unique.

```bash
cp .env.example .env && chmod 600 .env     # fill in the key, bucket and B2_REGION; endpoint is derived
b2-train publish-dataset --src ../../testdata/synthetic-shapes
b2-train train --run-id shapes-run-001 --checkpoint-every 3 --max-steps 7 --cache-dir cache   # simulated preemption
b2-train train --run-id shapes-run-001 --checkpoint-every 3 --cache-dir cache                 # resumes from step 6
b2-train promote --run-id shapes-run-001 --model shapes-cnn --model-version v1 --no-lock      # --lock-days N on a lock-enabled bucket
b2-train prune --run-id shapes-run-001 --keep 2
```

Behavior:

- `publish-dataset` checks every local shard's size and sha256 and the manifest fingerprint before uploading (exit 2, nothing uploaded), uploads shards with sha256 metadata, verifies the remote copies (exit 3), then writes the shard manifest. Reruns skip identical objects; different content at an existing key exits 1.
- `train` loads the shard manifest from B2 and refuses (exit 2) when `--expect-fingerprint` differs. Each shard is one streaming GET whose bytes are hashed as they are read; a mismatch fails the run. With `--cache-dir`, shards are kept as `<sha256>.tar` after verification and later epochs read them locally. A checkpoint is saved with DCP to `--staging-dir`, uploaded, verified, and committed by writing its manifest (which pins the dataset id and fingerprint). On start, `train` resumes from the highest step that has a manifest, verifies each downloaded file's sha256, restores model, optimizer and data position, and continues; steps without a manifest are ignored, and a checkpoint pinned to a different dataset fingerprint is refused (exit 2). `--max-steps` stops without a final checkpoint to simulate preemption; `--no-resume` starts over and refuses to overwrite a committed step.
- `promote` copies the checkpoint files server side (managed copy switches to UploadPartCopy above `--multipart-mb`, since a single CopyObject is capped at 5 GB), verifies size and sha256 metadata, then writes the model manifest. Retention is a required choice: `--lock-days N` gives every object GOVERNANCE retention and records `retention_mode: governance`; `--no-lock` is the explicit opt-out for demo buckets without Object Lock and records `retention_mode: none`. Set N from your audit policy; the sample has no default duration. Promoting the same checkpoint again is a no-op; a different checkpoint at an existing model version exits 1.
- `prune` keeps the newest `--keep` complete checkpoints and any step newer than the newest complete one (possibly still uploading), deletes the rest of the run, and prints counts only. Manifests are deleted first, so an interrupted prune leaves ignorable incomplete steps. On B2 a delete hides the current version; add a lifecycle rule to purge hidden versions.

`train` runs one process. The data loader honors `RANK`/`WORLD_SIZE` and DataLoader workers (`--workers`), but `WORLD_SIZE>1` is refused because the model is not wrapped in DDP; see [Kubernetes](../../infrastructure/kubernetes/README.md) for the multi-node path. Nothing prints keys or URLs.

Manifests follow the repository's [shard](../../schemas/shard-manifest.schema.json), [checkpoint](../../schemas/checkpoint-manifest.schema.json) and [model](../../schemas/model-manifest.schema.json) schemas; see the [checkpoint](../../manifests/checkpoint-manifest.example.json) and [model](../../manifests/model-manifest.example.json) examples and the [training run contract](../../docs/training-run-contract.md).

## Key capabilities

| Subcommand | Capabilities | Prefixes |
| --- | --- | --- |
| `publish-dataset` | `listFiles,readFiles,writeFiles` | `datasets/` |
| `train` | `listFiles,readFiles` on datasets; `listFiles,readFiles,writeFiles` on checkpoints | `datasets/`, `checkpoints/` |
| `promote` | `listFiles,readFiles` on checkpoints; `writeFiles` on models; add `writeFileRetentions` for `--lock-days` (Object-Lock-enabled bucket; retention is a real commitment) | `checkpoints/`, `models/` |
| `prune` | `listFiles,deleteFiles` | `checkpoints/` |

Use separate keys per job rather than one key with every capability; a training job should not be able to delete checkpoints or set retention. A B2 key carries at most one name prefix, and this sample reads one key per process, so `train` and `promote` need a key scoped to the bucket without a name prefix (`publish-dataset` and `prune` can use `datasets/` and `checkpoints/` prefixes). With separate buckets in production, give each stage its own bucket-scoped key.

## Container

`docker build -t pytorch-b2-checkpointed-training .` builds a CPU image that runs as a non-root user; credentials come from the environment at run time (`docker run --env-file .env ...` or a Kubernetes Secret). The [Kubernetes Job](../../infrastructure/kubernetes/training-job.yaml) shows retries resuming from B2.

## Tests

`python -m pytest`: config (missing env, endpoint derivation, secret not in repr, user agent), publish (manifest last, idempotent rerun, collision, local corruption and fingerprint refusal), streaming (all 80 train samples once, disjoint and complete rank/worker splits, cache avoids a second GET, tampered shard fails), training (schema-valid checkpoint manifest written last, interrupted run resumes bit-identical to an uninterrupted one, incomplete and corrupt checkpoints, fingerprint and dataset-pin refusals), promote (schema-valid model manifest, server-side copy, idempotent rerun, conflicting content, GOVERNANCE retention), prune, and the CLI end to end. Produced manifests are validated with `jsonschema` against `../../schemas`.
`RUN_B2_LIVE_TESTS=1 python -m pytest tests/test_live.py` runs a publish, interrupted train, resume, promote and prune against your `.env` bucket; use it as an acceptance test for the selected bucket, region and key.
moto covers S3 data operations only; key scoping, Object Lock and lifecycle need live B2.
