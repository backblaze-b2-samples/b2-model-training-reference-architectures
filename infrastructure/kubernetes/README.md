# Kubernetes: checkpointed training Job

Runs [`b2-train train`](../../applications/pytorch-b2-checkpointed-training/README.md) as a Kubernetes Job that survives pod loss.
The recovery mechanism is `backoffLimit` plus resume-from-manifest: a failed or evicted pod is replaced, the new pod lists `checkpoints/<run>/`, loads the newest step that has a `manifest.json`, verifies every file's sha256 and continues.
Nothing on the node is needed to recover; node-local disk is only staging and cache.

| File | Purpose |
| --- | --- |
| [`training-job.yaml`](training-job.yaml) | Job with `backoffLimit`, a `podFailurePolicy`, credentials via `envFrom`, and an `emptyDir` scratch volume for `--staging-dir` and `--cache-dir` |
| [`secret.example.yaml`](secret.example.yaml) | Shape of the credentials Secret; placeholders only |

## Run

```bash
docker build -t <registry>/pytorch-b2-checkpointed-training:<tag> ../../applications/pytorch-b2-checkpointed-training
docker push <registry>/pytorch-b2-checkpointed-training:<tag>
kubectl create secret generic b2-training-credentials --from-env-file=.env   # never commit the filled-in Secret
# edit image and --expect-fingerprint (from the published shard manifest) in training-job.yaml
kubectl apply -f training-job.yaml
kubectl logs -f job/shapes-run-001
```

To see recovery, delete the running pod mid-training: the replacement logs `resumed shapes-run-001 from step N`.
Keep `--run-id` stable across retries; a new run id starts a new lineage.

## Failure handling

- `backoffLimit: 6` retries crashes and transient B2 errors (exit 1). Each retry costs at most the steps since the last committed checkpoint, so set `--checkpoint-every` from how much recompute you can afford.
- `podFailurePolicy` fails the Job immediately on exit 2 (fingerprint or dataset mismatch, invalid names), because retrying cannot fix input, and does not count preemption or node drain (`DisruptionTarget`) against the limit.
- A pod killed during upload leaves files without a manifest. Resume ignores that step; the next save of the same step overwrites the files, and `b2-train prune` removes stale ones.

## Scratch storage

`emptyDir` is node-local and disappears with the pod, which is fine: B2 holds the durable copy and the cache is keyed by shard sha256.
Size it for one staged checkpoint plus the shards you want cached. On GPU nodes with local NVMe, use a local PersistentVolume or a generic ephemeral volume backed by it for higher throughput.

## Credentials and key scoping

The Secret is injected with `envFrom`, so the container reads the same `B2_*` variables as the CLI and nothing is baked into the image.
Use an application key restricted to the training bucket with `listFiles,readFiles,writeFiles`; give `promote` and `prune` their own keys (`promote` adds `writeFileRetentions` and `readFileRetentions`; `prune` needs only `listFiles,writeFiles` on `checkpoints/`, because a delete by name on B2 hides the file) instead of widening the training key.
When datasets, checkpoints and models live in separate buckets, scope each key to the buckets that job touches and make the dataset access read-only.

## Single-node vs multi-node

This Job is single-node, single-process: one pod trains and writes checkpoints.
The data loader already splits shards by `RANK`/`WORLD_SIZE` and by DataLoader worker, but `b2-train train` refuses `WORLD_SIZE>1` because it does not wrap the model in DistributedDataParallel.
For multi-node training, launch with `torchrun` (or the Kubeflow Training Operator `PyTorchJob`, which sets the rank environment for each pod), add a process group and DDP or FSDP, and have each rank save its `torch.distributed.checkpoint` shard, with one rank writing the manifest after all uploads are verified.
The commit rule stays the same: a step is complete only when its manifest exists.
