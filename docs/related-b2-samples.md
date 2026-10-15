# Related Backblaze B2 sample applications

The [`backblaze-b2-samples`](https://github.com/backblaze-b2-samples) organization already contains applications that cover parts of the model training workflow. This repository links to them instead of rebuilding them. They are unofficial, unsupported samples: review each repository's README, security notes, dependencies, and maintenance status before adapting it.

## Closest workflow matches

| Sample | What it demonstrates | Most relevant here |
|---|---|---|
| [WebDataset Streaming PyTorch Training](https://github.com/backblaze-b2-samples/webdataset-streaming-pytorch-training) | Media packed into shards with a JSON manifest, stored in B2, and streamed into PyTorch with worker-aware shard allocation | [Streaming](../architectures/stream-training-data/architecture.md) |
| [B2 AI and data notebooks](https://github.com/backblaze-b2-samples/notebooks) | Ray Train and Tune checkpoints on B2, PyTorch image classification, and lakeFS dataset versioning | [Checkpointing](../architectures/checkpoint-and-resume-training/architecture.md), [Pinning](../architectures/pin-dataset-versions/architecture.md) |
| [Unsloth LoRA Checkpoint Versioning](https://github.com/backblaze-b2-samples/unsloth-lora-checkpoint-versioning) | Experiment tracking for local LoRA and QLoRA fine-tuning with checkpoints on B2 | [Checkpointing](../architectures/checkpoint-and-resume-training/architecture.md), [Tracking](../architectures/track-experiments-and-register-models/architecture.md) |
| [LoRA Training Studio](https://github.com/backblaze-b2-samples/lora-training-studio) | Browser-based LoRA fine-tuning for image models with every lifecycle artifact on B2 | [Tracking](../architectures/track-experiments-and-register-models/architecture.md) |
| [lakeFS on B2 Quickstart](https://github.com/backblaze-b2-samples/lakefs-on-b2-quickstart) | Docker Compose stack running lakeFS with B2 as the blockstore | [Pinning](../architectures/pin-dataset-versions/architecture.md) |
| [LeRobot B2 Streaming](https://github.com/backblaze-b2-samples/lerobot-b2-streaming) | Streams LeRobot robot datasets from B2 over the S3 API for training without a full download | [Streaming](../architectures/stream-training-data/architecture.md) |
| [Hugging Face Cache Sync](https://github.com/backblaze-b2-samples/hf-cache-sync) | Syncs a Hugging Face model cache to B2 and shares it across machines with LRU eviction | Base-model distribution to training nodes |
| [Iceberg Feature Store](https://github.com/backblaze-b2-samples/iceberg-feature-store) | Versioned feature tables on B2 without a cloud data warehouse | Tabular training features |
| [Eval Vault](https://github.com/backblaze-b2-samples/eval-vault) | Every LLM evaluation stored as an auditable artifact trail on B2 | Evaluation evidence for promotion decisions |

## Operational building blocks

| Sample | Reusable pattern |
|---|---|
| [B2 Event Broker](https://github.com/backblaze-b2-samples/b2-event-broker) | Validates signed B2 event notifications, acknowledges promptly, and fans out to subscribers with retries. Useful for retraining triggers. |
| [Mountpoint for Amazon S3 CSI driver with B2](https://github.com/backblaze-b2-samples/b2-integration-mountpoint-csi) | Mounts a B2 bucket into Kubernetes pods through a file-system interface, for training code that expects paths |

## Upstream sources for training data

Training-ready datasets usually start as releases from an AI data provider workflow. See the [AI data provider reference architectures](https://github.com/backblaze-b2-samples/b2-ai-data-provider-reference-architectures), and samples such as [NeMo Curator Training Data](https://github.com/backblaze-b2-samples/nemo-curator-training-data), [DataComp Image-Text Filtering](https://github.com/backblaze-b2-samples/datacomp-image-text-filtering), and [img2dataset WebDataset Object Storage](https://github.com/backblaze-b2-samples/img2dataset-webdataset-object-storage).

## Selection guidance

- Link the sample that shows the missing implementation pattern, not every adjacent AI application.
- Keep run contracts, key design, retention, and lineage decisions in this repository's architecture documents.
- Treat sample manifests as implementation examples. Validate them against this repository's schemas before relying on them.
- Do not infer production readiness, or third-party support for B2, from a sample that only uses the S3-compatible API.
