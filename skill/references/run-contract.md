# Training run contract

Use three manifests, each written last under its prefix:

1. Shard manifest, `datasets/<dataset>/<version>/manifest.json`: dataset id `<dataset>@<version>`, format (`webdataset`), SPDX license, provenance class and optional upstream release, classes, sample keys, splits with shard lists and sample counts, statistics, per-shard size, SHA-256, and sample count, and a content fingerprint.
2. Checkpoint manifest, `checkpoints/<run>/step-<step:08d>/manifest.json`: run id, step, epoch, format (`torch.distributed.checkpoint`, `torch.save`, or `safetensors`), framework and version, dataset pin, world size, metrics, per-file size and SHA-256, fingerprint, and creation time.
3. Model manifest, `models/<model>/<version>/manifest.json`: model id `<model>@<version>`, format, framework, source run, step, and checkpoint fingerprint, dataset pin, metrics, per-file size and SHA-256, fingerprint, and retention mode.

The dataset pin is `{"id": "<dataset>@<version>", "fingerprint": "<sha256>"}`. The fingerprint is SHA-256 over the sorted lines `path:sha256`, with timestamps excluded, so identical content always reproduces it.

Upload data objects first, verify remote sizes and stored checksums, then write the manifest. Never overwrite a version: identical content is a no-op, and different content is a collision. Training stops on a fingerprint mismatch. Resume uses the newest step whose manifest exists and whose files match. Promotion copies files so pruning can never remove a model.

Store each object's SHA-256 in user metadata and keep other metadata small, because Object Lock buckets have a lower per-version metadata limit. Manifests never contain credentials or presigned URLs.
