---
pretty_name: Synthetic Shapes Classification Dataset
license: mit
task_categories:
  - image-classification
tags:
  - synthetic
  - image
  - webdataset
  - mlcroissant
---

# Synthetic Shapes Classification Dataset

96 deterministic 32x32 RGB PNG images of four shape classes (circle, square, triangle, cross), packaged as WebDataset tar shards. It is the shared category dataset for this repository: both main sample applications use it by default, so the same data works for tests, demos, and walkthroughs.

## Intended use

- Exercise shard publishing, streaming, fingerprint pinning, checkpointing, resume, promotion, and experiment tracking on B2.
- Train tiny models on a CPU in seconds, so the full workflow runs anywhere.
- Demonstrate the [shard manifest](manifest.json), the [training run contract](../../docs/training-run-contract.md), and Croissant metadata.

It is not suitable for benchmarking, fairness evaluation, or any claim about model quality.

## Composition and provenance

- Splits: `train` 80 samples in 4 shards, `val` 16 samples in 1 shard. Classes are balanced.
- Sample layout: `<key>.png` (image), `<key>.cls` (class index as text), `<key>.json` (`label`, `split`).
- Generation: [`tools/build_testdata.py`](../../tools/build_testdata.py) with seed 0. Shapes, positions, sizes, and colors are random but deterministic. Tar headers are fixed, so the same Pillow build reproduces identical bytes.
- No real people, customer data, or downloaded material.
- Release identity: `synthetic-shapes@v1`. The fingerprint is in [manifest.json](manifest.json).

Production shards are hundreds of megabytes to a few gigabytes each. These are tiny only to keep the repository small.

## Rights and sensitive data

Distributed under the [MIT license](https://spdx.org/licenses/MIT.html). It contains no personal information, biometric identifiers, or licensed third-party media.

## Machine-readable metadata

- [manifest.json](manifest.json): shard manifest with per-shard SHA-256 and the content fingerprint, written last when published to B2. Validated by [shard-manifest.schema.json](../../schemas/shard-manifest.schema.json).
- [croissant.json](croissant.json): Croissant JSON-LD for discovery and loading.
