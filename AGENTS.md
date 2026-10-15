# Agent guide

Use this repository to design or review storage workflows for model training on Backblaze B2: training data streaming, checkpointing and resume, experiment tracking and model registration, dataset version pinning, and pipeline orchestration.

## Sources of truth

1. `catalog/architectures.yaml` is the machine-readable architecture and integration catalog.
2. `schemas/` defines the shard, checkpoint, and model manifest contracts.
3. `architectures/*/architecture.md` contains the detailed design and evidence scope.
4. `applications/` contains executable demonstrations; `examples/` contains configuration templates; `infrastructure/` contains deployment templates.
5. `testdata/synthetic-shapes/` is the shared category dataset; regenerate it only with `tools/build_testdata.py`.
6. Current Backblaze product behavior is authoritative at <https://www.backblaze.com/docs/cloud-storage>. Pricing is authoritative at <https://www.backblaze.com/cloud-storage/pricing>.

Do not describe documentation review or emulator-backed tests as a production deployment. State the evidence basis when it matters to a recommendation. Label each integration by its catalog `support` value; "supports S3" alone is not evidence that a tool accepts a custom endpoint.

## Generated files

Run `python3 tools/generate_catalog.py` after editing the catalog. Do not hand-edit generated sections in `README.md`, `docs/architecture-index.md`, `skill/references/architecture-catalog.md`, or `llms.txt`.

Validate with:

```bash
python3 tools/generate_catalog.py --check
python3 tools/validate_contracts.py
python3 tools/check_markdown_links.py
cd applications/pytorch-b2-checkpointed-training && python3 -m pytest
cd applications/mlflow-b2-model-registry && python3 -m pytest
```

## Operational invariants

- Derive `https://s3.<region>.backblazeb2.com` from the bucket region; do not hardcode a real region.
- Use scoped application keys. Keep master or administrative keys out of runtime services. Training workers read datasets and never write them.
- Write `manifest.json` last under every dataset, checkpoint, and model prefix. Never resume from or register a prefix without one.
- Pin runs to `<dataset>@<version>` plus the fingerprint, and carry the pin into every checkpoint, model, and tracking record.
- Promote by copy, not by reference. Retention is an explicit decision at promotion: governance-mode Object Lock for models that production or audits depend on, with the duration set by deployment policy; an unlocked promotion must be an explicit opt-out, never a silent default.
- Keep tracking, registry, and orchestrator databases, queues, and hot caches off B2.
- Never commit credentials, customer data, webhook secrets, or real presigned URLs.
