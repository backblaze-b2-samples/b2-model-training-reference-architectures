# Contributing

The architecture catalog is the extension point. Edit `catalog/architectures.yaml`, then run `python3 tools/generate_catalog.py`; do not hand-edit generated README tables or indexes.

- New architecture: copy an existing `architectures/*/architecture.md`; name it by workflow, not by tool. Add its audience, problem, workflow, scale, validation evidence, security, compliance, and links to the catalog. Reuse `examples/`, `testdata/`, `manifests/`, and `schemas/`.
- New integration: add it under the relevant catalog architecture with its `support` level, connection method, evidence date, and authoritative documentation or source link. Use `candidate` until the B2 path has been tested. Add an architecture only if the tool materially changes the data flow or security boundary.
- New application: `applications/<name>/` with a README (including "B2's role"), `.env.example`, and tests; default to the shared `testdata/synthetic-shapes` dataset; add a row to the sample applications table and a CI step.
- Regenerate `README.md`, `docs/architecture-index.md`, `skill/references/architecture-catalog.md`, and `llms.txt` in the same change.

Evidence rules: attribute to a tool only what its documentation or code shows; label B2-side patterns as such; never hardcode a region or endpoint; link B2 pricing instead of quoting numbers; mark unverified claims.
Security: no real keys anywhere; secrets in gitignored `0600` `.env` files; scoped keys, not the master key, unless provisioning requires it (say so).
Tests: `python3 tools/generate_catalog.py --check`, `python3 tools/validate_contracts.py`, `python3 tools/check_markdown_links.py`, and `python -m pytest` in each application. Use moto for S3 data operations and environment-gated acceptance tests for Object Lock, key scoping, lifecycle, and event delivery.
