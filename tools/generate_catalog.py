#!/usr/bin/env python3
"""Validate the architecture catalog and render its human/agent indexes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "catalog" / "architectures.yaml"
SCHEMA_PATH = ROOT / "schemas" / "architecture-catalog.schema.json"
README_PATH = ROOT / "README.md"
DOC_INDEX_PATH = ROOT / "docs" / "architecture-index.md"
SKILL_INDEX_PATH = ROOT / "skill" / "references" / "architecture-catalog.md"
LLMS_PATH = ROOT / "llms.txt"

ARCH_START = "<!-- catalog:architectures:start -->"
ARCH_END = "<!-- catalog:architectures:end -->"
INT_START = "<!-- catalog:integrations:start -->"
INT_END = "<!-- catalog:integrations:end -->"


def load_catalog() -> dict:
    catalog = yaml.safe_load(CATALOG_PATH.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(catalog), key=lambda error: list(error.path))
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ValueError(f"catalog validation failed:\n{details}")
    validate_local_links(catalog)
    validate_integration_consistency(catalog)
    return catalog


def validate_local_links(catalog: dict) -> None:
    missing: list[str] = []
    for architecture in catalog["architectures"]:
        for value in architecture["links"].values():
            if value and not value.startswith(("http://", "https://")) and not (ROOT / value).exists():
                missing.append(value)
        for value in architecture["validation"]["evidence"]:
            if not value.startswith(("http://", "https://")) and not (ROOT / value).exists():
                missing.append(value)
    if missing:
        raise ValueError("catalog contains missing local paths: " + ", ".join(sorted(set(missing))))


def relative_link(target: str | None, base: Path) -> str | None:
    if target is None or target.startswith(("http://", "https://")):
        return target
    return Path(os.path.relpath(ROOT / target, base)).as_posix()


def markdown_link(label: str, target: str | None, base: Path) -> str:
    return label if not target else f"[{label}]({relative_link(target, base)})"


def workflow_summary(architecture: dict) -> str:
    return " → ".join(architecture["workflow"])


def render_readme_architectures(catalog: dict) -> str:
    rows = [
        "| Architecture | Purpose | Resources |",
        "|---|---|---|",
    ]
    for architecture in catalog["architectures"]:
        links = architecture["links"]
        resources = [
            markdown_link("app", links["application"], ROOT),
            markdown_link("deployment", links["deployment"], ROOT),
        ]
        resources = [resource for resource, target in zip(resources, [links["application"], links["deployment"]]) if target]
        rows.append(
            "| {name} | {summary} | {resources} |".format(
                name=markdown_link(architecture["name"], links["architecture"], ROOT),
                summary=architecture["summary"],
                resources=" · ".join(resources),
            )
        )
    return "\n".join(rows)


SUPPORT_LABELS = {
    "native-b2": "Native B2 backend",
    "s3-compatible-endpoint": "S3-compatible endpoint",
    "candidate": "Candidate, needs acceptance test",
}


def integrations(catalog: dict) -> list[tuple[dict, list[dict]]]:
    """Unique integrations in catalog order, each with every architecture that uses it."""
    result: dict[str, tuple[dict, list[dict]]] = {}
    for architecture in catalog["architectures"]:
        for integration in architecture["integrations"]:
            result.setdefault(integration["id"], (integration, []))[1].append(architecture)
    return list(result.values())


def validate_integration_consistency(catalog: dict) -> None:
    """An integration id must describe the same tool and support level everywhere it appears."""
    seen: dict[str, dict] = {}
    conflicts: list[str] = []
    for architecture in catalog["architectures"]:
        for integration in architecture["integrations"]:
            first = seen.setdefault(integration["id"], integration)
            for field in ("name", "role", "support"):
                if first[field] != integration[field]:
                    conflicts.append(f"{integration['id']}.{field}")
    if conflicts:
        raise ValueError("integration fields differ between architectures: " + ", ".join(sorted(set(conflicts))))


def render_readme_integrations(catalog: dict) -> str:
    rows = [
        "| Tool | Role | Support | How it connects to B2 | Architectures |",
        "|---|---|---|---|---|",
    ]
    for integration, architectures in integrations(catalog):
        used_in = "<br>".join(
            markdown_link(a["name"], a["links"]["architecture"], ROOT) for a in architectures
        )
        rows.append(
            f"| [{integration['name']}]({integration['url']}) | {integration['role']} | "
            f"{SUPPORT_LABELS[integration['support']]} | {integration['connection']} | {used_in} |"
        )
    return "\n".join(rows)


def replace_section(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"missing generated section markers: {start} / {end}")
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body}\n{end}{suffix}"


def render_readme(catalog: dict) -> str:
    text = README_PATH.read_text()
    text = replace_section(text, ARCH_START, ARCH_END, render_readme_architectures(catalog))
    return replace_section(text, INT_START, INT_END, render_readme_integrations(catalog))


def render_docs_index(catalog: dict) -> str:
    base = DOC_INDEX_PATH.parent
    lines = [
        "<!-- Generated by tools/generate_catalog.py; edit catalog/architectures.yaml. -->",
        "# Model training architecture index",
        "",
        "Canonical B2 product documentation: "
        f"<{catalog['canonical_documentation']}>.",
        "",
        "Use this index to choose a workflow. Each linked architecture states its evidence basis and security boundaries. "
        f"All workflows use the shared {markdown_link('training run contract', 'docs/training-run-contract.md', base)}.",
    ]
    for architecture in catalog["architectures"]:
        lines.extend([
            "",
            f"## {markdown_link(architecture['name'], architecture['links']['architecture'], base)}",
            "",
            architecture["summary"],
            "",
            f"- **Audience:** {', '.join(architecture['audience'])}",
            f"- **Problem:** {architecture['problem']}",
            f"- **Workflow:** {workflow_summary(architecture)}",
            f"- **Integrations:** " + ", ".join(
                f"{item['name']} ({SUPPORT_LABELS[item['support']].lower()})" for item in architecture["integrations"]
            ),
            f"- **Expected scale:** {architecture['expected_scale']['profile']}. {architecture['expected_scale']['notes']}",
            f"- **Evidence:** {architecture['validation']['status']} ({architecture['validation']['date']})",
        ])
    lines.extend([
        "",
        "## Machine-readable source",
        "",
        f"The catalog is available as [{CATALOG_PATH.name}]({relative_link('catalog/architectures.yaml', base)}) "
        f"and is validated by [JSON Schema]({relative_link('schemas/architecture-catalog.schema.json', base)}).",
        "",
    ])
    return "\n".join(lines)


def render_skill_index(catalog: dict) -> str:
    lines = [
        "<!-- Generated by tools/generate_catalog.py; edit the repository catalog, then package this skill directory. -->",
        "# Architecture catalog",
        "",
        f"Evidence catalog date: {catalog['updated']}. For changing B2 product details, use "
        f"<{catalog['canonical_documentation']}> as the authority.",
    ]
    for architecture in catalog["architectures"]:
        lines.extend([
            "",
            f"## {architecture['name']}",
            "",
            architecture["summary"],
            "",
            f"**Use when:** {architecture['problem']}",
            "",
            f"**Workflow:** {workflow_summary(architecture)}",
            "",
            f"**B2 capabilities:** {', '.join(architecture['capabilities'])}.",
            "",
            f"**Integrations:** " + "; ".join(
                f"[{item['name']}]({item['url']}) ({SUPPORT_LABELS[item['support']].lower()}) — {item['connection']}"
                for item in architecture["integrations"]
            ) + ".",
            "",
            f"**Scale:** {architecture['expected_scale']['profile']}. {architecture['expected_scale']['notes']}",
            "",
            "**Security:** " + "; ".join(architecture["security_considerations"]) + ".",
            "",
            "**Evidence basis:** " + "; ".join(architecture["validation"]["methods"]) + ".",
        ])
    lines.append("")
    return "\n".join(lines)


def render_llms(catalog: dict) -> str:
    lines = [
        "# B2 Reference Architectures for Model Training",
        "",
        "> Designs, run contracts, and runnable examples for streaming training data, checkpointing, tracking experiments, pinning dataset versions, and orchestrating training pipelines with Backblaze B2.",
        "",
        f"Canonical Backblaze product documentation: {catalog['canonical_documentation']}. "
        "Repository pages are maintained as clean Markdown; product documentation remains authoritative for changing service behavior. "
        "This community-convention file is a navigation aid, not an access-control policy or a search-ranking claim.",
        "",
        "## Start here",
        "",
        "- [Model training summary](docs/model-training.md): Concise workload and capability overview.",
        "- [Architecture index](docs/architecture-index.md): Human-readable index generated from the catalog.",
        "- [Machine-readable catalog](catalog/architectures.yaml): Audiences, workflows, capabilities, integrations, evidence, security, compliance, and resource links.",
        "- [Training run contract](docs/training-run-contract.md): Shard, checkpoint, and model manifests and the lineage that links them.",
        "",
        "## Architectures",
        "",
    ]
    for architecture in catalog["architectures"]:
        lines.append(
            f"- [{architecture['name']}]({architecture['links']['architecture']}): {architecture['summary']}"
        )
    lines.extend([
        "",
        "## Contracts and schemas",
        "",
        "- [Architecture catalog schema](schemas/architecture-catalog.schema.json): JSON Schema for catalog/architectures.yaml.",
        "- [Shard manifest schema](schemas/shard-manifest.schema.json): Immutable training dataset version packaged as WebDataset shards.",
        "- [Checkpoint manifest schema](schemas/checkpoint-manifest.schema.json): Commit marker for a resumable training checkpoint.",
        "- [Model manifest schema](schemas/model-manifest.schema.json): Promoted model with dataset and checkpoint lineage.",
        "- [Synthetic dataset card](testdata/synthetic-shapes/README.md): Shared category dataset used by the main samples.",
        "- [Synthetic Croissant metadata](testdata/synthetic-shapes/croissant.json): MLCommons-compatible discovery and loading metadata.",
        "",
        "## Applications and configuration",
        "",
        "- [PyTorch checkpointed training on B2](applications/pytorch-b2-checkpointed-training/README.md): Stream shards, checkpoint, resume, promote, and prune.",
        "- [MLflow model registry on B2](applications/mlflow-b2-model-registry/README.md): Track runs with a B2 artifact store and register models from B2 events.",
        "- [MLflow tracking server deployment](infrastructure/mlflow/README.md): Compose template for a tracking server with a B2 artifact root.",
        "- [Kubernetes training job](infrastructure/kubernetes/README.md): Job template with scoped B2 credentials and local checkpoint staging.",
        "- [B2 configuration examples](examples/README.md): Lifecycle, encryption, Object Lock, notifications, replication, and keys.",
        "- [Related B2 sample applications](docs/related-b2-samples.md): Existing training, fine-tuning, dataset-versioning, and caching samples from the Backblaze sample organization.",
        "",
        "## Official product documentation",
        "",
        f"- [Backblaze AI and machine learning]({catalog['canonical_documentation']}): Canonical B2 documentation hub.",
        "- [Backblaze B2 documentation](https://www.backblaze.com/docs/cloud-storage): Current product behavior and setup.",
        "- [Backblaze B2 Overdrive](https://www.backblaze.com/cloud-storage/b2-overdrive): High-throughput option for sustained training reads.",
        "- [Backblaze B2 pricing](https://www.backblaze.com/cloud-storage/pricing): Current pricing and egress terms.",
        "",
        "## Optional",
        "",
        "- [Repository agent guide](AGENTS.md): Source hierarchy, generation commands, and operational invariants.",
        "- [Portable agent skill](skill/SKILL.md): Self-contained guidance for B2 model training architecture work.",
        "",
    ])
    return "\n".join(lines)


def expected_outputs(catalog: dict) -> dict[Path, str]:
    return {
        README_PATH: render_readme(catalog),
        DOC_INDEX_PATH: render_docs_index(catalog),
        SKILL_INDEX_PATH: render_skill_index(catalog),
        LLMS_PATH: render_llms(catalog),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    try:
        catalog = load_catalog()
        outputs = expected_outputs(catalog)
    except (OSError, ValueError, jsonschema.SchemaError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    stale = [path for path, content in outputs.items() if not path.exists() or path.read_text() != content]
    if args.check:
        if stale:
            print("stale generated files: " + ", ".join(str(path.relative_to(ROOT)) for path in stale), file=sys.stderr)
            return 1
        print(f"catalog valid; {len(outputs)} generated files current")
        return 0

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
