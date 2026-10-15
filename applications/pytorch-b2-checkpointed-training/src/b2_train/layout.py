"""Object-key layout. Production splits these prefixes into separate buckets (see README)."""
from __future__ import annotations

import re

MANIFEST_NAME = "manifest.json"  # commit marker, always written last
STEP_RE = re.compile(r"^step-(\d{8})/")


def dataset_prefix(dataset: str, version: str) -> str:
    return f"datasets/{dataset}/{version}/"


def run_prefix(run: str) -> str:
    return f"checkpoints/{run}/"


def checkpoint_prefix(run: str, step: int) -> str:
    return f"{run_prefix(run)}step-{step:08d}/"


def model_prefix(model: str, version: str) -> str:
    return f"models/{model}/{version}/"
