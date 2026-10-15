#!/usr/bin/env python3
"""Build the shared synthetic-shapes dataset as WebDataset tar shards plus a shard manifest.

Deterministic for a given Pillow build: fixed seed, fixed tar headers, sorted members.
Run from the repository root: python3 tools/build_testdata.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "testdata" / "synthetic-shapes"
CLASSES = ["circle", "square", "triangle", "cross"]
SIZE = 32
SPLITS = {"train": 80, "val": 16}
SAMPLES_PER_SHARD = 20
DATASET = "synthetic-shapes"
VERSION = "v1"


def draw(label: str, rng: random.Random) -> bytes:
    bg = tuple(rng.randint(0, 60) for _ in range(3))
    fg = tuple(rng.randint(150, 255) for _ in range(3))
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)
    r = rng.randint(7, 12)
    cx, cy = rng.randint(r + 1, SIZE - r - 2), rng.randint(r + 1, SIZE - r - 2)
    if label == "circle":
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fg)
    elif label == "square":
        d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fg)
    elif label == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
    else:
        w = max(2, r // 3)
        d.rectangle((cx - r, cy - w, cx + r, cy + w), fill=fg)
        d.rectangle((cx - w, cy - r, cx + w, cy + r), fill=fg)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=9)
    return buf.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(files: list[dict]) -> str:
    lines = "\n".join(f"{f['path']}:{f['sha256']}" for f in sorted(files, key=lambda f: f["path"]))
    return sha256_bytes(lines.encode())


def build(out: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    files, splits = [], {}
    for split, count in SPLITS.items():
        samples = [(f"{split}-{i:05d}", CLASSES[i % len(CLASSES)]) for i in range(count)]
        shards = []
        for s in range(0, count, SAMPLES_PER_SHARD):
            rel = f"shards/{split}-{s // SAMPLES_PER_SHARD:06d}.tar"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                chunk = samples[s:s + SAMPLES_PER_SHARD]
                for key, label in chunk:
                    _add(tar, f"{key}.png", draw(label, rng))
                    _add(tar, f"{key}.cls", str(CLASSES.index(label)).encode())
                    _add(tar, f"{key}.json", json.dumps({"label": label, "split": split},
                                                        sort_keys=True).encode())
            data = buf.getvalue()
            (out / rel).write_bytes(data)
            files.append({"path": rel, "size": len(data), "sha256": sha256_bytes(data),
                          "samples": len(chunk), "media_type": "application/x-tar"})
            shards.append(rel)
        splits[split] = {"shards": shards, "samples": count}
    manifest = {
        "schema_version": "1.0.0",
        "id": f"{DATASET}@{VERSION}",
        "dataset": DATASET,
        "version": VERSION,
        "format": "webdataset",
        "license": "MIT",
        "provenance": {"class": "synthetic", "generator": f"tools/build_testdata.py seed={seed}"},
        "classes": CLASSES,
        "sample_keys": {"image": "png", "label": "cls", "metadata": "json"},
        "splits": splits,
        "statistics": {
            "file_count": len(files),
            "total_bytes": sum(f["size"] for f in files),
            "sample_count": sum(v["samples"] for v in splits.values()),
        },
        "files": files,
        "fingerprint": fingerprint(files),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "croissant.json").write_text(json.dumps(croissant(manifest), indent=2) + "\n")
    return manifest


def croissant(manifest: dict) -> dict:
    """Croissant 1.0 description of the shards; checksums come from the shard manifest."""
    shards = [{
        "@type": "cr:FileObject",
        "@id": Path(f["path"]).stem,
        "name": Path(f["path"]).name,
        "contentUrl": f["path"],
        "contentSize": f"{f['size']} B",
        "encodingFormat": "application/x-tar",
        "sha256": f["sha256"],
    } for f in manifest["files"]]
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
            "sc": "https://schema.org/",
        },
        "@type": "sc:Dataset",
        "name": "Synthetic Shapes Classification Dataset",
        "description": (f"{manifest['statistics']['sample_count']} deterministic 32x32 PNG images of "
                        f"{len(manifest['classes'])} shape classes in WebDataset tar shards, for testing "
                        "B2 training-data streaming, checkpointing, and experiment tracking."),
        "dct:conformsTo": "http://mlcommons.org/croissant/1.0",
        "version": manifest["version"],
        "license": "https://spdx.org/licenses/MIT.html",
        "url": "https://www.backblaze.com/docs/cloud-storage-ai-machine-learning",
        "creator": {"@type": "sc:Organization", "name": "Backblaze, Inc.", "url": "https://www.backblaze.com/"},
        "datePublished": "2026-10-14",
        "keywords": ["synthetic", "image classification", "WebDataset", "Backblaze B2"],
        "distribution": shards + [{
            "@type": "cr:FileSet",
            "@id": "samples",
            "name": "samples",
            "containedIn": [{"@id": s["@id"]} for s in shards],
            "encodingFormat": "image/png",
            "includes": "*.png",
        }],
        "recordSet": [{
            "@type": "cr:RecordSet",
            "@id": "images",
            "name": "images",
            "field": [
                {"@type": "cr:Field", "@id": "images/image", "name": "image", "dataType": "sc:ImageObject",
                 "source": {"fileSet": {"@id": "samples"}, "extract": {"fileProperty": "content"}}},
                {"@type": "cr:Field", "@id": "images/key", "name": "key", "dataType": "sc:Text",
                 "source": {"fileSet": {"@id": "samples"}, "extract": {"fileProperty": "filename"}}},
            ],
        }],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    m = build(a.out, a.seed)
    print(f"wrote {m['statistics']['file_count']} shards, {m['statistics']['sample_count']} samples, "
          f"fingerprint {m['fingerprint'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
