"""B2 access over the S3-compatible API: idempotent uploads, verified downloads, server-side copy."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import Config
from .manifest import sha256_bytes, sha256_file

MB = 1 << 20
USER_AGENT = "b2ai-pytorch-b2-checkpointed-training"


class StorageError(Exception):
    pass


class Collision(StorageError):
    """Same key already holds different content and overwrite is not allowed."""


def make_client(cfg: Config):
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        region_name=cfg.region,
        aws_access_key_id=cfg.key_id,
        aws_secret_access_key=cfg.key,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},  # backoff on 429/503
            # SDK default CRC checksums are not needed by B2; send them only when required.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            user_agent_extra=USER_AGENT,
        ),
    )


def _code(err: ClientError) -> str:
    return err.response.get("Error", {}).get("Code", "")


def _missing(err: ClientError) -> bool:
    return _code(err) in ("404", "NoSuchKey", "NotFound")


def _transfer(multipart_mb: int) -> TransferConfig:
    return TransferConfig(multipart_threshold=multipart_mb * MB,
                          multipart_chunksize=max(multipart_mb, 5) * MB)


def check_bucket(client, bucket: str) -> None:
    """Fail fast with an actionable message. Bucket names are globally unique on B2."""
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as e:
        if _code(e) in ("403", "AccessDenied"):
            raise StorageError(
                f"bucket '{bucket}' is not accessible: it belongs to another account, "
                "or the key is not scoped to it"
            ) from e
        if _code(e) in ("404", "NoSuchBucket"):
            raise StorageError(f"bucket '{bucket}' does not exist; create it first") from e
        raise


def head(client, bucket: str, key: str) -> dict | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if _missing(e):
            return None
        raise


def stored_sha(client, bucket: str, key: str) -> str | None:
    h = head(client, bucket, key)
    return None if h is None else h["Metadata"].get("sha256", "")


def lock_until(lock_days: int | None) -> dt.datetime | None:
    if not lock_days:
        return None
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0) + dt.timedelta(days=lock_days)


def _lock_args(lock_days: int | None, until: dt.datetime | None = None) -> dict:
    until = until or lock_until(lock_days)
    return {"ObjectLockMode": "GOVERNANCE", "ObjectLockRetainUntilDate": until} if until else {}


def _guard(existing: str | None, sha: str, key: str, overwrite: bool) -> bool:
    """True if an upload is needed. Identical content is skipped so reruns add no versions."""
    if existing == sha:
        return False
    if existing is not None and not overwrite:
        raise Collision(f"{key} already exists with different content")
    return True


def put_file(client, bucket, key, path: Path, *, overwrite=True, lock_days=None,
             multipart_mb=64, sha=None) -> str:
    sha = sha or sha256_file(path)
    if not _guard(stored_sha(client, bucket, key), sha, key, overwrite):
        return "skipped"
    client.upload_file(  # multipart above the threshold
        str(path), bucket, key,
        ExtraArgs={"Metadata": {"sha256": sha}, **_lock_args(lock_days)},
        Config=_transfer(multipart_mb),
    )
    return "uploaded"


def put_bytes(client, bucket, key, data: bytes, *, overwrite=True, lock_days=None, until=None) -> str:
    sha = sha256_bytes(data)
    if not _guard(stored_sha(client, bucket, key), sha, key, overwrite):
        return "skipped"
    lock = _lock_args(lock_days, until)
    if lock:  # Object Lock writes must carry an integrity header
        lock["ContentMD5"] = base64.b64encode(hashlib.md5(data).digest()).decode()
    client.put_object(Bucket=bucket, Key=key, Body=data, Metadata={"sha256": sha},
                      ContentType="application/json", **lock)
    return "uploaded"


def upload_tree(client, bucket, items: list[tuple[str, Path]], *, workers=4, **kw) -> dict[str, int]:
    """Upload (key, path) pairs with a bounded pool; parallelism is per file, not per part."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda kp: put_file(client, bucket, kp[0], kp[1], **kw), items))
    return {s: results.count(s) for s in ("uploaded", "skipped")}


def verify(client, bucket, prefix: str, files: list[dict]) -> list[str]:
    """Compare remote size and stored checksum against manifest entries."""
    errors = []
    for f in files:
        h = head(client, bucket, prefix + f["path"])
        if h is None:
            errors.append(f"{f['path']}: missing")
        elif h["ContentLength"] != f["size"] or h["Metadata"].get("sha256") != f["sha256"]:
            errors.append(f"{f['path']}: remote content differs from manifest")
    return errors


def load_json(client, bucket, key: str, what: str) -> dict:
    try:
        body = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if _missing(e):
            raise StorageError(f"{what} not found (manifest missing)") from e
        raise
    return json.loads(body["Body"].read())


def list_keys(client, bucket, prefix: str) -> list[str]:
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return [o["Key"] for page in pages for o in page.get("Contents", [])]


def download(client, bucket, key: str, dest: Path, size: int, sha: str) -> None:
    """Download and verify; a mismatch never leaves a file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    if dest.stat().st_size != size or sha256_file(dest) != sha:
        dest.unlink()
        raise StorageError(f"{key}: downloaded content differs from manifest")


def verify_object(client, bucket, key: str, f: dict) -> str:
    """Hash the stored bytes against a manifest entry and return the ETag they were read at.

    Metadata alone is not trusted: it is whatever the writer claimed. Reading every byte is the price of
    putting a checksum on a promoted model.
    """
    h = head(client, bucket, key)
    if h is None:
        raise StorageError(f"{key}: missing")
    if h["ContentLength"] != f["size"] or h["Metadata"].get("sha256") != f["sha256"]:
        raise StorageError(f"{key}: size or stored sha256 differs from manifest")
    body = client.get_object(Bucket=bucket, Key=key, IfMatch=h["ETag"])["Body"]
    digest = hashlib.sha256()
    for chunk in iter(lambda: body.read(MB), b""):
        digest.update(chunk)
    if digest.hexdigest() != f["sha256"]:
        raise StorageError(f"{key}: content differs from manifest")
    return h["ETag"]


def copy(client, bucket, src_key: str, dst_key: str, sha: str, *, etag: str, until=None,
         multipart_mb=64) -> None:
    """Server-side copy of the verified source version (If-Match on its ETag); switches to
    UploadPartCopy above the threshold, since CopyObject caps at 5 GB."""
    client.copy({"Bucket": bucket, "Key": src_key}, bucket, dst_key,
                ExtraArgs={"MetadataDirective": "REPLACE", "Metadata": {"sha256": sha},
                           "CopySourceIfMatch": etag, **_lock_args(None, until)},
                Config=_transfer(multipart_mb))


def retention(h: dict) -> tuple[str | None, dt.datetime | None]:
    return h.get("ObjectLockMode"), h.get("ObjectLockRetainUntilDate")


def ensure_retention(client, bucket, key: str, until: dt.datetime) -> None:
    """Governance retention can be set or extended without bypassGovernance; never shortened here."""
    mode, current = retention(head(client, bucket, key) or {})
    if mode == "GOVERNANCE" and current and current >= until:
        return
    client.put_object_retention(Bucket=bucket, Key=key,
                                Retention={"Mode": "GOVERNANCE", "RetainUntilDate": until})


def retention_errors(client, bucket, keys: list[str], until: dt.datetime) -> list[str]:
    errors = []
    for key in keys:
        mode, current = retention(head(client, bucket, key) or {})
        if mode != "GOVERNANCE" or current is None or current < until:
            errors.append(f"{key}: retention is {mode or 'none'}, expected GOVERNANCE until {until:%Y-%m-%d}")
    return errors


def delete_keys(client, bucket, keys: list[str]) -> int:
    """On B2 this hides the current version; a lifecycle rule purges hidden versions later."""
    for i in range(0, len(keys), 1000):
        resp = client.delete_objects(Bucket=bucket, Delete={
            "Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
        if resp.get("Errors"):
            raise StorageError(f"delete failed for {len(resp['Errors'])} objects "
                               f"({resp['Errors'][0].get('Code')})")
    return len(keys)
