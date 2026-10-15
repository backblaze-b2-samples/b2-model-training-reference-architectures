"""B2 reads over the S3-compatible API, used by the event receiver."""
from __future__ import annotations

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import Config

USER_AGENT = "b2ai-mlflow-b2-model-registry"
MAX_MANIFEST_BYTES = 1 << 20


class StorageError(Exception):
    pass


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
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            user_agent_extra=USER_AGENT,
        ),
    )


def _code(err: ClientError) -> str:
    return err.response.get("Error", {}).get("Code", "")


def get_bytes(client, bucket: str, key: str) -> bytes:
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if _code(e) in ("404", "NoSuchKey"):
            raise StorageError(f"{key} not found") from e
        raise
    if obj["ContentLength"] > MAX_MANIFEST_BYTES:
        raise StorageError(f"{key} is larger than {MAX_MANIFEST_BYTES} bytes")
    return obj["Body"].read()


def missing_files(client, bucket: str, prefix: str, files: list[dict]) -> list[str]:
    """Files listed in a manifest that are absent or a different size on B2."""
    errors = []
    for f in files:
        try:
            head = client.head_object(Bucket=bucket, Key=prefix + f["path"])
        except ClientError as e:
            errors.append(f"{f['path']}: {_code(e)}")
            continue
        if head["ContentLength"] != f["size"]:
            errors.append(f"{f['path']}: size differs from manifest")
    return errors
