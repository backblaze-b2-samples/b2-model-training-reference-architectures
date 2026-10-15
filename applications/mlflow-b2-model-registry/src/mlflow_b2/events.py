"""B2 event-notification signature check and model-manifest key matching."""
from __future__ import annotations

import hashlib
import hmac
import re

SIGNATURE_HEADER = "X-Bz-Event-Notification-Signature"  # value: "v1=<hex hmac-sha256 of body>"
MANIFEST_KEY = re.compile(r"^models/([a-z0-9._-]+)/([a-z0-9._-]+)/manifest\.json$")


def verify_signature(secret: str, body: bytes, header: str) -> bool:
    expected = "v1=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header or "")


def created_keys(payload: dict, prefix: str = "", bucket: str | None = None) -> list[str]:
    """Object names from ObjectCreated events under a prefix (and in the bucket, if given)."""
    return [e["objectName"] for e in payload.get("events", [])
            if e.get("eventType", "").startswith("b2:ObjectCreated:")
            and e.get("objectName", "").startswith(prefix)
            and (bucket is None or e.get("bucketName", bucket) == bucket)]


def manifest_keys(payload: dict, bucket: str | None = None) -> list[str]:
    return list(dict.fromkeys(k for k in created_keys(payload, "models/", bucket)
                              if MANIFEST_KEY.match(k)))
