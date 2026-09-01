"""Cloudflare R2 object storage client (S3-compatible).

Used to persist checkpoints, config snapshots and eval artifacts. Reads credentials
from environment variables so nothing sensitive is committed to the repo. Both the
project's canonical names and the Modal/AWS-style names are accepted (see r2_env):

    R2_ENDPOINT   (or R2_ENDPOINT_URL)  e.g. https://<accountid>.r2.cloudflarestorage.com
    R2_BUCKET     e.g. thunder-fast
    R2_ACCESS_KEY (or R2_ACCESS_KEY_ID)
    R2_SECRET_KEY (or R2_SECRET_ACCESS_KEY)

We use boto3 against the S3 API rather than the AWS console so the same code works
on Modal, RunPod or a local machine.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable

import boto3

_LOCK = threading.Lock()
_CLIENT = None


def r2_env() -> dict:
    """Read R2 credentials from env.

    Accepts either the project's canonical names (R2_ENDPOINT, R2_BUCKET,
    R2_ACCESS_KEY, R2_SECRET_KEY) or the Modal/AWS-style names that the Modal
    `r2-credentials` secret supplies (R2_ENDPOINT_URL, R2_BUCKET,
    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY). The canonical names win when both
    are set.
    """
    def _first(*names: str) -> str | None:
        for n in names:
            v = os.environ.get(n)
            if v:
                return v
        return None

    pairs = {
        "endpoint": ("R2_ENDPOINT", "R2_ENDPOINT_URL"),
        "bucket": ("R2_BUCKET",),
        "access_key": ("R2_ACCESS_KEY", "R2_ACCESS_KEY_ID"),
        "secret_key": ("R2_SECRET_KEY", "R2_SECRET_ACCESS_KEY"),
    }
    env = {k: _first(*names) for k, names in pairs.items()}
    missing = [k for k, v in env.items() if not v]
    if missing:
        tried = ", ".join("/".join(pairs[k]) for k in missing)
        raise KeyError(f"missing R2 env vars: {tried}")
    return env


def client():
    """Return a thread-safe boto3 S3 client pointed at R2."""
    global _CLIENT
    env = r2_env()
    with _LOCK:
        if _CLIENT is None:
            _CLIENT = boto3.client(
                "s3",
                endpoint_url=env["endpoint"],
                aws_access_key_id=env["access_key"],
                aws_secret_access_key=env["secret_key"],
                region_name="auto",
            )
    return _CLIENT


def bucket() -> str:
    return r2_env()["bucket"]


def upload_file(local_path: Path, key: str) -> str:
    _client = client()
    _client.upload_file(str(local_path), bucket(), key)
    return key


def download_file(key: str, local_path: Path) -> Path:
    _client = client()
    _client.download_file(bucket(), key, str(local_path))
    return local_path


def exists(key: str) -> bool:
    _client = client()
    try:
        _client.head_object(Bucket=bucket(), Key=key)
        return True
    except Exception:  # noqa: BLE001 - boto3 raises ClientError on missing keys
        return False


def list_keys(prefix: str = "") -> Iterable[str]:
    _client = client()
    paginator = _client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def upload_dir(local_dir: Path, prefix: str) -> None:
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            upload_file(path, f"{prefix}{rel}")


def download_dir(prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for key in list_keys(prefix):
        rel = key[len(prefix):]
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        download_file(key, target)
