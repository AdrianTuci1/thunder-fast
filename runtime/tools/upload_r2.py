"""Upload build artifacts to Cloudflare R2 (S3-compatible).

Credentials come from the environment (never committed, never hardcoded):
    R2_ACCOUNT_ID, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET, (optionally) R2_PREFIX

Usage:
    python3 tools/upload_r2.py <local_path> <object_key>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3


def main() -> None:
    local = Path(sys.argv[1])
    key = sys.argv[2]

    if not local.exists():
        sys.exit(f"missing artifact: {local}")

    account = os.environ["R2_ACCOUNT_ID"]
    prefix = os.environ.get("R2_PREFIX", "thunder-fast/runtime")
    endpoint = f"https://{account}.r2.cloudflarestorage.com"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        region_name="auto",
    )
    object_key = f"{prefix}/{key}"

    if local.is_dir():
        for f in sorted(local.rglob("*")):
            if f.is_file():
                client.upload_file(str(f), os.environ["R2_BUCKET"], f"{object_key}/{f.relative_to(local)}")
                print(f"uploaded {f}")
    else:
        client.upload_file(str(local), os.environ["R2_BUCKET"], object_key)
        print(f"uploaded {local} -> {object_key}")


if __name__ == "__main__":
    main()
