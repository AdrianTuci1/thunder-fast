"""Launch a thunder-fast training job on RunPod via the RunPod Python API.

Requires the `runpod` package and RUNPOD_API_KEY (pod creds) + the R2 env vars.
This is a convenience wrapper for the serverless/jobs API; a plain interactive Pod
using infra/Dockerfile.runpod also works.

Usage:
    RUNPOD_API_KEY=... \
    R2_ENDPOINT=... R2_BUCKET=... R2_ACCESS_KEY=... R2_SECRET_KEY=... \
    R2_ENDPOINT=... python infra/runpod_launch.py --image <image> --gpu A100
"""

from __future__ import annotations

import argparse
import os

import runpod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="container image (see infra/Dockerfile.runpod)")
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--gpu-count", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--upload-r2", action="store_true", help="push checkpoints to R2 during training")
    ap.add_argument("--timeout", type=int, default=86400)
    args = ap.parse_args()

    r2_env = {
        "R2_ENDPOINT": os.environ.get("R2_ENDPOINT"),
        "R2_BUCKET": os.environ.get("R2_BUCKET"),
        "R2_ACCESS_KEY": os.environ.get("R2_ACCESS_KEY"),
        "R2_SECRET_KEY": os.environ.get("R2_SECRET_KEY"),
    }
    if not all(r2_env.values()):
        raise SystemExit("All R2_* env vars are required.")

    cmd = ["python", "src/train/train.py", "--config", "config/train_config.yaml"]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.upload_r2:
        cmd += ["--upload-r2"]

    job = runpod.create_job(
        image=args.image,
        gpu_type=args.gpu,
        gpu_count=args.gpu_count,
        env=r2_env,
        command=" ".join(cmd),
        timeout=args.timeout,
    )
    print(f"RunPod job submitted: {job}")


if __name__ == "__main__":
    main()
