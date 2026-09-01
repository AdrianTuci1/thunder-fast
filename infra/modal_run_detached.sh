#!/usr/bin/env bash
# Launch thunder-fast training as a truly detached, server-side Modal job.
#
# Strategy: `modal run` ephemeral apps are tied to the lifetime of the local `modal
# run` client, so the old approach (keep a detached client alive) died whenever the
# client was reaped. Instead we:
#   1. `modal deploy` a NAMED app ("thunder-fast-train") -> persistent, monitorable,
#      and reused instead of piling up ephemeral apps in the dashboard.
#   2. `.spawn()` the train function -> it runs on Modal's infra independent of this
#      host process, so the launcher can exit immediately.
#
# Usage (from repo root), optionally override env:
#   TF_GPU=H100 ./infra/modal_run_detached.sh
#
# Notes:
#   - Do NOT export TF_MAX_STEPS: the run stops on `training.max_time_hours` (config) and
#     saves a final checkpoint, so a 4h run is bounded by wall-clock, not step count.
#   - The checkpoint volume is created if missing (create_if_missing), so deleting the old
#     `thunder-checkpoints` volume before launching is safe; the run recreates it fresh.

set -euo pipefail
cd "$(dirname "$0")/.."

export TF_CONFIG="${TF_CONFIG:-config/train_config.yaml}"
export TF_GPU="${TF_GPU:-H100}"

MODAL_PY="$(ls -d ~/.local/pipx/venvs/modal/lib/python*/site-packages 2>/dev/null | head -1)"
if [ -z "$MODAL_PY" ]; then
  echo "error: modal python site-packages not found" >&2
  exit 1
fi

echo "==> deploying app 'thunder-fast-train' (cache hit on unchanged source)"
modal deploy infra/modal_train.py

echo "==> spawning detached training"
MODAL_PY="$MODAL_PY" TF_CONFIG="$TF_CONFIG" python3 - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ["MODAL_PY"])

import modal

config_text = Path(os.environ["TF_CONFIG"]).read_text(encoding="utf-8")

def _int_env(name):
    v = os.environ.get(name)
    return int(v) if v else None
max_steps = _int_env("TF_MAX_STEPS")
ckpt_every_steps = _int_env("TF_CKPT_EVERY_STEPS")

train = modal.Function.from_name("thunder-fast-train", "train")
call = train.spawn(config_text, max_steps=max_steps, ckpt_every_steps=ckpt_every_steps)
print(f"[launch] spawned training call: {call.object_id}")
print("[launch] the run continues on Modal; this host process may exit now.")
PY

echo "==> done. Monitor with:"
echo "   modal app logs thunder-fast-train"
echo "   (plus Weights & Biases run under project 'thunder-fast')"
