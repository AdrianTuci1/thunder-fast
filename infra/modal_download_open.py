"""Download a masked-diffusion checkpoint (with its self-contained modeling code) into the Modal volume.

Pulls the full repo (self-contained MDM `modeling_qwen2.py`, tokenizer with the `<M>` mask token,
`model.safetensors`) into the persistent `thunder-checkpoints` volume so inference can run offline
from the volume, without re-downloading from the Hub.

Usage (from the repo root):
    TF_MODEL=staticlabs/dlm-code0.6b-exp modal run infra/modal_download_open.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = "/vol/checkpoints"
REPO_REMOTE = "/repo"
DEST = f"{CHECKPOINT_DIR}/thunder-dlm-0.6b"


def _ignore(p: Path) -> bool:
    name = p.name
    if name.startswith("."):
        return True
    if name in {"checkpoints", "__pycache__", "node_modules", ".venv", "venv"}:
        return True
    if name.endswith((".pyc", ".pt", ".log", ".png", ".pdf", ".sqlite", ".sqlite.bak", ".bin")):
        return True
    if p.suffix in {".yaml", ".yml"}:
        return True
    return False


app = modal.App("thunder-fast-download-open")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .apt_install("git")
    .add_local_dir(REPO, remote_path=REPO_REMOTE, copy=True, ignore=_ignore)
)

# Not strictly required for a public repo, but the HF token avoids rate limits and makes the
# download robust if the repo ever becomes gated. Create once: modal secret create hf HF_TOKEN=...
hf_secret = modal.Secret.from_name("hf")

checkpoints_volume = modal.Volume.from_name("thunder-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={CHECKPOINT_DIR: checkpoints_volume},
    timeout=7200,
    memory=32768,
)
def download(model_id: str, dest: str) -> str:
    import os

    from huggingface_hub import snapshot_download

    os.makedirs(dest, exist_ok=True)
    path = snapshot_download(model_id, local_dir=dest, max_workers=8)
    files = sorted(os.listdir(dest))
    print(f"[download] repo: {model_id} -> {path}", flush=True)
    for f in files:
        print(f"  {f}", flush=True)
    # Flush the writes to the volume explicitly so a subsequent inference run sees the files.
    checkpoints_volume.commit()
    return str(path)


@app.local_entrypoint()
def run():
    model_id = os.environ.get("TF_MODEL", "")
    if not model_id:
        raise ValueError("set TF_MODEL=<hf/repo-id> to download a masked-diffusion checkpoint")
    dest = os.environ.get("TF_DEST", DEST)
    path = download.remote(model_id, dest)
    print(f"[download] done: {path}", flush=True)
