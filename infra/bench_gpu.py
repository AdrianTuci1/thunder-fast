"""GPU throughput / VRAM benchmark for thunder-fast on an H100.

Loads the real DiffusionLM (Qwen3-0.6B) and runs forward+backward at several batch
sizes, reporting per-micro-batch time, tokens/sec, GPU utilization (nvidia-smi) and peak
VRAM. Use it to pick a better `batch_size_seq` than the current 32.

One-shot diagnostic (transient container that exits when done):
    modal run infra/bench_gpu.py
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REPO_REMOTE = "/repo"


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


app = modal.App("thunder-fast-bench")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .apt_install("git")
    .add_local_dir(REPO, remote_path=REPO_REMOTE, copy=True, ignore=_ignore)
)

hf_secret = modal.Secret.from_name("hf")


@app.function(image=image, gpu="H100", secrets=[hf_secret], timeout=3600, memory=32768)
def bench():
    import subprocess
    import sys
    import time

    import torch

    sys.path.insert(0, REPO_REMOTE)
    from src.train.diffusion import ContinuousDiffusion
    from src.train.model import DiffusionLM

    model = DiffusionLM("Qwen/Qwen3-0.6B").to("cuda")
    diff = ContinuousDiffusion(
        hidden_size=model.hidden_size,
        schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        train_steps=24,
        infer_steps=24,
        prediction="x0",
        mask_ratio=0.25,
    )
    model.set_train_ctx()
    model.train()
    L = 256

    def gpu_info():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            return out.decode().strip().replace("\n", " | ")
        except Exception:  # noqa: BLE001
            return "n/a"

    print(f"[bench] device={torch.cuda.get_device_name(0)}", flush=True)

    def step(batch):
        # Fresh inputs each step (like a new micro-batch) -> independent autograd graph.
        inp = torch.randint(0, model.vocab_size, (batch, L), device="cuda")
        x0 = model.embed_tokens(inp)
        t = torch.rand(batch, device="cuda")
        loss = diff.training_loss(model, x0, model.mask_embedding.data, t)
        loss.backward()
        model.zero_grad()

    for batch in [16, 32, 48, 64, 96, 128]:
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):  # warmup
                step(batch)
            torch.cuda.synchronize()

            n = 5
            t0 = time.time()
            util = ""
            for i in range(n):
                step(batch)
                if i == 2:
                    util = gpu_info()  # sample util mid-burst (rough)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / n
            tok = batch * L
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"[bench] batch={batch:4d} tok/micro={tok:6d} fwdbwd={dt:.4f}s "
                f"tokens/s={tok / dt:9.0f} peakVRAM={peak:5.2f}GB nvidia=[{util}]",
                flush=True,
            )
        except RuntimeError as e:  # noqa: BLE001
            print(f"[bench] batch={batch}: {type(e).__name__}: {e}", flush=True)
            break


@app.local_entrypoint()
def run():
    bench.remote()
