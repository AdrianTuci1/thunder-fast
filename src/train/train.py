"""Adaptation training loop: autoregressive checkpoint -> continuous diffusion.

Runs the DiffuLLaMA-style denoising objective on GPU (Modal or RunPod), checkpoints
periodically, and uploads each checkpoint to Cloudflare R2 so the run is resumable.
Start from a checkpoint when one already exists in R2.

Environment (set by the runner): R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY, R2_SECRET_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# Make the repo root importable regardless of how this module is launched (directly, or
# imported from the Modal container). Must run before the `src.*` imports below.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from src.train.data import build_loader
from src.train.diffusion import ContinuousDiffusion
from src.train.model import DiffusionLM

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_and_diffusion(cfg: dict, device: torch.device):
    model = DiffusionLM(cfg["model"]["base"]).to(device)
    diffusion = ContinuousDiffusion(
        hidden_size=model.hidden_size,
        schedule=cfg["diffusion"]["schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        train_steps=cfg["diffusion"]["train_steps"],
        infer_steps=cfg["diffusion"]["infer_steps"],
        prediction=cfg["diffusion"]["prediction"],
        mask_ratio=cfg["diffusion"]["mask_ratio"],
    )
    return model, diffusion


def save_ckpt(model, opt, sched, step, tokens_seen, cfg, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"step_{step}"
    model_file = out_dir / f"{prefix}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict() if sched is not None else None,
            "step": step,
            "tokens_seen": tokens_seen,
            "base_model": cfg["model"]["base"],
        },
        model_file,
    )
    meta = {
        "step": step,
        "tokens_seen": tokens_seen,
        "base_model": cfg["model"]["base"],
        "created_at": time.time(),
    }
    (out_dir / f"{prefix}.meta.json").write_text(json.dumps(meta, indent=2))
    return model_file


def upload_ckpt(r2, out_dir: Path, step: int, prefix: str):
    for suffix in ["pt", "meta.json"]:
        local = out_dir / f"step_{step}.{suffix}"
        key = f"{prefix}/step_{step}.{suffix}"
        r2.upload_file(local, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train_config.yaml")
    ap.add_argument("--out", default="checkpoints", help="save dir (on Modal: the mounted volume)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap for smoke test")
    ap.add_argument("--ckpt-every-steps", type=int, default=None,
                    help="override the time-based interval with a step interval (smoke test)")
    ap.add_argument("--upload-r2", action="store_true", help="additionally push ckpts to R2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_train(cfg, args.out, max_steps=args.max_steps,
              ckpt_every_steps=args.ckpt_every_steps, upload_r2=args.upload_r2)


def run_train(cfg, out, max_steps=None, ckpt_every_steps=None, upload_r2=False):
    r2_prefix = cfg.get("storage", {}).get("r2_prefix", "thunder-fast/checkpoints")

    # R2 is OPTIONAL now (the default is to save straight into the Modal volume at --out).
    r2_available = False
    if upload_r2:
        try:
            from infra.r2 import exists, download_file, upload_file as r2_upload
            r2_available = True
        except Exception:  # noqa: BLE001
            r2_available = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion = build_model_and_diffusion(cfg, device)
    model.set_train_ctx()

    # Two parameter groups: base weights (inherited from the AR init) are kept at a
    # conservative LR; newly added modules (time_mlp, mask_embedding) get a higher LR so
    # they do not lag behind the frozen-knowledge backbone.
    base_params, new_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("time_mlp") or name.endswith("mask_embedding"):
            new_params.append(p)
        else:
            base_params.append(p)
    lr_base = cfg["training"]["learning_rate"]
    lr_new = lr_base * cfg["training"].get("new_params_lr_mult", 5.0)
    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr_base},
            {"params": new_params, "lr": lr_new},
        ],
        lr=lr_base,
        weight_decay=cfg["training"]["weight_decay"],
    )
    # Estimate total steps so we can schedule a cosine LR decay over the token budget.
    tokens_per_step = cfg["training"]["batch_size_seq"] * cfg["diffusion"]["seq_len"] * cfg["training"]["grad_accum"]
    est_steps = max(1, cfg["training"]["total_tokens"] // tokens_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, (step + 1) / cfg["training"]["warmup_steps"])
        if step < cfg["training"]["warmup_steps"]
        else 0.5 * (1.0 + __import__("math").cos(3.14159265 * (step - cfg["training"]["warmup_steps"]) / max(1, est_steps - cfg["training"]["warmup_steps"]))),
    )

    train_loader = build_loader(cfg, model.tokenizer)

    start_step = 0
    tokens_seen = 0
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from the latest checkpoint in the save dir (the Modal volume). R2 is optional.
    latest = _find_latest_local(out_dir)
    if latest is None and r2_available:
        latest = _find_latest(r2_prefix)
        if latest is not None:
            local = out_dir / f"step_{latest}.pt"
            download_file(f"{r2_prefix}/step_{latest}.pt", local)
    if latest is not None:
        ckpt = torch.load(out_dir / f"step_{latest}.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        print(f"Resumed from step {start_step} (tokens seen {tokens_seen})", flush=True)

    model.train()
    cfg_t = cfg["training"]
    # Time-based checkpointing (saves every ckpt_every_hours, keeps last `keep_last`).
    import time as _time

    ckpt_interval_s = cfg_t.get("ckpt_every_hours", 2.0) * 3600.0
    keep_last = cfg_t.get("keep_last", 3)
    last_ckpt_time = _time.time()
    last_ckpt_step = -1  # guard so step-based checkpointing fires once per step, not per micro-batch

    # Weights & Biases observability (optional: skipped if no key or if it errors, so training
    # never dies on logging). The key is injected by Modal via the `wandb` secret.
    import os as _os

    _wandb = None
    if _os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            _wandb = wandb
            wandb.init(
                entity="adrian-tucicovenco-staticlabs",
                project="thunder-fast",
                name=f"run-{time.strftime('%Y%m%d-%H%M%S')}",
                config=cfg,
            )
        except Exception as _e:  # noqa: BLE001
            print(f"[wandb] disabled: {_e}", flush=True)
            _wandb = None

    LOG_EVERY = 25  # optimizer-steps between wandb.log calls
    acc_loss = 0.0
    acc_micro = 0
    log_tokens = tokens_seen
    log_time = _time.time()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] device={device} | params={n_params/1e6:.1f}M | "
        f"batch={cfg_t['batch_size_seq']}x{cfg['diffusion']['seq_len']} | "
        f"grad_accum={cfg_t['grad_accum']} | start_step={start_step} | tokens_seen={tokens_seen:,}",
        flush=True,
    )

    stream = iter(train_loader)
    global_step = start_step
    start_wall = _time.time()
    max_time_s = cfg_t.get("max_time_hours", 0.0) * 3600.0

    # Save on a graceful termination signal (e.g. cloud stop) so we resume from the current
    # step, not from the last 2h timer-based checkpoint.
    def _on_signal(signum, _frame):
        print(f"[train] received signal {signum}, saving checkpoint at step {global_step}...", flush=True)
        save_ckpt(model, opt, sched, global_step, tokens_seen, cfg, out_dir)
        _rotate(out_dir, keep_last)
        raise SystemExit(0)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)
    _first_batch = True
    _first_forward = True
    _dbg_micro = 0  # diagnostic: time data-fetch vs forward for the first few micro-batches
    while True:
        if max_steps is not None and global_step >= max_steps + start_step:
            break
        if max_time_s and (_time.time() - start_wall) >= max_time_s:
            print(f"[train] reached max_time_hours ({cfg_t.get('max_time_hours')}), stopping gracefully", flush=True)
            break
        _t_next = _time.time()
        try:
            batch = next(stream)
        except StopIteration:
            break
        # The "data-fetch" cost = pulling one micro-batch (32 sequences) out of the loader.
        if _dbg_micro < cfg_t["grad_accum"]:
            print(f"[dbg] micro {_dbg_micro}: data ok in {_time.time() - _t_next:.1f}s", flush=True)

        batch = batch.to(device)  # [B, L] input ids
        B, L = batch.shape
        if _first_batch:
            print(f"[train] first batch: B={B} L={L}", flush=True)
            _first_batch = False
        if _first_forward:
            print("[train] first forward start", flush=True)
        _t_fwd = _time.time()
        x0 = model.embed_tokens(batch)  # [B, L, D] clean embeddings
        t = torch.rand(B, device=device)  # continuous timesteps in [0, 1]
        loss = diffusion.training_loss(model, x0, model.mask_embedding.data, t)
        loss = loss / cfg_t["grad_accum"]
        loss.backward()
        if _first_forward:
            print("[train] first forward done", flush=True)
            _first_forward = False
        if _dbg_micro < cfg_t["grad_accum"]:
            print(f"[dbg] micro {_dbg_micro}: fwd+bwd done in {_time.time() - _t_fwd:.1f}s", flush=True)
            _dbg_micro += 1

        tokens_seen += B * L
        acc_loss += loss.item()
        acc_micro += 1

        if acc_micro % cfg_t["grad_accum"] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_t["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad()
            global_step += 1

            avg_loss = acc_loss / max(1, acc_micro)
            acc_loss = 0.0
            acc_micro = 0

            if global_step % 50 == 0 or global_step <= 3:
                print(
                    f"step {global_step} | tokens {tokens_seen:,} | loss {avg_loss:.4f} | grad_norm {grad_norm:.3f}",
                    flush=True,
                )
            if _wandb is not None and global_step % LOG_EVERY == 0:
                now = _time.time()
                dt = now - log_time
                tps = (tokens_seen - log_tokens) / max(dt, 1e-6)
                _wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/global_step": global_step,
                        "train/tokens_seen": tokens_seen,
                        "train/tokens_per_sec": tps,
                        "train/grad_norm": grad_norm,
                        "train/lr_base": opt.param_groups[0]["lr"],
                        "train/lr_new": opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else opt.param_groups[0]["lr"],
                        "train/token_budget_fraction": min(1.0, tokens_seen / max(1, cfg_t["total_tokens"])),
                    }
                )
                log_tokens = tokens_seen
                log_time = now

        # Time-based checkpointing (~every 2h), or step-based for a smoke test.
        now = _time.time()
        step_due = ckpt_every_steps is not None and global_step and global_step % ckpt_every_steps == 0 and global_step != last_ckpt_step
        time_due = global_step and (now - last_ckpt_time) >= ckpt_interval_s and global_step != last_ckpt_step
        if step_due or time_due:
            save_ckpt(model, opt, sched, global_step, tokens_seen, cfg, out_dir)
            _rotate(out_dir, keep_last)
            if r2_available:
                r2_upload(out_dir / f"step_{global_step}.pt", f"{r2_prefix}/step_{global_step}.pt")
                r2_upload(out_dir / f"step_{global_step}.meta.json", f"{r2_prefix}/step_{global_step}.meta.json")
            print(f"checkpoint step {global_step} (tokens {tokens_seen:,}) -> {out_dir}", flush=True)
            if _wandb is not None:
                _wandb.log(
                    {
                        "checkpoint/step": global_step,
                        "checkpoint/tokens_seen": tokens_seen,
                    }
                )
            last_ckpt_time = now
            last_ckpt_step = global_step

    # Always save a final checkpoint at the end of the run.
    if global_step:
        save_ckpt(model, opt, sched, global_step, tokens_seen, cfg, out_dir)
        _rotate(out_dir, keep_last)

    if _wandb is not None:
        _wandb.finish()


def _step_from_meta(meta: Path) -> int:
    """Extract the step number from a `step_<N>.meta.json` checkpoint filename."""
    return int(meta.name.split("step_")[-1].split(".")[0])


def _find_latest_local(out_dir: Path) -> int | None:
    steps = []
    for meta in out_dir.glob("step_*.meta.json"):
        steps.append(_step_from_meta(meta))
    return max(steps) if steps else None


def _rotate(out_dir: Path, keep_last: int) -> None:
    """Keep only the `keep_last` most recent checkpoints (by step) in `out_dir`."""
    metas = sorted(out_dir.glob("step_*.meta.json"), key=_step_from_meta)
    while len(metas) > keep_last:
        old = metas.pop(0)
        step = _step_from_meta(old)
        old.unlink(missing_ok=True)
        (out_dir / f"step_{step}.pt").unlink(missing_ok=True)


def _find_latest(r2_prefix: str) -> int | None:
    from infra.r2 import list_keys
    steps = []
    for key in list_keys(r2_prefix):
        if key.endswith(".meta.json"):
            step = int(key.split("step_")[-1].split(".")[0])
            steps.append(step)
    return max(steps) if steps else None


if __name__ == "__main__":
    main()
