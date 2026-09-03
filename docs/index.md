# thunder-fast — documentation

This maps the repository and explains how the pieces fit. The model is a **non-autoregressive
discrete masked-diffusion (MDM)** model adapted from **Qwen3-0.6B**: it reconstructs a whole
window in parallel over 24 diffusion steps with bidirectional attention and a `[MASK]` token.

| Doc | Covers |
|---|---|
| [architecture](architecture.md) | How a causal AR model becomes a bidirectional diffusion denoiser; the model-agnostic core |
| [training](training.md) | Config, data pipeline, objective, training loop, checkpointing, R2 resume |
| [inference](inference.md) | Reference PyTorch engine, block-wise generation, measured speeds |
| [runtime](runtime.md) | The standalone ggml runtime: CLI + OpenAI-compatible server, build → R2 |
| [conversion](conversion.md) | AR→diffusion checkpoint conversion and GGUF export |
| [evaluation](evaluation.md) | The eval harness (reconstruction loss, sample, wired NLU/MT stubs) |
| [infrastructure](infrastructure.md) | Modal + RunPod + R2; credentials stay in env only |
| [config](config.md) | Reference for `config/train_config.yaml` |

Reading order: architecture → training → inference → runtime → conversion → evaluation →
infrastructure → config.

Decisions are recorded as ADRs in `.agents/decisions/` (see also `.agents/memory.md` and
`.agents/roadmap.md`).

## Repo layout

```
src/train/      model.py (DiffusionLM), diffusion.py (MaskedDiffusion), data.py, train.py
src/infer/      reference PyTorch inference path + kernels/ (unused SIMD, ADR 0012)
convert/        convert_to_diffusion.py (AR -> diffusion checkpoint)
eval/           eval.py (reconstruction loss + sample)
infra/          modal_train.py, modal_infer_open.py, modal_r2_transfer.py, runpod_launch.py,
                Dockerfile.runpod, r2.py (R2 client)
config/         train_config.yaml
runtime/        standalone ggml inference engine (ADR 0012)
docs/           this documentation
.agents/        memory, roadmap, ADRs
```
