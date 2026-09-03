# Runtime (ggml)

A standalone C++ inference engine for the discrete masked-diffusion model, living in
`runtime/` (ADR 0012). Unlike llama.cpp it uses **bidirectional attention** (an all-ones KQ
mask) and **no KV-cache**; the whole window is one forward pass. It is built on **ggml** for
the tensor kernels (CPU SIMD / CUDA / Metal / Vulkan), so the custom contribution is the
denoise loop + bidirectional attention, not the kernels.

## Layout

```
runtime/
  src/diffusion/   denoise + schedule + sampler      (custom core, tested)
  src/engine/      Qwen3 backbone over ggml (bidirectional attention)
  src/tokenizer/   byte-level BPE
  tests/           functional test of the diffusion core (no ggml)
  tools/           convert_to_gguf.py, upload_r2.py
  CMakeLists.txt   primary build (CI / Modal)
  Makefile         local convenience
```

## Why a custom runtime (not llama.cpp / LM Studio)

llama.cpp and LM Studio are built for autoregressive generation (causal + KV-cache). A masked
diffusion model needs bidirectional attention and a denoise loop. There is **no plugin API** to
add this: architectures are hardcoded (enum + switch + graph builder). A fork of llama.cpp is
possible but, because our backbone is Qwen3 (already fully supported), the fork would be small
— yet a plugin/preset is not available. Building on ggml directly keeps us independent of
llama.cpp's generation machinery while reusing its kernels.

## Build & deployment

Compilation is gated on the **Modal/CI build** because local dev has no ggml; the diffusion
core is the exception (it compiles and tests without ggml):

```
cd runtime && make test          # builds + runs the diffusion-core test (no ggml)
gh workflow run build-runtime    # dispatch build -> uploads the binary to R2
```

The workflow (`build-runtime.yml`) is **dispatch-only** (never runs on push/PR). It fetches
ggml, compiles, runs the diffusion-core smoke test, and pushes artifacts to R2
(`runtime/tools/upload_r2.py`; credentials come from GitHub secrets).

## Usage

CLI:
```
runtime --model model.gguf --prompt "Bună ziua" --steps 24 --target 256
```

Server (OpenAI-compatible, LM-Studio style):
```
runtime --model model.gguf --serve 8000
curl -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"Bună ziua","max_tokens":256}'
```
Endpoints: `GET /health`, `POST /v1/completions`, `POST /v1/chat/completions`.

## Tokenizer

Each backbone model ships **its own tokenizer** (Qwen3 the same). The converter embeds the
backbone's tokenizer into the GGUF and the runtime reads it from there — the tokenizer is not
shared across models. (v1 runtime tokenizer is a simplified byte-BPE; see `runtime/README.md`.)

## Status & gaps

- `diffusion/*` — implemented and unit-tested (verified locally with clang).
- `engine/`, `tokenizer/`, `main.cpp`, `tools/` — written as a reference; **not yet compiled**
  here. `RuntimeModel::forward` still needs the ggml graph materialized and the tensor-name
  mapping validated against the converter. These are validated on the Modal build.
