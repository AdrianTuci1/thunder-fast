# thunder-fast runtime — masked-diffusion inference engine (over ggml)

A standalone C++ inference engine for the discrete masked-diffusion (MDM) text model
trained in this repo. It loads a converted GGUF and runs the progressive-unmasking
denoise loop. Unlike llama.cpp it uses **bidirectional attention** (an all-ones KQ mask)
and **no KV-cache** — the whole window is one forward pass.

## Layout

```
runtime/
  src/
    diffusion/    denoise loop + mask schedule + sampler      (custom core)
    engine/       Qwen2 backbone over ggml, bidirectional attention
    tokenizer/    byte-level BPE (vocab.json + merges.txt)
  tests/          functional test of the diffusion core (no ggml)
  tools/          convert_to_gguf.py, upload_r2.py
  CMakeLists.txt  primary build (CI / Modal)
  Makefile        local convenience
```

The diffusion core is the project's contribution and is **self-contained**: it only needs a
`forward(ids) -> logits` callable, so it works against ggml, torch, or MLX. It is verified
by `tests/test_diffusion.cpp`.

## Build

The engine links `ggml` (fetched as a CMake dependency). Local dev has no compiler/ggml
here, so the real build runs in the dispatch workflow and is verified on Modal:

```
gh workflow run build-runtime
```

Local (no ggml) — build + run the diffusion-core test:

```
cd runtime
make test
```

## Usage

CLI (one-shot):
```
runtime --model model.gguf --prompt "Bună ziua" --steps 24 --target 256
```

HTTP server (OpenAI-compatible), the LM-Studio-style interface:
```
runtime --model model.gguf --serve 8000
curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Bună ziua","max_tokens":256}'
```
Endpoints: `GET /health`, `POST /v1/completions`, `POST /v1/chat/completions`.

## Status

- `diffusion/*` — implemented and unit-tested (local clang).
- `engine/`, `tokenizer/`, `main.cpp`, `tools/` — written as a reference; **not yet compiled**
  here. The ggml graph materialization, tensor-name mapping, and the exact byte-level
  render table are validated on the Modal build (ADR 0012).

## Build notes / current gaps

- `RuntimeModel::forward` still needs the ggml graph materialized (see TODO in
  `engine/model.cpp`); building it requires the ggml checkout.
- `convert_to_gguf.py` tensor-key mapping and `n_layer` guess are validated against the
  actual `DiffusionLM` state dict at build time.
- `tokenizer.cpp` v1 does raw-byte BPE; Qwen2's GPT-2-style byte->unicode render and
  pre-tokenization regex are a follow-up.
