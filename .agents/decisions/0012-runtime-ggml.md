# ADR 0012 — Runtime: motor propriu peste ggml pentru modele de difuzie pe text

**Status:** Accepted
**Data:** 2026-09-03
**Rezolvă:** decizia de runtime pentru inferența locală a modelului MDM

## Context
llama.cpp și LM Studio sunt construite pentru generare **autoregresivă** (atenție cauzală +
KV-cache + decode token-cu-token). Un model de difuzie mascată (MDM) cere **atenție
bidirecțională** și o **buclă de denoising** peste întreaga secvență. Confirmat prin căutare
(2026): nu există suport nativ, și **nu există un API de plugin** — arhitecturile sunt
hardcodate (enum + switch + builder de graf), deci a adăuga o clasă nouă de model înseamnă
modificarea codului C++. Un "plugin/preset" LM Studio nu poate produce atenție bidir sau un
loop de denoising.

## Decizie
- Construim un **runtime propriu** (folder separat `runtime/`) peste **ggml** ca motor de
  calcul (kernel-uri CPU SIMD / CUDA / Metal / Vulkan gratuite). Contribuția centrală e
  **atenția bidirecțională (mască all-ones)** + **bucla de denoising**, nu kernel-urile.
- Arhitectura rămâne **agnostică de model**: modelul e încă Qwen3; doar masca de atenție și
  bucla de generare diferă de llama.cpp. Deci se recală la un 7B printr-o schimbare de
  hiperparametri, fără rescriere.
- Build-ul și verificarea se fac prin **workflow GH Actions dispatch**
  (`.github/workflows/build-runtime.yml`), activat manual cu `gh workflow run`, care build-ul
  îl pune în **R2**; compilarea se validează pe **Modal**. Local nu se instalează/clonează
  ggml (regula AGENTS.md de "no local deps"); nucleul de difuzie e testabil oricum, fiind
  agnostic de ggml.

## Alternative respinse (reconfirmat în 2026)
- **llama.cpp ca atare / LM Studio** — nu suportă difuzie; nu există plugin/preset pentru
  asta. S-ar ajunge oricum la un fork.
- **Plugin peste llama.cpp** — ar necesita mai întâi construirea sistemului de înregistrare
  de plugin-uri (că nu există), deci mai multă muncă decât runtime-ul propriu.
- **GGUF/extra-pytorch ca "formate"** — GGUF conține cuantizare + runtime non-Pytorch;
  un safetensors cuantizat nu aduce automat viteza pe CPU (are nevoie de runtime-ul schemei).
- **Doar torch/MLX (GPU/Mac)** — rămâne calea rapidă pe GPU/Mac; runtime-ul peste ggml e
  pentru ținte CPU/AMD (Strix Halo, VPS fără GPU) unde torch e lent.

## Consecințe
- `runtime/` conține: nucleu de difuzie (testat), engine ggml, tokenizer, CLI + server HTTP
  OpenAI-compatibil (interogabil ca un serviciu, la fel ca LM Studio), converter GGUF, upload
  R2, build files.
- Partea **nefiltrată încă**: materializarea grafului ggml (mapping de tensori + atenție
  bidir) și exactitatea tokenizer-ului byte-level; se validează pe build-ul Modal.
- Portabilitate: același checkout de checkpoint R2 → GGUF servește GPU (torch/MLX) și
  runtime-ul ggml (CPU/AMD).
