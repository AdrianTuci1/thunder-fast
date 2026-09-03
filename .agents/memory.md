# memory.md — viziune & stare thunder-fast

> Fișierul principal de memorie. Se citește primul la fiecare sesiune.

---

## 1. Viziunea proiectului

Construim un **model de limbaj bazat pe difuzie mascată discretă** (non-autoregresiv), adaptat
folosind tehnica **DiffuLLaMA** la varianta **MDM (Masked Diffusion Model)**. Generăm
**256 de tokeni în paralel** prin
**24 pași de difuzie** (ADR 0007), cu **atenție bidirecțională** și un token discret `[MASK]`.
Ieșirile lungi (până la **2048 tokeni**) se fac **pe blocuri** de 256.

Diferența cheie față de un LLM clasic: nu generăm token cu token, ci pornim dintr-o stare plină de
zgomot/mască și rafinăm **întreaga secvență simultan** la fiecare pas. Cum am subliniat: contribuția
noastră e **modul în care se face difuzia** și, foarte important, **codul de calcul CPU** (instrucțiuni
SIMD) care face inferența rapidă.

### De ce
- Generare paralelă: cost crește cu numărul de **pași** (24), nu cu lungimea (256), deci ieftin la
  ieșiri lungi.
- Atenție bidirecțională => context complet la fiecare pas, infilling natural.
- Optimizări SIMD pe CPU => rulare fără GPU (Apple-Silicon, laptopuri, VPS AMD/Intel), cost mic.

---

## 2. Constrângeri hard (de la utilizator)

- Trebuie să **păstrăm difuzia continuă** cerută (nu autoregresie) + **atenție bidirecțională**.
- Generare: **256 tokeni în paralel**, **24 pași de difuzie**.
- Folosim **tehnica DiffuLLaMA** pentru adaptarea modelului (nu construim de la zero).
- **Doar CPU** ca target de rulare: Apple Silicon (M-chip), Linux, Windows, VPS AMD/Intel.
  - x86: **AVX2**, **AVX-512**, **AMX**, **AVX-512 VNNI**
  - ARM: **NEON**, **SVE**
- **Cuantizare INT / Q4** (căutăm viteze mari).
- Model **≤1B multilingual care suportă și română** — ales: `Qwen3-0.6B` (ADR 0002), ca să încapă
  pe 2–4 GPU în 24h și să vadă mai mulți tokeni.
- **Nu instalăm dependențe momentan** (doar planificare/documente).
- Totul fundamentat pe **informații reale** (surse, trade-uri tehnice reale).
- Se creează **`.agents/`** ca memorie proprie de lucru.

---

## 3. Decizia: modelul de bază (ADR 0002 — ACCEPTED)

Vrem ≤1B multilingual cu română, licență permisivă, adaptabil la difuzie, care să **încadreze pe
2–4 GPU în ~24h și să vadă câți mai mulți tokeni**.

**Ales: `Qwen/Qwen3-0.6B`** (~0.6B, Apache-2.0, LLaMA-style, GQA, multilingual puternic).

| Candidat | Mărime | Licență | Română | Încape pe 2 GPU / 24h | Notă |
|---|---|---|---|---|---|
| **Qwen3-0.6B** | 0.6B | Apache-2.0 | Bună | ✓ (20B în ~20h la 50% MFU) | **ALES** — cel mai bun raport ~1B |
| Qwen2.5-1.5B | 1.5B | Apache-2.0 | Bună | ✗ (~63h pt 20B) | Calitate mai bună, dar NU încape |
| Qwen3-1.7B | 1.7B | Apache-2.0 | Bună | ✗ | Peste bugetul de GPU |
| Llama-3.2-1B | 1B | Llama license | Medie | ✓ | Licență restrictivă |
| SmolLM2-1.7B | 1.7B | Apache-2.0 | Slabă | — | Româna slab acoperită |

> Dacă vrem mai multă calitate pe română la scara asta, continuăm de la Qwen3-0.6B cu un mic
> fine-tune de limbă (română) — cost mic față de a porni de la un model mai mare.

---

## 4. Abordarea de difuzie (DiffuLLaMA-style)

Adaptăm un checkpoint autoregresiv gata antrenat la un objectiv de difuzie:

1. **Reutilizăm greutățile** modelului pre-antrenat (Qwen/Llama).
2. **Înlocuim masca de atenție cauzală** cu **mască bidirecțională** (full attention).
3. **Schimbăm obiectivul** de învățare: de la next-token cross-entropy la **denoising**
   (predicția tokenilor curați dintr-o secvență parțial zgomotată/mascată).
4. Adăugăm **time-step embeddings** (codificarea pasului de difuzie).
5. Antrenăm să inverseze procesul de zgomot.

**Varianta "difuzie continuă"** (cerută explicit): zgomot **Gaussian** pe **embeddings continue**,
apoi un **head de roundare**/proiecție la tokeni discreți. Alternativa (discretă, mască absorbantă)
există, dar utilizatorul a cerut **continuă** => mergem pe continuă.

### Bucla de inferență (256 tokeni, 24 pași)
```
x = zgomot Gaussian / secvență inițializată (256 tokeni)
for t in range(24):                    # pași de difuzie
    logits = model(x, t)               # forward complet, atenție bidirecțională
    x = denoise_step(x, logits, t)     # argmax/sampling + re-noising după program
repetă până la curățare
```

---

## 5. Arhitectura modelului (schiță)

- **Backbone:** transformer LLaMA-style (Qwen), cu attention bidirecțională.
- **Input:** embeddings continue (zgomot la pas inițial).
- **Adiții pentru difuzie:** time-step embedding, head de proiecție la logits/tokens.
- **Numerotare pași:** **18 pași** de difuzie (ADR 0007) — schedule liniar/cosinus de la zgomot
  -> curat; train & infer consistent (18/18).
- **Dimensiuni (ex. 1.5B):** ~28 straturi, hidden ~1536, head-uri GQA, vocab ~151k (Qwen).
- **Lungimea secvenței:** **N = 256 tokeni** (confirmat). Atenția bidirecțională se calculează pe
  matricea întreagă [256 × hidden]; la n=256 atenția completă O(n²) e ieftină (256×256 = 65.536),
  deci **nu e nevoie de KV-cache** și nu avem problemă de memorie/latency la atenție.
- **Runtime (ADR 0012):** motor propriu în `runtime/` peste **ggml** (atenție bidirecțională + buclă
  de denoising). Înlocuiește direcția simplă din ADR 0003; kernel-urile SIMD custom rămân atribut
  opțional de tuning, nu contribuția principală. Nucleul de difuzie e **agnostic de backend**
  (`forward(ids) -> logits`), deci funcționează și pe torch/MLX.

---

## 6. Cuantizare (INT / Q4)

- **Recomandare:** `Q4_K_M` (GGUF) — raport calitate/mărime bun, kernel-uri bune pe AVX2/AVX-512/NEON.
- **Alternative:** `Q4_0` (mai clasic, kernel foarte optimizat), `IQ4_XS` (mai mic), activări `Q8_0`.
- Aktivările (embeddings continue de difuzie) rămân **FP16/FP32**; doar greutățile se cuantizează la 4-biți.
- **VNNI avantaj:** `VPDPBUSD` pentru produs scalar int8 — se potrivește perfect cu Q4/Q8 dot products.
- **Verificare:** kernel-urile trebuie să suporte pattern-ul difuziei (matmuls mari, atenție full).

---

## 7. Kernel-uri SIMD CPU (contribuția noastră principală)

Dominanta costului la CPU = **matmul** de de-cuantizare + GEMM la fiecare pas de difuzie (24 de forward-uri).

**De ce adaptăm, nu doar reutilizăm:** kernel-urile ggml din llama.cpp sunt optimizate pentru
generare autoregresivă:
- **Decode AR** = GEMV (1 token, batch mic) + KV-cache + mască cauzală — profil memory-bound.
- **Difuzia noastră** = **GEMM batched** (256 tokeni simultan: [256 × hidden] × weights), **fără
  KV-cache**, **atenție completă non-cauzală** — profil de compute/bandwidth diferit.

Abordare în cascade (ADR 0003):
1. **Baseline** cu kernel-urile ggml stock (corectitudine + referință).
2. **Adaptare:** kernel de **atenție non-cauzală** + tunning GEMM la shape 256 tokeni + fuziuni/layout
   pentru fluxul de denoising.
3. **Maximizare:** cai micro-arhitecturale și kernel-uri proprii unde calea ggml e suboptimă.

**Instrucțiuni target per micro-arhitectură:**
- **x86:** scalar -> SSE -> **AVX2** (256-bit) -> **AVX-512** (512-bit) -> **AMX** (Intel Sapphire Rapids+;
  tile ops, potrivit int8 MAC) -> **AVX-512 VNNI** (`VPDPBUSD`/`VPDPBUSDS`, int8 dot).
- **ARM:** **NEON** (128-bit) -> **DotProd/I8MM** (ARMv8.2+/8.6+) -> **SVE** (unde există, ex. Grace).
  Apple Silicon folosește NEON (nu are SVE pe chip-uri consumer), plus **AMX Apple** (matrice coprocesor).
- **Dispecerizare runtime** prin detecție CPUID/features (ca în ggml `ggml_cpu_has_avx512_vnni()`),
  sau flag-uri la compilare.
- **Target:** kernel-uri de **de-cuantizare + dot** pentru Q4/Q8, adaptate la pattern-ul nostru de GEMM
  batched și atenție non-cauzală.

---

## 8. Structura propusă a repo-ului

```
thunder-fast/
  .agents/            <- memoria (de aici)
  AGENTS.md
  README.md
  docs/
    design.md         <- design-ul difuziei
    roadmap.md
  model/
    config.json       <- config backbone + adapter difuzie
    tokenizer/
  src/
    train/            <- training difuzie (continuă)
    infer/            <- inferență + runtime
    kernels/          <- SIMD (x86/arm)
    dllama/           <- logica DiffuLLaMA (adaptare)
  convert/            <- convert_to_diffusion.py (AR -> difuzie inițial)
  scripts/            <- utilitare (eval română, bench, export)
  tests/
  eval/               <- eval.py (română + sanity gen.)
  infra/              <- Modal (modal_train.py), RunPod (Dockerfile.runpod, runpod_launch.py), R2 (r2.py)
  config/             <- train_config.yaml
  requirements.txt    <- dep. de runtime (doar pentru imaginile cloud)
```

> **Stare scaffolding:** au fost create `infra/*`, `config/*`, `src/train/*`, `convert/*`, `eval/*`,
> `src/infer/*`, `requirements.txt` ca schelet de lucru. Modulul trece `py_compile`.

> **BLOCANT (smoke test):** tokenul HF disponibil local este **un DEMO, nu unul real**. Secretul Modal
> `hf` a fost creat cu acest token, deci **NU poate autentifica** descărcările gated (Qwen3-0.6B, OSCAR).
> Pentru a rula smoke test-ul avem nevoie de un token HF real:
> `modal secret delete hf && modal secret create hf HF_TOKEN=<TOKEN_REAL>`.
> (C4 și OPUS sunt publice, nu necesita auth; Qwen3-0.6B și OSCAR da.)

---

## 8b. Volumul de cod (ce scriem vs ce reutilizăm)

Lasă-mă să estimez cât cod avem **noi** de scris, pentru că reutilizarea e masivă.

**REUTILIZĂM (nu le scriem):**
- Kernel-urile SIMD (AVX2/AVX-512/VNNI/NEON + cuantizare Q4_K/Q8_0) — **ggml**.
- Backbone transformer (attention, FFN, RMSNorm) și formatul GGUF — **ggml**.
- Tokenizatorul (Qwen) și transformerele/matmul-urile din training — **HF transformers / PyTorch**.

**SCRIEM NOI** (ordonați după efort):

| Modul | Ce conține | Efort |
|---|---|---|
| `convert/` | Script Python: checkpoint AR → format difuzie (maskă bidirecțională, time-embed, head) + export GGUF | Mediu |
| `src/infer/` | Motorul de execuție difuzie: graful ggml cu atenție bidirecțională + bucla 24 pași + sampler | **Mare** |
| `src/train/` | Pipeline difuzie: forward, program de zgomot, loss de denoising, date | Mare |
| `src/dllama/` | Logica de adaptare (împrumutată/înțeleasă din DiffuLLaMA) | Mic-Medi |
| `src/infer/cli` | CLI: prompt → tokenizare → 256 tokeni → încarcă GGUF | Mic |
| `src/infer/kernels` | **Adaptăm/întindem kernel-urile ggml** la pattern-ul nostru: atenție non-cauzală, GEMM batched 256 tokeni, cai VNNI/AMX/NEON-I8MM/SVE | Mediu–Mare |

**Estimare realistă (binar C++):** ~3-6k LOC de cod nou (loader + graful de atenție bidir + buclă de
denoising + sampler + CLI + **kernel-uri adaptate/întinse**); **~1-2k LOC** Python pentru training +
conversie. Părțile grele (SIMD de bază, cuantizare, backbone) provin din ggml ca baseline, dar
**adaptarea lor la pattern-ul nostru de calcul** e munca noastră. Partea cu adevărat delicată e
**bucla de difuzie + sampler + kernel-urile de atenție non-cauzală**, nu rescrierea algoritmului de matmul.

---

## 8c. Stare inferență (2026-09-03): baseline + diagnostic

Am rulat inferența de referință (PyTorch, re-împachetat ca `infra/modal_infer.py`) pe
`step_14939` (volumul `thunder-checkpoints`), cu config-ul de training. Rezultate:

- **Viteză (256 tokeni, 24 pași):** CPU Modal **3.48 tok/s** (~74s); A10G (GPU mic) **372 tok/s** (~0.7s).
- **Calitate: GARBAGE** (chineză/ebraică/poloneză amestecate).

**Diagnostic (`infra/diag_loss.py` pe `step_14939`)** a **respins colapsul** (pred std=0.76) și a
arătat că modelul e excelent la **reconstrucția pozițiilor mascate** (`mse[x0] MASKED=1e-4`), dar
calea de referință `sample()` (Gaussian+`epsilon`, fără mascare) nu folosește forța asta. Cauze:
program de zgomot slab (`ā(t=1)=0.784`), embeddings minuscule (`x0`-norm ~0.2), obiectiv amestecat
(Gaussian+mascare+`epsilon`). Generare mascată pe checkpointul vechi => garbage repetitiv (antrenat
doar cu `mask_ratio=0.25`). **=> Corecție:** obiectiv de mascare continuă + `x0` + curriculum de
mascare (ADR 0009), implementat și smoke-testat.

**Smoke retrain (80 pași din 0, `/vol/checkpoints/v2-masked-x0/step_75.pt`):** loss **1.91→0.025**,
fără colaps. Generare la 80 pași încă repetitivă (doar 5M tokeni) — **rularea lungă e pending**
(decizie utilizator; nu se cheltuiește compute acum).

**Acesta NU e runtime-ul optimizat SIMD/ggml (ADR 0003)** — viteza CPU de mai sus e baseline
PyTorch, nu ținta finală.

## 8d. Migrare la mascare discretă (ADR 0010, 2026-09-03)

Am evaluat obiectivul de difuzie continuu (ADR 0008/0009) pe mărimea noastră (Qwen3-0.6B): la volum mic
de antrenare nu scoate text coerent și e sensibil la scara embeddings (vectori off-manifold). Am migrat
la **mascare discretă (MDM)**: token `[MASK]` în vocabular, rapoarte de mascare uniforme din `[0,1]`,
loss = **cross-entropy mascată** (`mdm_loss + path_loss`), atenție bidirecțională, fără condiționare de
timp.

Decizia utilizatorului (prin întrebare): **migrăm la mascare discretă**, **păstrăm 24 pași / 256 tokeni**
și **suportăm ieșiri până la 2048** prin **generare pe blocuri (block-wise)**: fiecare pas de difuzie
generează 256 tokeni, apoi îi adaugă ca context și repetă (8× pentru 2048).

**Implementat (2026-09-03):**
- `src/train/diffusion.py` — `MaskedDiffusion` (CE mascată discretă, `mdm_loss + path_loss`).
- `src/train/model.py` — `[MASK]` + `resize_token_embeddings`; `forward(input_ids)->logits`;
  `generate()` unmask progresiv discret; `generate_long()` pe blocuri. Scad `time_mlp`/`mask_embedding`.
- `src/train/train.py`, `config/train_config.yaml`, `eval/eval.py`, `src/infer/inference.py`,
  `infra/diag_loss.py`, `infra/modal_infer.py`, `infra/bench_gpu.py`, `convert/convert_to_diffusion.py` — actualizate.
- Toate modulele trec `py_compile`.

**Config-ul actual:** `seq_len: 256`, `infer_steps: 24`, `mask_ratio_min: 0.002`, `mask_ratio_max: 0.998`.
**De reținut:** reantrenare de la 0 (checkpoint-urile continue sunt incompatibile). Modelul învață pe
ferestre de 256; contextul agregat la generarea pe blocuri depășește fereastra de antrenare — dacă
coerența pe distanțe lungi suferă, luăm în calcul ferestre mai lungi la antrenare.

## 8e. ROOT-CAUSE inferență: atenția bidirecțională trebuie impusă prin mască 4D (ADR 0011)

Observam că modelul denoisează corect de la pași mici, dar motorul nostru scotea garbage la orice număr
de pași. Am izolat cauza: **modelul MDM trebuie rulat cu atenție BIDIRECȚIONALĂ, iar motorul nostru rula
cauzal.**

Mecanismul (validat pe GPU):
- `_update_causal_mask` (HF) construiește mască 4D **cauzală** când `attention_mask=None` (sau 2D).
  Acea mască e pasată direct kernelului și **suprascrie** `is_causal=False`. Deci `is_causal=False` + `mask=None`
  = cauzal (no-op).
- Singura cale reală de bidirecțional: **mască 4D all-zero** (folosită direct de `_update_causal_mask`).

Experimente decisive (A10G, același model + bucla noastră):
- **cauzal** → garbage; **bidirecțional (mască 4D all-zero)** → **quicksort coerent** (`def quick_sort`, `partition`, `assert quick_sort([5,6,8,10,1,2,1]) == ...`).
- Calea de referință `diffusion_generate` (pasează mask=None) → garbage (adică și ea rula cauzal în
  acest container; doar căile care impun mască 4D dau output bun).

Fix aplicat:
- `infra/modal_infer_open.py`: forward cu mască 4D all-zero; viteză **66.2 tok/s** la 100 pași /
  264 tokeni, output coerent.
- `src/train/model.py`: `DiffusionLM.forward` trece mască 4D (bool all-`False`) în loc de `attention_mask=None`;
  `make_bidirectional` doc. actualizat (NU e suficient de una singură).

**IMPORTANT:** același bug latent afecta modelul nostru (Qwen3-0.6B) — `attention_mask=None` la backward =
atenție cauzală, deci obiectivul MDM s-ar antrena cauzal. Fixul de mai sus trebuie validat pe Qwen3 exact
printr-un smoke test cloud (AGENTS.md).
**Starea inferenței:** motorul nostru funcționează acum (output coerent). Validarea pe Qwen3-0.6B +
reantrenare MDM de la 0: **pending** (decizie utilizator privind compute).

## 9. Întrebări deschise (una cu una cu utilizatorul)

1. **Modelul final** — Qwen3-1.7B vs Qwen2.5-1.5B? (sau altul după testele tale anterioare)
2. **Difuzie continuă** — confirmăm Gaussian pe embeddings + head de roundare (varianta DiffuLLaMA
   continuă + mascare), sau preferi pure Gaussian fără mascare?
3. **Training** — continuăm pre-antrenarea unui checkpoint existent (RECOMANDAT — mult mai ieftin
   decât de la zero; vezi research notes §5).
4. ~~**Runtime**~~ — **DECIS (ADR 0003):** binar custom peste kernel-urile **ggml** (llama.cpp ca atare
   nu suportă difuzie); scriem noi motorul de difuzie (atenție bidirecțională + buclă 24 pași + sampler).
5. ~~**Atenție cu cache**~~ — **DECIS:** atenție bidirecțională pe N=256 este ieftină (O(n²) la 256),
   nu e nevoie de KV-cache. Folosim atenție completă simplă pe [256 × hidden].
6. **Date românești** — curățăm/selectăm din OSCAR ro, Europarl, OPUS, WMT (EN–RO); eval cu
   LiRo + FLORES/WMT (vezi research notes §6). **Bugetul de tokeni** de stabilit.
7. **Hardware de antrenament** — ce folosim pentru adaptare (cloud GPU A100/H100, sau local)?
