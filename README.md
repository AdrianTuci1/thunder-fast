# thunder-fast

Model de limbaj bazat pe **difuzie mascată discretă** (non-autoregresiv): adaptăm un LLM
autoregresiv existent la un **Masked Diffusion Model (MDM)**. Modelul țintă este **Qwen3-0.6B**.
Generăm **256 de tokeni în paralel** prin **24 de pași de difuzie**, cu **atenție bidirecțională**
și un token discret `[MASK]`. Ieșirile lungi (până la **2048 tokeni**) se produc **pe blocuri** de 256.

> Stare: motorul de antrenare + inferență funcționează (mascare discretă, block-wise); runtime-ul
> CPU SIMD este încă în plan (ADR 0003). Inferența curentă este de referință PyTorch.
> Memoria proiectului: [`.agents/`](.agents/).

## Scopul (ce facem și de ce)

Construim un model care **generează întreaga secvență simultan**, nu token cu token:

1. Pornim de la un checkpoint **autoregresiv** antrenat (Qwen3-0.6B) și îl **adaptăm la difuzie**
   fără a-l antrena de la zero.
2. **Atenție bidirecțională** — la fiecare pas modelul vede întregul context, deci **infilling
   natural** și reconstrucție din context complet.
3. **Un token discret `[MASK]`** este "zgomotul": modelul reconstruiește tokenii mascați cu
   **cross-entropy** (descoperire progresivă la generare).
4. Diferențiatorul proiectului: **rulare rapidă pe CPU** (Apple Silicon, laptop, VPS x86/ARM)
   printr-un runtime custom peste kernel-uri SIMD cu cuantizare Q4 (ADR 0003).

Detaliile tehnice ale adaptării (atenție bidir, token de mascare, obiectiv, unmask progresiv,
generare pe blocuri) sunt documentate în [`docs/porting-to-diffusion.md`](docs/porting-to-diffusion.md).

## Cum îl folosim

Modelul publicat este **[`staticlabs/thunder-dlm-0.6b`](https://huggingface.co/staticlabs/thunder-dlm-0.6b)**
(Qwen3-0.6B adaptat la difuzie mascată discretă). Totul rulează pe imagini Modal/RunPod — nu instalezi
nimic local, credențialele sunt doar în env.

1. **Descarcă modelul** în volumul Modal:
   ```bash
   TF_MODEL=staticlabs/thunder-dlm-0.6b modal run infra/modal_download_open.py
   ```
2. **Generează** cu motorul nostru (block-wise, 24 de pași):
   ```bash
   TF_PROMPT="Scrie un quicksort in python." modal run infra/modal_infer_open.py
   ```
   - `TF_MODE=single` = o singură fereastră mare (implicit `block` = pe blocuri).
   - `TF_STEPS=24` pași de difuzie; `TF_GPU=A10G` etc. pentru GPU (implicit CPU).
3. **Antrenează / fine-tunează** — `modal run infra/modal_train.py -- --config config/train_config.yaml`
   (checkpoint-uri în `/vol/checkpoints`).
4. **Evaluează** — `python eval/eval.py --ckpt <path>`.

Detaliile variabilelor, modurilor de generare și vitezele sunt în secțiunea
[Cum rulezi (cloud)](#cum-rulezi-cloud).

## Eficiența

| Dimensiune | Autoregresiv (AR) | Difuzie (noi) |
|---|---|---|
| Generare | 1 token / pas secvențial (n pași) | **256 tokeni / pas** (pași independenți de lungimea ferestrei) |
| Cost pe fereastră | O(n) forward-uri, KV-cache, memory-bound | **O(steps)** forward-uri peste fereastra întreagă, **fără KV-cache** |
| Atenție | cauzală, cu cache | **bidirecțională** O(n²), la n=256 = 65k (ieftină) |
| Ieșiri lungi (2048) | 2048 pași | **8 blocuri × 24 pași** = 192 forward-uri (context crește) |

Puncte-cheie:

- **Generare paralelă**: într-o fereastră de 256, costul crește cu **numărul de pași** (24), nu cu
  lungimea. Din punctul de vedere al runtime-ului, fiecare pas este un **GEMM batched** [256 × D] —
  profil de **compute/bandwidth**, nu memory-bound ca GEMV-ul din AR. Asta se potrivește bine cu
  SIMD + cuantizare Q4 pe CPU.
- **Fără KV-cache**: atenția completă la 256 este ieftină (256×256), deci nu avem memoria/costul
  cache-ului de la AR.
- **Block-wise**: pentru 2048, concatenăm rezultatele a 8 ferestre de 256, fiecare cu contextul
  anterior ca prefix (contextul crește). Oprire la EOS dacă vrem.
- **Nivel de calitate vs pași**: la 24 de pași obținem ieșiri coerente; mai mulți pași cresc treptat
  calitatea, la cost proporțional.

Vitezele sunt de referință **PyTorch** (nu ținta SIMD): la 24 de pași ≈ 215 tok/s pe blocuri
(A10G). Runtime-ul SIMD este încă de construit (ADR 0003).

## Posibile utilizări

- **Generare de cod** — completare de funcții/algoritmi (ex. quicksort, palindrom).
- **Infilling de cod (FIM)** — completare în mijloc, posibilă nativ datorită atenției bidir.
- **Răspunsuri/text lungi** până la **2048 tokeni** prin generare pe blocuri.
- **Generare paralelă în batch** — mai multe secvențe complete simultan (avans pe GPU, agregat pe
  CPU-SIMD).
- **Inferență locală / offline pe CPU** — Apple Silicon, laptopuri, VPS x86/ARM, fără GPU, cu cost
  mic (ținta ADR 0003).
- **Multilingual** — datele proprii mixează engleză + română (C4 ro, OPUS en–ro).

## Structura repository-ului

```
.agents/              memoria proiectului (decizii, roadmap, research)
docs/                 porting-to-diffusion.md (cum adaptăm un LLM AR la difuzie)
config/               train_config.yaml (model, difuzie discretă, date)
src/train/            training MDM (model, diffusion, data, train)
src/infer/            inferență de referință PyTorch (generate/generate_long)
convert/              convert_to_diffusion.py (AR -> difuzie inițial)
eval/                 eval.py (reconstruction loss + generare)
infra/                Modal + RunPod + R2 (trainers, download, inferență)
```

## Cum rulezi (cloud)

Nu instala nimic local; se rulează pe imagini Modal/RunPod.

1. **Descarcă modelul** `staticlabs/thunder-dlm-0.6b` în volumul Modal:
   `TF_MODEL=staticlabs/thunder-dlm-0.6b modal run infra/modal_download_open.py`
2. **Inferență cu motorul nostru** (block-wise, 24 pași):
   `TF_PROMPT="Scrie un quicksort in python." modal run infra/modal_infer_open.py`
   - `TF_MODE=single` = o singură fereastră mare; `TF_MODE=block` (implicit) = pe blocuri.
   - `TF_MODEL_DIR=<path din volum>` — de unde se încarcă checkpointul (implicit folderul
     descărcat la pasul 1).
   - `TF_STOP_AT_EOS=true` — oprește la `EOS` (implicit `false`, deci produce exact
     `TF_MAX_NEW_TOKENS`).
   - `TF_GPU=A10G` etc. (implicit CPU). Măsurat pe A10G: **~215 tok/s** la 24 pași (blocuri).
3. **Antrenare** — `modal run infra/modal_train.py -- --config config/train_config.yaml`
   (checkpoint-uri în `/vol/checkpoints`, opțional R2).
4. **Evaluare** — `python eval/eval.py --ckpt <path>`.

### Exemplul 1 — completare de cod (blocuri)

```bash
TF_PROMPT="Write a quick sort algorithm in python." TF_MAX_NEW_TOKENS=512 \
  TF_MODE=block TF_STEPS=24 TF_GPU=A10G modal run infra/modal_infer_open.py
```

Produce o implementare de quicksort (aproximativ 512 tokeni), în ~2.4s pe A10G (24 pași).

### Exemplul 2 — alt prompt, fereastră unică

```bash
TF_PROMPT="Write a Python function that checks if a given string is a palindrome." \
  TF_MAX_NEW_TOKENS=256 TF_MODE=single TF_STEPS=24 TF_GPU=A10G \
  modal run infra/modal_infer_open.py
```

Aceleași moduri (`single` / `block`), același motor; schimbi doar promptul și numărul de tokeni
generați. Motorul este auto-conținut: primește `mask_token_id` din tokenizer și `forward(x)->logits`
din model.

## Credențiale (doar env, niciodată în repo)

R2: `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`.
HF: `modal secret create hf HF_TOKEN=<token>`.

## Limitări / de validat

- Inferența curentă este **de referință PyTorch**, nu runtime-ul SIMD final (ADR 0003).
- **Atenție bidirecțională**: se impune printr-o **mască 4D all-zero** la forward (nu prin
  `is_causal=False`, care este suprascrisă de `_update_causal_mask`). Trebuie validată pe fiecare
  arhitectură (HF variază) — vezi `.agents/AGENTS.md` și `docs/porting-to-diffusion.md`.
- **Block-wise cu context în creștere** (până la 2048) depășește fereastra de antrenare (256) pentru
  modelul nostru; ultimele blocuri se pot degrada ușor (repetiție) — luăm în calcul ferestre mai lungi
  la antrenare.
- **Calitatea la 24 de pași** trebuie măsurată pe modelul antrenat (cu cât mai mulți pași, cu atât
  mai bine, la cost proporțional).
- **`mask_token_id`**: tokenizerul are `<M>` = id **151665**; `generation_config.json` conține
  `mask_token_id: 151643` (= `<|endoftext|>`/EOS) — **stale**, de ignorat.

## Stare de verificare (motorul nostru)

Pe A10G, cu motorul nostru (block-wise / single), atenție bidirecțională reală:
- **24 pași, bloc 512 tok:** ~215 tok/s, ieșire coerentă (ex. quicksort).
- **100 pași, fereastră 256 tok:** ~66 tok/s, ieșire coerentă.
- Cauza key pentru calitate este **atenția bidirecțională reală**: fără masca 4D (adică `mask=None`,
  implicit cauzal) ieșirea devine degenerată/necondiționată de prompt.
