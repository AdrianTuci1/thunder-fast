# ADR 0010 — Migrare la mascare discretă (masked diffusion, MDM)

**Status:** Accepted
**Data:** 2026-09-03
**Supersedează:** ADR 0009 (mascare continuă + `x0`). Extinde/redefinește obiectivul de difuzie din
ADR 0001/0008.

## Context

După smoke-ul continuu (ADR 0009), am evaluat obiectivul de difuzie la mărimea noastră (Qwen3-0.6B).
Obiectivul continuu (ADR 0008/0009) — Gaussian pe embeddings + predicție `x0`/`epsilon` — nu scoate text
coerent la volum de antrenare mic și este sensibil la scara embeddings (vectori off-manifold). Varianta
**discretă (masked diffusion, MDM)** este mai robustă la această scară și este abordarea dominantă de a
adapta un LLM autoregresiv la difuzie cu cost mic.

**Ce implică MDM discret:**
- Un token `[MASK]` adăugat în vocabular; rapoarte de mascare eșantionate uniform din `[0,1]`;
  reconstrucție cu **cross-entropy** peste vocabular. Continuitatea Gaussiană pe embeddings
  (ADR 0008/0009) **NU** este folosită.
- Loss: `mdm_loss + path_loss` (CE mascată + termen re-weightat de `exp(-CE) * CE * (1/mask_ratio)`).
- Atenție **bidirecțională** (`is_causal=False`), fără condiționare de timp.
- Generare: unmask progresiv (algoritmi `entropy`/`p2`/`origin`), logits shift-uite `cat([logits[:,:1], logits[:,:-1]])`
  (aliniere next-token, consecventă cu trainingul), decodare `lm_head.argmax`.

## Decizie

Migrăm la **mascare discretă (discrete MDM)**, cu două constrângeri din partea utilizatorului:
1. **Păstrăm 24 pași de difuzie la inferență și ferestre de 256 tokeni.**
2. **Suportăm ieșiri mai lungi, până la 2048 tokeni**, prin **generare pe blocuri (block-wise)**:
   fiecare pas de difuzie generează 256 tokeni, îi adaugă ca context, apoi generează următorul bloc
   (contextul crește). Oprire la EOS dacă e dorită.

Consecințe asupra codului:
- `src/train/diffusion.py`: `ContinuousDiffusion` -> `MaskedDiffusion` (obiectiv CE mascat discret).
- `src/train/model.py`: se adaugă `[MASK]` + `resize_token_embeddings`; `forward(input_ids)->logits`;
  scad `time_mlp`/`mask_embedding`; `generate()` discret + `generate_long()` (blocuri).
- `src/train/train.py`: înlocuirea apelului de loss; optimizer cu un singur grup de LR.
- Config: scot cheile continue (`schedule`, `beta_*`, `prediction`, `mask_ratio`, `train_steps`).

## Alternative respinse

- **Continuare pe mascare continuă (ADR 0009):** nu a fost reținută; migrăm la discretă, mai robustă
  la această scară și cu cost de adaptare mai mic.
- **Representation alignment** (`repr_align_wt`, „Don't Retrain—Align", ~4x speedup): tehnică de adaptare
  a unui model AR cu mai puțin compute. Respinsă acum (mai multă plombare: model teacher + loss de
  aliniere); poate fi adăugată ulterior ca extensie.

## Consecințe

- **Nu mai avem program de zgomot continuu, predicție `x0`/`epsilon`/`v`, schedule sau beta.**
- **Reantrenare de la step 0:** checkpoint-urile continue (`step_14939`, `v2-masked-x0`) sunt
  structural incompatibile; `train.py` le ignoră și reia de la 0 (fallback la key-mismatch).
- **Atenție:** modelul învață pe ferestre de 256; generare pe blocuri cu context în creștere
  (până la 2048) păstrează fiecare pas într-o fereastră de 256, dar contextul agregat depășește
  fereastra de antrenare. Dacă coerența pe distanțe lungi suferă, luăm în calcul antrenare pe ferestre
  mai lungi sau includerea de context lung în date.
- **Verificare bidirecțională** (AGENTS.md): forward-ul discret (input_ids + `attention_mask=None` +
  `is_causal=False`) trebuie validat pe arhitectura Qwen3 exactă (HF variază). Vezi ADR 0011 —
  `attention_mask=None` nu este suficient; trebuie mască 4D.

## Status

ACCEPTED — implementat (config + `MaskedDiffusion` + `DiffusionLM.generate`/`generate_long`).
Rulare lungă (reantrenare MDM de la 0): **pending** (decizie utilizator privind costul de compute).
