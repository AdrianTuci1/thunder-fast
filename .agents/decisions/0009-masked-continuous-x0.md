# ADR 0009 — Obiectiv: mascare continuă + `x0` (în loc de Gaussian + `epsilon`)

**Status:** Accepted
**Data:** 2026-09-03
**Supersedează:** interpretarea din ADR 0008 (miza doar pe `epsilon`); extinde ADR 0001.

## Context

Inferența de referință pe `step_14939` (979M tokeni) a produs **garbage** (chineză/ebraică/poloneză
amestecate). Diagnosticul `infra/diag_loss.py` pe `step_14939` a **respins ipoteza de colaps
posterior** și a arătat:

- `pred per-token std=0.7589` (modelul NU emite ~0).
- `mse[x0] MASKED = 0.000100` — modelul reconstruiește excelent embedding-ul curat la pozițiile
  **mascate**.
- `mse[eps] UNMASKED = 0.003963` — la pozițiile **nemascate** (zgomotate) prezice `epsilon`.
- Program `linear` + `beta_end=0.02` + 24 pași => `ā(t=1)=0.784` (zgomotul maxim reține 78% din
  semnal — verificat numeric). Embeddings minuscule (`x0`-norm ~0.2).

**Cauze:**
1. Calea de inferență `sample()` folosește exclusiv Gaussian + `epsilon` pe toate pozițiile,
   **fără mascare** — dar forța modelului e reconstrucția pozițiilor **mascate**. Cu `ā(t=1)=0.784`
   și scara mică a embeddings, eroarea mică de `epsilon`, după conversia
   `x0=(x−√(1−ā)·eps)/√ā`, devine comparabilă cu embedding-ul => decod `argmax(lm_head)` pe
   vectori off-manifold => tokeni greșiți.
2. Generare mascată pe checkpointul existent (init la `mask_emb` + unmask progresiv) a dat
   **garbage repetitiv** ("Furniture Furniture…", "paren paren…") pentru că modelul a fost antrenat
   doar cu `mask_ratio=0.25` pe poziții aleatorii + Gaussian + țintă `epsilon` — nu a învățat
   generarea de secvență lungă (mascare masivă).

## Decizie

Trecem la un obiectiv **de mascare (absorbing), continuu**:
- O fracțiune de poziții e înlocuită cu `mask_embedding` (stare absorbantă);
- Pozițiile **nemascate rămân la embedding-ul CURAT** (`x_t = x0`) — **fără zgomot Gaussian**, care
  era declanșatorul colapsului din ADR 0008;
- Modelul prezice **`x0`** (embedding curat) peste tot; pozițiile mascate se reconstruiesc din context;
- **Curriculum de mascare**: fracțiunea mascată se eșantionează per batch în `[0.1, 0.9]`, ca
  modelul să învețe și generare din mascare masivă;
- **Timestep legat de fracțiunea mascată** (`t = r`), ca time-conditioning-ul să transporte
  semnalul "cât e încă ascuns";
- **Generare** prin unmask progresiv din `mask_emb` (metoda `DiffusionLM.generate`, LLaDA-style
  continuu) + **decod cosine nearest-neighbor** la matricea de embeddings;
- **Reantrenare de la step 0** (checkpointul vechi poartă obiectivul degenerat).

## Alternativa respinsă (momentan)

- Rămânem pe Gaussian + `schedule="cosine"` (ā(1)=0) + `prediction="v"` + zgomot pur la generare:
  mai aproape de ADR 0001, dar continua să rămână fragilă (repetiție/colaps) — nerecomandat pe
  baza evidenței.
- **Mascare discretă (LLaDA)**: dovedită (LLaDA-8B), dar nu mai e "continuă" și schimbă runtime-ul —
  se reia doar dacă run-ul lung continuu nu dă text coerent.

## Consecințe

- Smoke (80 pași din 0, `/vol/checkpoints/v2-masked-x0/step_75.pt`): **loss 1.91→0.28→0.025**, fără
  NaN, **fără semn de colaps** — o îmbunătățire clară față de `epsilon`.
- La 80 pași (5M tokeni) generarea e încă repetitivă — **normal** la scara asta. Testul real e
  rularea lungă (~576M tokeni / 4h cu config-ul actual), amânată de o decizie de utilizator
  (nu se cheltuiește compute acum).
- **Risc rezidual:** difuzia continuă pe embeddings rămâne intrinsec fragilă; dacă run-ul lung
  tot dă text repetitiv/incoerent, migrăm la mascare discretă (LLaDA).

## Status

ACCEPTED — implementat (config + `diffusion.training_loss` + `DiffusionLM.generate` + decode
cosine-NN). Rulare lungă: **pending** (decizie utilizator).
