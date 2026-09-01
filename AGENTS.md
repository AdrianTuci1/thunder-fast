# AGENTS.md — ghid pentru agenți care lucrează în acest repo

## Memoria proiectului

Începe prin a citi [`./.agents/README.md`](./.agents/README.md). Memoria de proiect este în
`.agents/`:

- `.agents/memory.md` — viziune, constrângeri, decizii curente, întrebări deschise.
- `.agents/roadmap.md` — foaia de parcurs.
- `.agents/decisions/` — jurnal ADR.
- `.agents/research/` — note fundamentate pe surse reale.

## Reguli

- **Nu instala dependențe local**; training/eval rulează pe imagini cloud (Modal/RunPod) cu
  `requirements.txt`. Local doar `py_compile` pentru verificare de sintaxă.
- Documentează deciziile în `.agents/` (memory + ADR).
- Fundamentăm totul pe informații reale; nu inventăm cifre, biblioteci sau API-uri.
- Credențialele (R2, Modal, RunPod) sunt doar în env — nu se comit.

## Cum validez scaffolding-ul (fiecare modul)

- `src/train/diffusion.py`, `src/train/model.py`, `src/train/data.py`, `src/train/train.py`,
  `convert/`, `eval/` — verifică **sintaxa** (py_compile) și apoi **o rulare scurtă de smoke test**
  (`--max-steps` mic) pe cloud.
- **Atenție bidirecțională:** `make_bidirectional` este best-effort pentru familia Llama/Qwen;
  **verifică** pe arhitectura exactă aleasă (HF variază între versiuni). Dacă nu se poate dezactiva
  cauzalitatea elegant, înlocuiește cu `_attn_implementation="flash_attention_2"` + mască all-ones
  testată.
- **Program de zgomot:** `cosine_alpha_bar` vs `linear` trebuie verificate să coincidă cu ce
  presupune bucla `sample()` (nu amesteca cele două căi).
- **Infra:** `infra/r2.py`, `modal_train.py`, `runpod_launch.py`, `Dockerfile.runpod` necesită
  credențiale; se validează printr-un upload/download de probă pe R2.

## Plan smoke test (validarea pipeline-ului în ~1h pe 1× H100)

Rulăm un `--max-steps` mic (ex. 40–80 pași reali) pe cloud pentru a valida întregul lanț înainte
de rularea completă. Checkpoint-urile merg în R2, deci putem relua.

Ordinea de verificare:
1. **Încărcare model 1.5B + conversie atenție bidir** — `DiffusionLM` se inițializează, `input`
   embeddings corecte (verifică fără segfault/OOM).
2. **Loss scade** — rulezi 20–40 de pași și te uiți la `loss` (trebuie să scadă, nu NaN/inf).
3. **Checkpoint local** — `save_ckpt` scrie `.pt` + `.meta.json`; reîncarcă și compară tensorii.
4. **Upload R2 + resume** — după primul checkpoint, întrerupi și repornești; `train.py` trebuie să
   reia de la `step_X` (verifică log "Resumed from step ...").
5. **Sample/decodare** — `eval.py` rulează `sample()` la 18 pași și decodează tokeni (nu trebuie
   să fie coerent — doar nu garbage/special-only).
6. **Eval care rulează** — `reconstruction_loss` se calculează (valoare finită).
7. **Granițe de memorie** — batch-ul configurat încape pe 80GB; dacă OOM, reduci `batch_size_seq`.

Comanda de bază:
- Modal: `modal run infra/modal_train.py -- --max-steps 80`
- RunPod: rulează `python src/train/train.py --config config/train_config.yaml --max-steps 80` în container.

## Puncte de extensie (modularitate)

Arhitectura e separată pe responsabilități, ca fiecare piesă să poată fi schimbată/întinsă independent:

- **Program de zgomot / difuzie** — `src/train/diffusion.py` (`ContinuousDiffusion`). Adaugă variante
  noi (ex. mai multă mascare, schedule exponențial, predare v/epsilon/x0) fără a atinge modelul.
- **Model / backbone** — `src/train/model.py` (`DiffusionLM`). Schimbă arhitectura de bază (Qwen/Llama/Gemma)
  sau modul de condiționare de timp (AdaLN în loc de MLP) doar aici.
- **Date** — `src/train/data.py` (`PackedDataset`). Adaugă surse prin `config` (`data.sources`),
  fără a modifica trainingul.
- **Training** — `src/train/train.py`. Single entrypoint; oricând poți adăuga scheme noi de optimizator,
  gradient accumulation, EMA, etc.
- **Infrastructură GPU** — `infra/` (Modal + RunPod). Aceleași checkpoint-uri R2, deci portabil.
- **Evaluare** — `eval/eval.py`. Hook-uri pentru LiRo/FLORES/WMT; adaugă metrici fără a atinge training.
- **Runtime de inferență** — (de construit) separat de training; poate împărți kernel-urile ggml în
  module per micro-arhitectură pentru a fi extins per CPU.

Regula: nu cupla modulele între ele; folosește config + interfețe simple (de ex. obiectul
`ContinuousDiffusion` e singura sursă de adevăr pentru programul de zgomot, folosit și în `sample()`).
