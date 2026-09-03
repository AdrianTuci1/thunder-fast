# ADR 0011 — Atenția bidirecțională se impune prin mască 4D, nu prin `is_causal=False`

**Status:** Accepted
**Data:** 2026-09-03
**Corectează:** presupunerea din ADR 0010 ("forward discret cu `attention_mask=None` + `is_causal=False`").

## Context

Inferența prin motorul nostru producea **garbage repetitiv** la orice mărime de pași (24/64/200),
prompt-independent. Pentru a găsi cauza am urmărit firul: modelul denoisează corect de la pași mici,
deci problema nu era în programul de denoising, ci în modul în care rulam atenția.

Am stabilit pe rând că **nu** e vina la:
- **greutăți / încărcare** — checkpointul se încarcă complet, embeddings legate cu `lm_head` prin
  `tie_word_embeddings`, număr de parametri consistent cu config-ul; deci greutățile sunt corecte.
- **bucla de sampling** — bucla noastră de unmask (`_discrete_generate_window`) este logic identică
  cu calea de referință a modelului (Gumbel-TopK ≡ softmax+multinomial; shift next-token identic).
- **transformers versiune** — `_update_causal_mask` este self-contained, deci masca nu depinde de
  versiunea instalată.

**Cauza reală (demonstrată prin experimente pe GPU):** cu același model, rulând **bucla noastră** cu
atenție **cauzală** → garbage; cu atenție **bidirecțională** (mască 4D all-zero) → **cod corect**
(quicksort coerent). Concluzia: modelul MDM necesită atenție bidirecțională, iar motorul nostru rula
de fapt cauzal.

**De ce era cauzal:** HF `_update_causal_mask` construiește o mască 4D **cauzală** de fiecare dată când
`attention_mask` este `None` (sau 2D). Această mască e pasată direct kernelului de atenție și
**suprascrie** `is_causal=False`. Deci:
- `is_causal=False` cu `attention_mask=None` → rămâne cauzal (no-op).
- `is_causal=False` + mască 4D cauzală pasată de model → rămâne cauzal.
- Singura cale de a obține bidirecțional: **mască 4D all-zero** (sau all-`False` la bool), care e folosită
  direct de `_update_causal_mask` și face fiecare poziție să vadă toate celelalte.

## Decizie

1. **Inferență** (`infra/modal_infer_open.py`): forward-ul nostru trece o **mască 4D all-zero**:
   ```python
   m = torch.zeros((1, 1, L, L), device=device)
   return model(input_ids=x, attention_mask=m, is_causal=False).logits
   ```
2. **Motorul nostru** (`src/train/model.py`): `DiffusionLM.forward` trece același tip de mască
   (bool all-`False` 4D) către backbone, în loc de `attention_mask=None`. `make_bidirectional` rămâne
   pentru calea SDPA, dar docstring-ul documentează că NU e suficientă de una singură.

## Consecințe

- `attention_mask=None` la un model HF de difuzie = **cauzal** (greșit) → nu mai folosim.
- Fixul trebuie **validat pe arhitectura Qwen3 exactă** (smoke test pe cloud) — HF variază între versiuni
  (AGENTS.md): dacă `_update_causal_mask` nu folosește direct masca 4D la Qwen3, înlocuim cu
  `_attn_implementation="flash_attention_2"` + mască all-ones testată.
- Inferență cu motorul nostru: 66.2 tok/s la 100 pași / 264 tokeni (fereastră unică), output coerent.

## Status

ACCEPTED — implementat și verificat pe GPU (bidirecțional → cod coerent; cauzal → garbage).
Validarea pe Qwen3-0.6B (smoke cloud) e **pending**.
