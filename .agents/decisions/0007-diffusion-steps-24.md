# ADR 0007 — Pași de difuzie: 24 (calitate)

**Status:** Accepted
**Data:** 2026-08-31
**Supersedează:** ADR 0007 (18 pași), referința la "24 de pași" din ADR 0001 și memory §5

## Context
Generarea paralelă folosește un număr de pași de difuzie. Utilizatorul a optat pentru **24 de
pași** (standardul LLaDA, tavan de calitate mai bun), acceptând costul suplimentar la inferență.

## Decizie
- **Pași de difuzie = 24**, folosit la **și antrenament și inferență** (consistent, ca programul
  de zgomot să coincidă — evită mismatch-ul train/infer care degradează calitatea).
- Se pot reduce mai târziu (ex. 16 = mai rapid) pentru viteză extra, dar numai după validare.

## Consecințe
- Calitate la tavan mai bun (24 de pași e punctul de operare standard în literatură).
- Inferență mai lentă (24 de forward-uri în loc de 16/18); pe CPU Q4 acesta e costul acceptat.
- Antrenamentul folosește timestep continuu în [0,1], deci este agnostic la numărul de pași de
  la inferență — activăm 24 pentru consistența programului de zgomot.

## Ajustări
- `config/train_config.yaml`: `train_steps = infer_steps = 24`.
