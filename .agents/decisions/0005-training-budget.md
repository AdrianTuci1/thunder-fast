# ADR 0005 — Buget de training, mărimea modelului & GPU

**Status:** Accepted
**Data:** 2026-08-31
**Actualizat:** după alegerea modelului (ADR 0002: Qwen3-0.6B) și constrângerea de GPU 2–4

## Context
Pipeline complet (checkpointing + evaluare) într-un buget de ~24h, pe GPU Modal/RunPod, salvare R2.
Obiectiv: model multilingual să **vadă cât mai mulți tokeni** și să încapă pe **2–4 GPU**.

## Decizie
- **Model:** `Qwen/Qwen3-0.6B` (~0.6B).
- **Buget de tokeni:** **20B** (confortabil; se poate urca la ~40B pe 4 GPU dacă vrem mai multă expunere).
- **Pași de difuzie:** 18 (ADR 0007).

## Compute
- FLOPs = 6 × 0.6e9 × 20e9 ≈ **7.2×10^19** (pt 20B tokeni).
- Capacitate ≈ 11.9B tokeni/GPU/zi (H100, MFU 50%) — cu **zgomot la 40%** ≈ 9.5B/GPU/zi.

### Timp pentru 20B tokeni (H100, bf16)
| GPU | ~40% MFU | ~50% MFU | Tokeni în 24h (50%) | Tokeni în 24h (40%) |
|---|---|---|---|---|
| 1× H100 | ~21h | ~17h | ~12B | ~9.5B |
| 2× H100 | ~10.5h | ~8.5h | ~24B | ~19B |
| 4× H100 | ~5h | ~4h | ~48B | ~38B |
| 8× H100 | ~2.6h | ~2.1h | ~95B | ~76B |

## Recomandare GPU
- **Smoke test:** 1× H100 (câteva sute de pași).
- **20B tokeni în 24h:** **2× H100** (confortabil la 50% MFU, strâns la 40%) sau **1× H100** dacă
  accepți la limita de 24h.
- **Mai mulți tokeni (mai multă expunere pe română):** **4× H100** → ~40B tokeni în 24h.
- Prevăd: la 0.6B pe H100, MFU tinde spre **40–55%**; A100 80GB e alternativă mai ieftină.

## Comparativ (de ce l-am ales)
- **1.5B × 20B** ≈ 1.8×10^20 FLOPs → 1× H100 ~100–126h, 2× ~51–63h (**NU** încape 2 GPU în 24h).
- **0.6B × 20B** ≈ 7.2×10^19 → **încape pe 2 GPU** și permite **mai mulți tokeni pe 4 GPU**.

## Consecințe
- Calitate pe română mai mică decât 1.5B, dar multilingual (Qwen3) și încape în bugetul de GPU.
- Obiectivul "vezi mai mulți tokeni" e îndeplinit: putem fugi 20–40B tokeni în 24h pe 2–4 GPU.
- Checkpointing + resume (R2) obligatoriu (totuși, la 0.6B rularea completă e sub 24h chiar și pe 1–2 GPU).
