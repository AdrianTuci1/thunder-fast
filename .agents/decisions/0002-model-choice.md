# ADR 0002 — Modelul de bază

**Status:** Accepted
**Data:** 2026-08-31

## Context
Avem nevoie de un model ≤1B, multilingual (română), open, adaptabil la difuzie, care să ruleze pe
CPU cu Q4 **și** să încapă pe 2–4 GPU în ~24h cu un buget generos de tokeni. Un 1.5B nu se încadrează
pe 2 GPU în 24h (vezi ADR 0005), deci coborâm sub pragul 1B.

## Decizie
- **Model:** `Qwen/Qwen3-0.6B` (~0.6B / 598M).
  - Apache-2.0, LLaMA-style (ușor de adaptat la difuzie), GQA, vocab ~151k.
  - **Multilingual puternic** — mult mai bine pe română decât Qwen2.5-0.5B; potrivit ca bază pentru
    continuarea pe română.
  - Suportat în GGUF (export + Q4) pentru runtime-ul de inferență.

## Justificare mărime
- 0.6B × 20B tokeni ≈ **7.2×10^19** FLOPs.
  - 2× H100 (MFU ~50%): 20B tokeni în **~20h** (încape în 24h) sau ~23B tokeni în 24h.
  - 4× H100: 20B în **~10–12h**, sau până la **~47B tokeni în 24h** (mai mulți tokeni, cum vrem).
- Opțiunea de **1B** nu există în familia Qwen (mers 0.5→1.5).

## Alternative analizate
| Model | Motiv nefiind ales |
|---|---|
| Qwen2.5-1.5B | Calitate mai bună, dar NU încape pe 2 GPU în 24h |
| Qwen2.5-0.5B | Multilingual mai slab decât Qwen3-0.6B |
| Llama-3.2-1B | Licență restrictivă; singurul ~1B, dar nu permisiv |
| SmolLM2 | Româna slab acoperită |
| Qwen3-0.6B (ales) | Cel mai bun raport multilingual/mărime/performanță la ≤1B |

## Consecințe
- Tokenizer, config, convertor și runtime se leagă de Qwen3-0.6B.
- `MODEL_ID` rămâne configurabil prin `config.train_config.yaml` (`model.base`).
- Dacă vrem mai multă calitate pe română la scara asta: continuăm de la Qwen3-0.6B cu un mic
  fine-tune de limbă (română) — cost mic față de a porni de la un model mai mare.
