# ADR 0004 — Atenție bidirecțională pe secvențe de 256 de tokeni

**Status:** Accepted
**Data:** 2026-08-31

## Context
Generarea se face în paralel pe un bloc de 256 de tokeni, cu atenție bidirecțională (confirmat de
utilizator: "atentie bidirectionala dar in batch-uri de 256 de tokens").

## Decizie
- Lungimea secvenței pentru un pas de denoising este **N = 256 de tokeni**.
- Atenția este **bidirecțională** (non-cauzală): fiecare poziție vede toate cele 256.
- Deoarece N=256, complexitatea atenției complete O(n²) e mică (256×256 = 65.536), deci:
  - **nu folosim KV-cache** (nu e necesar și nu se aplică natural la denoising);
  - folosim atenție completă simplă pe matricea [256 × hidden].
- "Batch" înseamnă fie generația de 256 tokeni, fie mini-batch-uri de secvențe de lungime 256 la training.

## Consecințe
- Fără grijă de memorie/latency la nivel de atenție pentru N=256.
- La training, putem face batch de secvențe de lungime 256 (padding la 256).
- Kernel-ul de atenție trebuie să suporte **non-cauzal** (fără mască, sau mască all-ones) — scriem
  acest agregat în motorul nostru, peste kernel-urile ggml.
