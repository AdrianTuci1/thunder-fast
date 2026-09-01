# ADR 0008 — Predicția modelului: `x0` → `epsilon`

## Context

Antrenamentul adaptării (AR → difuzie continuă) a produs `loss ≈ 0.0000` și `grad_norm ≈ 0.05`
încă de la pași mici, iar diagnosticul `infra/diag_loss.py` pe `step_3821` a arătat:

- `||x0|| = 35.88` și `||pred|| = 21.68` → modelul emite valori aproape de zero.
- `mse ALL mean = 0.000154`, `masked pred mean/std = -0.0003/0.0074` vs `masked x0 mean/std = -0.0002/0.0121`
  → predicția e practic constanta 0, iar MSE ≈ varianța lui `x0`.

## Decizie

Schimbăm parametrizarea din predicție `x0` (embedding curat) în `epsilon` (zgomotul adăugat)
în `config/train_config.yaml`:

```yaml
diffusion:
  prediction: "epsilon"   # din "x0"
```

### De ce

Cu predicție `x0` pe embeddings, minimul MSE este **media embeddings** (~0), deci modelul
colapsează la soluția trivială "emit 0", pierderea devine ≈ varianța lui `x0`, iar gradienții
dispar (posterior collapse). Cu predicție `epsilon`, predicția zero dă `loss ≈ ||eps||² ≈ D`
(foarte mare), deci modelul e forțat să învețe zgomotul real; `x0` se recuperează la inferență
prin formula deja implementată în `model.sample()`.

## Alternativa respinsă

- `v` (velocity): la t mare (αbar mic) și `pred=0`, `loss ≈ (1-αbar)·||x0||²` — încă mic, risc
  de aspect triviale; `epsilon` e mai robust.
- Normalizarea `x0` / head de proiecție: mai multe schimbări; `epsilon` rezolvă cauza cu o linie.

## Consecințe

- `_prediction_target()` și `sample()` suportau deja `epsilon` — doar config-ul s-a schimbat.
- `diag_loss.py` actualizat să folosească `_prediction_target()` în loc de `target = x0`.
- La reluare din `step_3821` modelul a colapsat (emite ~0); recomand reantrenare de la `step 0`,
  altfel modelul trebuie să "decolapseze" pe parcurs.

## Status

ACCEPTED — schimbare de o linie în config; de validat printr-o rulare scurtă pe Modal.
