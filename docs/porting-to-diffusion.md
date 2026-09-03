# Portarea unui LLM autoregresiv la difuzie mascată discretă

Documentul explică metoda prin care transformăm un LLM autoregresiv (AR) — în cazul nostru
**Qwen3-0.6B** — într-un **Masked Diffusion Model (MDM)** discret, non-autoregresiv. Scopul este să
reținem greutățile pre-antrenate și să schimbăm doar *ceea ce face modelul*: în loc să prezică tokenul
următor, denoisează secvența pornind dintr-o stare plină de `[MASK]`.

Implementarea trăiește în `src/train/model.py` (`DiffusionLM`), `src/train/diffusion.py`
(`MaskedDiffusion`), `src/train/train.py` și `eval/eval.py`.

---

## 0. Ce înseamnă "difuzie mascată discretă"

- **Autoregresiv (AR):** generează token cu token; la pasul *t* vede doar prefixul `x[0..t-1]`.
- **Mascare discretă (MDM):** pornește cu o secvență în care o parte din poziții sunt **mascate**
  (înlocuite cu tokenul special `[MASK]`) și reconstruiește *toate* pozițiile mascate **în paralel**,
  la fiecare pas de difuzie.

Diferența practică: la AR costul crește cu lungimea secvenței; la MDM costul crește cu **numărul de
pași** de denoising (fix, ex. 24), independent de lungimea ferestrei.

---

## 1. Cele patru schimbări structurale

### 1.1 Atenție bidirecțională

Modelul AR folosește o **mască cauzală** (poziția *i* vede doar pozițiile ≤ *i*). Pentru denoising
avem nevoie ca fiecare poziție să vadă **întregul context** (inclusiv pozițiile din dreapta), ca să
poată reconstrui tokeni mascați folosind vecini de ambele părți.

**Atenție (capcană de versiune HF):** setarea `is_causal=False` pe module **nu este suficientă**.
Metoda `_update_causal_mask` din transformers construiește intern o mască 4D **cauzală** oricând
`attention_mask` este `None` (sau 2D), iar acea mască este pasată direct kernelului de atenție și
**suprascrie** `is_causal`. Soluția corectă:

```python
L = input_ids.shape[1]
attn_mask = torch.zeros((1, 1, L, L), dtype=torch.bool, device=input_ids.device)  # all-unmasked
out = backbone(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
```

O mască 4D all-zero (respectiv all-`False` la bool) este folosită direct de `_update_causal_mask`,
deci pozițiile se văd toate reciproc. Fără această mască, modelul rămâne cauzal și produce ieșiri
degenerate (repetă tokeni de frecvență înaltă, necondiționate de prompt) la antrenare și la inferență.

> **Verificare obligatorie:** comportamentul `_update_causal_mask` variază între versiuni de
> transformers și între arhitecturi. Dacă pe arhitectura aleasă masca 4D nu este folosită direct,
> înlocuim cu `_attn_implementation="flash_attention_2"` + mască all-ones testată.

### 1.2 Token discret de mascare `[MASK]`

Adăugăm un token special `[MASK]` în vocabular (reprezintă "zgomotul" / poziția necunoscută) și
mărim embeddings-urile:

```python
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
model.resize_token_embeddings(len(tokenizer))
```

La antrenare, pozițiile mascate primesc `input_ids = mask_token_id`; la inferență, pozițiile de generat
încep ca `[MASK]` și sunt descoperite progresiv.

### 1.3 Re-țintirea head-ului la reconstrucția tokenilor mascați

În loc de cross-entropy "next-token", obiectivul este **reconstrucția** tokenilor curați la pozițiile
mascate. `forward(input_ids) -> logits [B, L, V]` produce logits la fiecare poziție; loss-ul se aplică
**doar** pe pozițiile cu `[MASK]`.

### 1.4 Obiectiv de antrenare (CE mascată + termen de "path")

```python
mdm_loss = CE(logits[mask], x0[mask])            # reconstrucția pozițiilor mascate
path_loss = exp(-mdm_loss) * mdm_loss * (1/mask_ratio)  # re-ponderare pe calea de denoising
loss = mdm_loss + path_loss
```

Rapoartele de mascare sunt eșantionate uniform dintr-un interval (ex. `[0.002, 0.998]`), deci modelul
învață să gestioneze de la foarte puține la aproape toate pozițiile mascate.

---

## 2. Generarea (inferență)

Deoarece modelul reconstruiește poziții mascate, generarea este un **unmask progresiv**:

1. Inițializează pozițiile de generat ca `[MASK]`.
2. La fiecare pas: `logits = forward(x)`, apoi **shift next-token**
   (`logits = cat([logits[:, :1], logits[:, :-1]], dim=1)`) — aliniere identică cu antrenarea.
3. Eșantionează tokeni candidați + o **măsură de încredere** (implicit **entropia**; alternative
   `topk_margin`, `p2`).
4. Descoperă cele mai "încrezătoare" poziții; cele rămase rămân mascate. Repetă pentru pașii rămași.

Algoritmi de unmask: `entropy` (implicit), `p2` (re-mascare a pozițiilor cu încredere mică),
`origin` (transfer aleatoriu). Parametrul `alg_temp` (implicit `0.6`) îndulcește distribuția pozițiilor
de unmask.

### Ieșiri lungi — generare pe blocuri

O fereastră antrenează pe `seq_len` (ex. 256), dar vrem până la 2048 tokeni. Folosim **generare pe
blocuri**: generăm un bloc de `block_len` tokeni, îl adăugăm ca prefix, apoi generăm următorul bloc.
Contextul crește la fiecare pas. `stop_at_eos` oprește mai devreme dacă apare `EOS`.

> **Limitare cunoscută:** contextul agregat depășește fereastra de antrenare, deci ultimele blocuri se
> pot degrada (repetiție). Soluție: ferestre mai lungi la antrenare sau includerea de context lung.

---

## 3. Unde este implementat

| Componentă | Fișier | Rol |
|---|---|---|
| Model + adaptare | `src/train/model.py` | `make_bidirectional`, `[MASK]`, `forward`, `generate`/`generate_long` |
| Obiectiv MDM | `src/train/diffusion.py` | `MaskedDiffusion` (CE mascată + path loss) |
| Antrenare | `src/train/train.py` | bucla de training, optimizator, resume |
| Date | `src/train/data.py` | `PackedDataset` |
| Eval | `eval/eval.py` | `reconstruction_loss` + generare |
| Inferență de referință | `infra/modal_infer_open.py` | `_discrete_generate_window` / `_discrete_generate_long` |
| Runtime CPU SIMD | `src/infer/` | ținta finală (ADR 0003), kernel-uri ggml adaptate |

---

## 4. Validare (smoke test)

Înainte de rularea lungă, validăm întregul lanț cu un `--max-steps` mic (ex. 40–80) pe cloud, în
ordinea:

1. modelul se inițializează + `forward` merge (fără crash/OOM),
2. **loss-ul scade** (nu NaN/inf),
3. salvare/reîncărcare checkpoint (`save_ckpt` → `.pt` + `.meta.json`),
4. upload pe R2 + resume de la `step_X`,
5. generare la 24 de pași (decodare → nu garbage/special-only),
6. `reconstruction_loss` finită,
7. încape în memoria GPU-ului (altfel reducem `batch_size_seq`).

Atenția bidirecțională se verifică explicit în pasul 2/5 (fără masca 4D, modelul ar rămâne cauzal).
