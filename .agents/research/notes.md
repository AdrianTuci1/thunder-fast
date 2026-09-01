# research/notes.md — note fundamentate (surse reale)

> Note scurte, cu link-uri, pentru ca planul să fie bazat pe informații reale, nu inventate.
> De completat pe măsură ce cercetăm mai departe.

## 1. DiffuLLaMA (adaptarea unui model LLaMA la difuzie)

- Definește modele de limbaj bazate pe difuzie obținute prin **adaptarea unui checkpoint LLaMA**:
  modelul vede secvența completă (parțial zgomotată) și reface tokenii curați, iterativ.
- **Atenție bidirecțională** (non-cauzală, ca BERT): fiecare poziție vede toate celelalte.
  Masca cauzală ar dăuna obiectivului de denoising.
- **Generare paralelă:** toate pozițiile sunt actualizate simultan la fiecare pas de denoising.
  Câștig apare la ieșiri lungi; costul = numărul de forward-uri (pași) + cache/overhead.
- Proces de zgomot: fie mască absorbantă (discret), fie Gaussian pe embeddings (continuu).
- Pași tipici: 8–64; **24 de pași** e un punct de operare comun bun pentru calitate/viteză.

**Surse de consultat în continuare:** paper-ul DiffuLLaMA + repo-ul de cod, paper/repo LLaDA.

## 2. LLaDA (Large Language Diffusion Models)

- Generează prin **denoising iterativ**, nu left-to-right. Secvența întreagă (128/256) e rafinată
  în paralel, de obicei **24 de pași**, pornind din zgomot sau [MASK].
- Diferența practică față de GPT: ieșirile mai lungi devin relativ mai ieftine, dar plătești un
  număr **fix de forward-uri** (24).

## 3. Kernel-uri SIMD & cuantizare (llama.cpp / GGML)

- GGML folosește kernel-uri specifice arhitecturii pentru GEMV/GEMM cuantizat (familia `ggml_vec_dot_*`),
  cu **dispecerizare runtime** prin detecție de CPU (`ggml_cpu_has_avx512()`, `ggml_cpu_has_avx512_vnni()`, NEON...).
- **Formate:** Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ*. Blocuri packate (32/256 valori) cu scale/min per bloc.
- **x86:**
  - AVX-512 = 512-bit (16×FP32); **AVX-512 VNNI** adaugă `VPDPBUSD`/`VPDPBUSDS` pentru MAC int8 —
    ideal pentru Q4/Q8 dot products.
  - **AMX** (Intel Sapphire Rapids+) = operații pe tile-uri, bun la int8 MAC; necesită tratament special.
  - Runtime preferă VNNI când CPU și tipul de cuantizare suportă calea int.
- **ARM:**
  - **NEON** (128-bit) cu DotProd (v8.2+) / I8MM (v8.6+): `vdotq_s32`, `vmull`+`vpadal`, `vfmaq`.
  - **SVE** unde există (ex. Graviton/Grace).
  - **Apple Silicon**: AArch64 => NEON; **AMX Apple** (matrix coprocesor) există dar în llama.cpp se
    folosește mai ales **Metal GPU** pentru viteză; pe CPU e NEON.

## 4. DiffuLLaMA — acoperirea "rețetei" de adaptare (confirmat)

Varianta **DiffuLLaMA (2025) cu embeddings continue + mascare** e exact ce ne trebuie pe training:
- Atenție **cauzală → bidirecțională** (full).
- **Condiționare pe timestep** (AdaLN / embeddings / LoRA).
- Obiectiv: next-token → **denoising loss** pe embeddings (noisy/masked).
- **Difuzie continuă pe embeddings** (Gaussian + absorbție/mascare) + recuperare tokeni prin
  **nearest-neighbor lookup** sau **softmax head** pe embeddings finale.
- **Inițializare din checkpoint AR pre-antrenat** (păstrăm tokenizer + backbone; doar condiționare de timp + head).

**Concluzie:** partea de **training** e acoperită de DiffuLLaMA. Ce rămâne al nostru:
runtime-ul custom + adaptarea kernel-urilor + tunarea programului de zgomot/remasking.

## 5. Referință de compute — LLaDA-8B (de la zero, pentru comparație)

- LLaDA-8B: **2.3T tokeni**, ~6ND ≈ **1.1×10^23 FLOPs**, ~**2.800 H800 GPU-days**
  (utilizare ~40–50% MFU), echivalent **1–3 săptămâni** pe 128–256 GPU. Cost cloud estimat:
  **zeci-sute de mii USD** (la ~$2–4/GPU-hr). (Estimări — verifica paper-ul original.)
- **Adaptarea AR→difuzie e mult mai ieftină decât pretraining de la zero** — de asta pornești de la
  un checkpoint pre-antrenat.

## 6. Date pentru un model multilingual (română)

- **Bază alesă (Qwen) e deja multilingual** => româna e parțial acoperită; completăm prin
  **continuare de pre-training pe corpus mixt + română intensivă**.
- **Monolingual română:** mC4/OSCAR ro, Wikipedia ro, CommonCrawl ro, corpus de știri.
- **Paralel EN–RO:** Europarl, JRC-Acquis, OPUS, Tatoeba, WMT (bun pt transfer cross-lingual + traducere).
- **Multilingual general:** se păstrează cunoștințele din pre-training-ul de bază; adăugăm româna curată.
- **Eval:** LiRo (NLU), FLORES + WMT (EN–RO), seturi de instrucțiuni/chat în română
  (RoMMLU, RoGSM8K etc.), plus bench-uri generale.

## 7. Timp & cost de antrenament — estimare pentru portarea noastră (1B–1.5B)

- Adaptarea e **mult mai ieftină decât LLaDA-8B de la zero**. Pentru **1.5B**, un buget tipic de
  continuare de pre-training pe difuzie: **zeci de miliarde de tokeni** (ex. 10–100B).
- FLOPs estimat: 6 × 1.5e9 × (ex. 50e9) ≈ **4.5×10^20** — de sute de ori mai mic decât LLaDA-8B.
- Hardware: **4–8 × A100/H100 pentru câteva zile** (sau mai puțin pe mai multe GPU). Cost cloud
  estimat: **mii – zeci de mii USD** (nu sute de mii).
- **Atenție:** inferența rulează pe **CPU** (target), deci cost de runtime la deploy ≈ **0 GPU**.
  Antrenamentul e o cheltuială separată, doar în faza de adaptare.

### Punct de plecare existent? (verificat)
- **Nu există** un checkpoint de difuzie mic (1B/1.5B) public, multilingual. LLaDA a publicat doar
  **8B** (engleză-centric, nu strong multilingual). DiffuLLaMA a publicat checkpoints LLaMA mari.
  => Trebuie să construim noi (adaptare), nu "luăm gata făcut".
- **Constrângere utilizator:** buget de **24h**, modelul ar trebui să fie "complet".
  => "Complet" trebuie definit. Adaptare reală de calitate (zeci de B tokeni) NU încape în 24h;
  **da** încape un **PoC/produs redus** dacă reducem: mărime modelului + bugetul de tokeni +
  scope-ul runtime-ului (minimal pe ggml, nu kernel-uri SIMD reglate la producție).

### Lever pe care le avem pentru "mai repede / mai mic"
- **Model mai mic** (ex. 0.5B în loc de 1.5B): rescalează FLOPs la ~1/9 și scade latența CPU.
- **Buget de tokeni redus** (ex. 1–2B în loc de 10–100B): al doilea mare multiplicator de timp.
  Adaptarea din AR init e aici avantajul-cheie: transferă cunoștințe, deci un **fine-tune scurt**
  poate da rezultate "decente" fără pre-training lung.
- **Pași de difuzie** reduși (ex. 8–12 în loc de 24): direct mai puțin timp de inferență per token.
- **Compromis clar:** "viteză imensă + calitate SOTA bilingual" **NU** e un rezultat de 24h.

### Drivers de cost (ce urmărim)
| Cost | Detaliu | Ordine de mărime |
|---|---|---|
| **Train GPU** | adaptare 1–1.5B pe zeci de B tokeni, 4–8 × A100/H100 | mii–zeci de mii USD |
| Date | open-source (OSCAR, Europarl, OPUS, WMT, Wikipedia ro) | ~0 (doar curățare/eb-uri) |
| Eval | LiRo, FLORES+WMT, seturi ro (open); LLM-as-judge opțional | mic (posibil API) |
| **Runtime deploy** | CPU (Apple Silicon / VPS AMD-Intel) — fără GPU | ~0 (scopul proiectului) |
| **Inginerie SIMD** | runtime custom + adaptare kernel-uri (timp de dezvoltare) | cost principal (persoană-zile) |

## 8. Consecințe pentru thunder-fast

- Dominanta la CPU = **matmul de-cuantizare + GEMV/GEMM**, repetat la **fiecare din cele 24 de forward-uri**.
- Trebuie **kernel-uri per micro-arhitectură** (AVX2/AVX-512/VNNI/AMX/NEON/SVE/I8MM) cu dispatch runtime,
  optimizate pentru Q4/Q8.
- llmama.cpp/GGML e autoregresiv — pentru **difuzie** trebuie **fork cu suport de diffusion** (sau
  binar propriu cu kernel-urile noastre). De verificat dacă există backend difuzie existent (LLaDA).
