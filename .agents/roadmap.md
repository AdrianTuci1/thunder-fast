# roadmap.md — foaia de parcurs thunder-fast

> Ordinea de lucru propusă. Fiecare fază are un obiectiv verificabil. Nu am instalat încă dependențe.

---

## Faza 0 — Decizii & schelet (ÎN CURS)
- [x] Creăm `.agents/` + memoria.
- [x] Confirmăm varianta de difuzie (continuă Gaussian + mascare; ADR 0001/0004).
- [x] Decidem runtime: binar custom peste kernel-urile ggml (ADR 0003).
- [x] Confirmăm buget: 0.5–1B, ~20B tokeni, 24h (ADR 0005) + infra Modal/RunPod/R2 (ADR 0006).
- [x] Creăm scheletul repo-ului (foldere + AGENTS.md + README).
- [x] Scaffolding pipeline: `src/train/*`, `convert/*`, `eval/*`, `infra/*`, `config/*` (NEVALIDAT — de testat pe cloud).
- [ ] Alegem modelul de bază exact (`MODEL_ID`: Qwen2.5-0.5B vs 1.5B).

## Faza 1 — Fundamente (research + eval)
- [ ] Elevăm modelul de bază pe română (benchmark de calitate, viteză pe CPU).
- [ ] Verificăm suportul real al modelului ales pe română; eventual fine-tune de limbă.
- [ ] Studiem DiffuLLaMA/LLaDA (cod + paper) pentru detaliile de adaptare.
- [ ] Stabilim set de date român (train + eval).

## Faza 2 — Adaptare la difuzie
- [x] ~Convertire checkpoint AR -> difuzie~ — schelet în `src/train/model.py` + `convert/convert_to_diffusion.py` (NEVALIDAT).
- [x] ~Implementăm bucla de denoising~ — schelet în `src/train/diffusion.py` + `model.sample()` (NEVALIDAT).
- [x] ~Training de difuzie la scară mică~ — schelet în `src/train/train.py` (NEVALIDAT).
- [ ] Rulare reală pe cloud (Modal/RunPod) + validare calitate vs model autoregresiv de referință.

## Faza 3 — Cuantizare & runtime
- [x] ~~Integrare în llama.cpp (fork) SAU binar propriu~~ — **DECIS (ADR 0003):** binar custom
      peste kernel-urile ggml.
- [ ] Cuantizare Q4 (Q4_K_M / Q4_0) + activări Q8_0.
- [ ] Conversie checkpoint → GGUF + motor de difuzie (atenție bidirecțională + buclă 24 pași).
- [ ] Verificare că inferența difuzie rulează corect cu weight-uri cuantizate.

## Faza 4 — Kernel-uri SIMD (contribuția principală; adaptare la pattern-ul nostru)
- [ ] **Baseline:** rulează corect cu kernel-urile ggml stock (referință corectitudine + viteză de start).
- [ ] **Adaptare atenție:** kernel de **atenție non-cauzală** (fără mască/KV-cache).
- [ ] **Adaptare GEMM:** tunning la batch de 256 tokeni ([256 × hidden] × weights).
- [ ] **Maximizare eficiență:** x86 AVX2 -> AVX-512 -> AVX-512 VNNI -> AMX; ARM NEON -> I8MM/DotProd -> SVE; AMX Apple.
- [ ] Dispecerizare runtime (CPUID/features).
- [ ] Bench: tokens/sec pe 256 tokeni, pe Apple Silicon + VPS AMD/Intel.

## Faza 5 — Optimizare & scalare
- [ ] Optimizare program de zgomot, număr de pași, sampling.
- [ ] Posibil early-exit (oprire când secvența e "curată").
- [ ] Tune calitate vs viteză (24 pași -> posibil mai puțini).

## Faza 6 — Livrare & documentare
- [ ] Scripturi de export model + binar.
- [ ] Documentație: build pe Linux/Windows/macOS, VPS.
- [ ] Benchmark final + raport de evaluare (inclusiv română).
