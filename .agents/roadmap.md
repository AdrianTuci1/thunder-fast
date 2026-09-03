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
- [x] **Obiectiv corectat (ADR 0009):** mascare continuă + `x0` + curriculum de mascare (în loc de
      Gaussian+`epsilon` care dădea garbage/colaps). Implementat + smoke-testat (80 pași, loss
      `1.91→0.025`, fără colaps).
- [ ] **Rulare lungă de la 0** (~576M tokeni / 4h) + validare calitate vs autoregresiv.
      **PENDING** — decizie utilizator (nu se cheltuiește compute acum). Checkpoint de continuare:
      `/vol/checkpoints/v2-masked-x0/step_75.pt`.
- [ ] (de decalat la discrete LLaDA) dacă run-ul lung continuu nu dă text coerent.

## Faza 3 — Runtime ggml (ADR 0012) & cuantizare
- [x] **Decizie runtime:** motor propriu în `runtime/` peste ggml (atenție bidir + buclă de denoising)
      — **DECIS (ADR 0012)**; înlocuiește simpla direcție din ADR 0003.
- [x] Nucleu de difuzie (`runtime/src/diffusion/`) — implementat + testat (funcțional, local).
- [x] Schelet runtime: CLI + server HTTP OpenAI-compatibil, tokenizer, engine ggml, converter, upload R2.
- [ ] Build prin **workflow GH Actions dispatch** (`gh workflow run build-runtime`) → R2; compilare
      validată pe **Modal** (materializare graf ggml + mapare tensori + atenție bidir).
- [ ] Cuantizare Q4 (Q4_K_M / Q4_0) + activări Q8_0 (pe greutățile GGUF).
- [ ] Verificare că inferența difuzie rulează corect cu weight-uri cuantizate.
- [ ] Tokenizer byte-level exact (render GPT-2 + pre-tokenizare regex) + test pe vocabularul Qwen3.

## Faza 4 — Kernel-uri SIMD (DEZACTIVAT/opțional, vezi ADR 0012)
Nu mai sunt contribuția principală: ggml oferă deja kernel-urile (CPU SIMD / CUDA / Metal / Vulkan).
Devreme doar **tuning de atenție non-cauzală + GEMM** la shape 256 tokeni, dacă e nevoie de viteză
suplimentară pe CPU:
- [ ] **Baseline:** rulează corect cu kernel-urile ggml stock.
- [ ] **Adaptare atenție:** kernel de **atenție non-cauzală** (fără mască/KV-cache).
- [ ] **Adaptare GEMM:** tunning la batch de 256 tokeni ([256 × hidden] × weights).
- [ ] Bench: tokens/sec pe 256 tokeni, pe Apple Silicon + VPS AMD/Intel.

## Faza 5 — Optimizare & scalare
- [ ] Optimizare program de zgomot, număr de pași, sampling.
- [ ] Posibil early-exit (oprire când secvența e "curată").
- [ ] Tune calitate vs viteză (24 pași -> posibil mai puțini).

## Faza 6 — Livrare & documentare
- [ ] Scripturi de export model + binar.
- [ ] Documentație: build pe Linux/Windows/macOS, VPS.
- [ ] Benchmark final + raport de evaluare (inclusiv română).
