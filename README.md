# thunder-fast

Model de limbaj bazat pe **difuzie continuă** (non-autoregresiv), adaptat prin tehnica
**DiffuLLaMA**, care generează **256 de tokeni în paralel** prin **18 pași de difuzie** (stabilitate; ADR 0007),
cu **atenție bidirecțională**, rulând doar pe **CPU** (Apple Silicon, Linux, Windows, VPS AMD/Intel)
cu kernel-uri **SIMD** (AVX2/AVX-512/AMX/NEON/SVE/VNNI) și cuantizare **Q4**.

> Stare: **planificare + scaffolding**. Nu s-au instalat dependențe local.
> Memoria proiectului: [`.agents/`](.agents/).

## Structura repository-ului

```
.agents/              memoria proiectului (decizii, roadmap, research)
config/               train_config.yaml (model, buget, difuzie, date)
src/train/            training diffuzie (model, diffusion, data, train)
convert/              convert_to_diffusion.py (AR -> difuzie inițial)
eval/                 eval.py (română + sanity generation)
infra/                Modal + RunPod + R2 (runners, Dockerfile, r2.py)
```

## Cum rulezi pipeline-ul (cloud)

Nu instala nimic local. Se folosesc GPU din cloud:

1. **Convertire** (creează checkpoint-ul de difuzie inițial din modelul AR):
   - se rulează o dată, pe orice mașină cu GPU (Modal/RunPod), via `convert/convert_to_diffusion.py`.

2. **Training** — un singur entrypoint partajat:
   - **Modal:** `modal run infra/modal_train.py -- --config config/train_config.yaml`
   - **RunPod:** build imagine din `infra/Dockerfile.runpod` + rulează `src/train/train.py`.
   - Salvare în **R2**: `src/train/train.py` încarcă checkpointurile periodic la prefixul
     configurat din `config.train_config.storage.r2_prefix` și reia automat de la ultimul.

3. **Evaluare:** `python eval/eval.py --ckpt <path>` (reconstruction loss + eșantion gen.).

4. **Inferență CPU:** încă de construit — runtime custom peste kernel-urile ggml (ADR 0003).

## Credențiale (doar env, niciodată în repo)

R2: `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`
Modal secret: `modal secret create r2 ...` (vezi `infra/modal_train.py`)
RunPod: `RUNPOD_API_KEY` (+ variabilele R2 de mai sus)

## Limitări / de validat

- Scripturile de infra sunt **scaffolding** — trebuie rulate o dată pe cloud pentru validare.
- Schema exactă de atenție bidirecțională și programul de zgomot trebuie verificate pe arhitectura
  aleasă (vezi `.agents/AGENTS.md`).
- Runtime-ul custom + kernel-urile SIMD sunt încă în plan, nu implementate.
