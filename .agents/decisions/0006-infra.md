# ADR 0006 — Infrastructură GPU & stocare (Modal, RunPod, R2)

**Status:** Accepted
**Data:** 2026-08-31

## Context
Antrenamentul de adaptare trebuie să ruleze pe GPU în cloud, să facă checkpointing periodic și
să salveze tot în **Cloudflare R2**. Utilizatorul folosește **Modal GPU** și/sau **RunPod GPU**.

## Decizie
- **GPU:** Modal (serverless, `modal` SDK) pentru rulări rapide/scalate și **RunPod** (pod-uri, imagine
  Docker) ca alternativă/backup. Ambele rulează același entrypoint de training.
- **Stocare:** **Cloudflare R2** (S3-compatible) ca locație unică pentru checkpoint-uri + rezultate.
  Se folosește `boto3` cu endpoint R2 (nu S3 AWS) — setat prin env vars.
- **Checkpointing:** la fiecare N pași se salvează starea completă (model + optimizer + scheduler +
  counter global de tokeni) și se încarcă în R2. **Resumable:** la start, se încearcă descărcarea
  ultimului checkpoint din R2.
- **Un singur entrypoint de training** partajat de Modal și RunPod, ca să nu dubleze logica.

## Consecințe
- Portabilitate: același training merge pe ambele furnizori; datele trăiesc în R2, nu pe volumul efemer.
- Credentials (Modal, RunPod, R2) se seteză prin env vars / secrets (nu se comit în repo).
- Scripturile sunt **scaffolding validabil pe cloud**; nu au fost rulate local (nu am instalat
  dependențe; R2/Modal/RunPod necesită credențiale).
