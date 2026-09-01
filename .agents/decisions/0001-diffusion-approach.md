# ADR 0001 — Abordarea de difuzie (DiffuLLaMA-style, continuă)

**Status:** Accepted
**Data:** 2026-08-31

## Context
Avem nevoie de un model de limbaj care să genereze 256 de tokeni în paralel, nu autoregresiv,
folosind atenție bidirecțională și 24 de pași de difuzie. Nu construim de la zero — adaptăm un
checkpoint autoregresiv existent.

## Decizie
- Adoptăm **difuzie continuă**: zgomot **Gaussian** pe **embeddings continue** + un **head** care
  proiectează către logits/tokeni discreți (roundare).
- Folosim **tehnica DiffuLLaMA** pentru a adapta un checkpoint autoregresiv pre-antrenat:
  1. reutilizăm greutățile;
  2. înlocuim masca cauzală cu **mască bidirecțională**;
  3. schimbăm obiectivul de la next-token la **denoising** (predict tokeni curați din secvență parțial zgomotată);
  4. adăugăm **time-step embeddings**;
  5. antrenăm să inverseze procesul de zgomot.
- Bucla de inferență: pornim din zgomot Gaussian de lungime 256 și rafinăm întreaga secvență pe 24 de pași.

## Alternative respinse
- **Difuzie discretă (mască absorbantă, ca LLaDA discret):** mai simplă, dar utilizatorul a cerut
  explicit **difuzie continuă**.
- **Autoregresie pură (arhitectura originală):** nu respectă cerința de generare paralelă.

## Consecințe
- Calitatea poate fi ușor sub modelele autoregresive la același compute — de monitorizat.
- Atenție bidirecțională completă pe 256 de tokeni = cost O(n²) per pas; optimizare cu kernel-uri SIMD.
- KV-cache clasic nu se aplică direct; trebuie tratată atenția full pe CPU.
