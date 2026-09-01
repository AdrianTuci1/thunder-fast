# .agents — memoria proiectului thunder-fast

Această carte este memoria persistentă a proiectului. Aici se scriu **deciziile**, **contextul**
și **starea** pentru ca orice sesiune de lucru să poată continua fără a reconstrui totul de la zero.

## Reguli de folosire

- **memory.md** — ține minte totul despre proiect: viziune, obiective, constrângeri, decizii curente,
  întrebări deschise. Este fișierul "de pornire" al oricărei sesiuni.
- **decisions/** — jurnal de tip ADR (Architecture Decision Record). Fiecare decizie importantă
  primește un fișier `NNNN-titlu.md` cu: context, decizie, alternativă, consecințe.
- **roadmap.md** — foaia de parcurs (faze, obiective verificabile, ordinea de lucru).
- **research/** — note despre tehnicile folosite (DiffuLLaMA, difuzie, cuantizare, kernel-uri SIMD),
  cu link-uri către surse reale pentru ca planul să fie fundamentat, nu inventat.

## Stare actuală (snapshot)

Proiectul este în **faza de planificare**. Nu există încă cod, nu am instalat dependențe.
Directorul `thunder-fast/` este gol. Toate deciziile de mai jos sunt **schiță de lucru** — marcate
explicit acolo unde trebuie confirmate cu utilizatorul.

## Cum lucrez cu memoria

1. La început de sesiune: citesc `memory.md` + `roadmap.md` + ultimele decizii.
2. Schimbări de stare importante => se updaterază `memory.md`.
3. Decizie luată => se adaugă în `decisions/`.
4. Informație tehnică nouă => se notează în `research/`.
