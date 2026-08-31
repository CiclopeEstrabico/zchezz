# Statistical Regression Testing

Playing-strength evidence is statistical. A point Elo estimate without uncertainty is not a sufficient promotion rule.

## Pairing

Use paired openings and reverse colors. Keep engine settings identical except for the tested change. Record the opening source and random seed.

## Required run metadata

Every promotion run records:

- candidate Git SHA;
- baseline Git SHA;
- candidate/baseline labels;
- NNUE SHA-256 values;
- opening source and seed;
- time control;
- Threads and Hash settings;
- concurrency;
- W/D/L;
- crashes and time losses;
- Elo estimate and confidence interval, or SPRT state;
- PGN path.

## Decision method

Prefer SPRT for the promotion gate. Define the unacceptable-regression and acceptable-change hypotheses before the run. Do not move the boundaries after seeing results.

A short fixed-game H2H remains useful as an early smoke test. It does not replace the promotion gate.

## Tiers

Documentation and repository-organization changes do not require Elo testing. Search/evaluation, NNUE, tablebase, threading, and time-management changes do.

