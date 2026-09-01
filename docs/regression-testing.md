# Statistical Regression Testing

Playing-strength evidence is statistical. A deterministic green suite proves
correctness contracts, not equal Elo.

## When required

Statistical testing is required for changes that can alter playing strength:

- search or evaluation;
- NNUE architecture, weights, or feature handling;
- tablebase logic;
- threading/search parallelism;
- time management;
- move ordering or pruning parameters.

Documentation, repository organization, CI-only, and test-harness-only changes
do not require Elo evidence unless they also alter an engine/runtime input.

## Pairing

Use paired openings and reverse colors. Keep engine settings identical except
for the tested change.

Record the opening source and seed.

## Required metadata

Every promotion run records:

- candidate Git SHA;
- baseline Git SHA;
- candidate and baseline labels;
- candidate and baseline NNUE SHA-256;
- opening source and seed;
- time control;
- Threads;
- Hash;
- concurrency;
- W/D/L;
- crashes;
- time losses;
- Elo estimate and confidence interval, or SPRT state;
- PGN/result artifact path.

## Quick H2H

The runner profile:

```bash
python tests/run_tests.py regression --version v403 --baseline v402
```

is an early smoke test. It catches large regressions and harness problems. It
is not sufficient for promotion.

## Promotion decision

Prefer SPRT when the native arena supports the required experiment.

Before the run, define:

- H0: unacceptable regression boundary;
- H1: acceptable change boundary;
- alpha;
- beta.

Do not move the boundaries after observing results.

If fixed-game testing is used instead, report the Elo confidence interval and
use a predeclared acceptance rule. Never promote from a point estimate alone.

## Tablebase experiments

When comparing tablebase behavior:

- use the same engine binary/configuration except for tablebase enablement;
- use paired positions;
- record whether Fathom was compiled into the binary;
- record the tablebase path/set;
- do not compare raw endgame NPS as if it measured the same search tree.
  Tablebase cutoffs change the number of searched nodes.

## Threading experiments

Separate game concurrency from engine `Threads`.

A threading validation must record both. A run with 14 concurrent games and
`Threads=1` is not a test of a four-thread engine.

## Interpretation

- Confidence interval crossing zero: no clear strength conclusion.
- Large positive/negative point estimate with wide CI: collect more evidence.
- Crashes/time losses: correctness failure regardless of Elo.
- SPRT accepted H1/H0: use the predeclared decision.
