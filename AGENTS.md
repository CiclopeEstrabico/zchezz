# Zchezz — Agent Project Guide

## Project status

Zchezz is a C11 UCI chess engine with NNUE evaluation. It targets native builds
and WebAssembly.

- Active development candidate: `engine/c/zchezz_v403/`.
- Stable comparison baseline for this branch: `engine/c/zchezz_v402/`.
- Do not assume that v403 and v402 are identical. Verify source and artifact
  hashes when that fact matters.
- Released engine source directories are immutable. Create a new version
  directory for a new engine release.

Detailed architecture belongs in `README.md` and `docs/`. This file contains
cross-agent development rules.

## Canonical validation workflow

`tests/run_tests.py` is the canonical test entry point. The authoritative test
catalog, purpose of every gate, pass criteria, prerequisites, and failure
meaning are in `docs/testing.md`.

Use these profiles:

```bash
python tests/run_tests.py smoke --version v403 --baseline v402
python tests/run_tests.py full --version v403 --baseline v402 --keep-going
python tests/run_tests.py web --version v403 --baseline v402 --keep-going
python tests/run_tests.py regression --version v403 --baseline v402
python tests/run_tests.py release --version v403 --baseline v402 --keep-going
```

Rules:

1. Run `smoke` after native engine changes and after build/test infrastructure
   changes.
2. Run `full` before review or merge of engine changes.
3. Run `web` when WASM, browser, bundle, opening-book HTML, or web assets change.
4. Run `release` before calling a candidate release-ready.
5. A `SKIP` is not evidence that a feature works. It means a prerequisite was
   unavailable. Read `docs/testing.md` before interpreting a skipped gate.
6. Strength-affecting changes also require the statistical procedure in
   `docs/regression-testing.md`. A deterministic green suite does not prove
   equal playing strength.
7. Do not change a golden baseline only to make a failing test green. Explain
   and review the behavior change first.

## Bug workflow

For a logic defect:

1. Reproduce the defect.
2. Add a focused regression test when practical.
3. Confirm that the test fails before the fix when the environment permits.
4. Implement the smallest correct fix.
5. Run the focused test.
6. Run `smoke`.
7. Run `full` if engine behavior can change.
8. Run statistical strength testing only when the change can affect strength.

Do not hide real defects by weakening tests. Remove brittle assertions only
when they test an implementation accident, machine performance, optional local
data, or text formatting rather than the intended contract.

## Statistical regression rule

Playing strength is statistical. Follow `docs/regression-testing.md`.

- Use paired openings and reverse colors.
- Record candidate and baseline Git SHA, NNUE hashes, opening source and seed,
  time control, Threads, Hash, concurrency, W/D/L, crashes, time losses, and
  the Elo confidence interval or SPRT state.
- Prefer SPRT for promotion decisions.
- A short fixed-game H2H is an early warning, not the promotion gate.
- Documentation-only and repository-only changes do not require Elo testing.
- Search, evaluation, NNUE, tablebase, threading, and time-management changes do.

## Build rules

The shared build entry point is `engine/build/Makefile`. See `docs/build.md`.

- The default active engine is v403 on this branch.
- The caller-selected compiler must be respected. CI must genuinely compile
  with both GCC and Clang.
- Local Fathom files are optional repository resources. If both Fathom source
  and header are present, the native build enables Syzygy. If they are absent,
  a clean checkout builds with `NO_TABLEBASES`.
- Use `require-tablebases` when the test or release claim specifically requires
  a tablebase-capable native binary.
- Never delete ignored datasets, checkpoints, openings, tablebases, local
  engines, or other user resources as part of build cleanup.

## Critical engine invariants

These are correctness landmines. Keep them true.

- NNUE concatenation order is always `[stm, opp]`.
- A king move that crosses that perspective's king-bucket boundary invalidates
  that perspective's accumulator. The dirty state belongs to the accumulator
  stack frame and must survive push/pop correctly.
- TT probe occurs before TB probe.
- Self-play shares one `TTable` between colors of one game and physically
  clears it between games.
- Arena isolates a `TTable` per player.
- Lazy-SMP helpers share the main thread's `TTable`.
- Only the main search thread increments TT generation.
- Board square encoding is `a8=0` through `h1=63`.
- Piece encoding uses `COL_W=8`, `COL_B=16`, piece types 1..6.
- Native engine code is C11.

Executable representation invariants are tested by
`engine/c/tests/test_engine_invariants.c`.

## Training-data convention

Use only these semantic names:

| Name | Meaning | Frame / range |
|---|---|---|
| `result` | real game outcome | White-relative, 0.0 / 0.5 / 1.0 |
| `cp` | centipawn evaluation | White-relative integer |
| `wdl` | `sigmoid(cp / 320)` | White-relative, 0..1 |

The training target is computed at training time:

```text
target = lambda * result + (1 - lambda) * wdl
```

Do not bake this blend into generated datasets. `wdl` is derived from `cp`;
when both exist, training code must treat `cp` as the primitive and detect
inconsistency.

The packed native `.bin` sample format may use STM-relative internal fields;
the reader is responsible for converting them to the canonical training frame.

## Tool configuration

Every configurable test, training, labeling, or native utility must have a
documented configuration block near the top of the file. CLI defaults must
refer to those constants rather than duplicate literals.

Keep path defaults repository-relative or derive them through
`utils/repo_paths.py`. Existing machine-local defaults are tracked as explicit
migration debt. Do not add new `C:\Zchezz` roots.

## Native and Python tools

Native tools are faster execution paths, not reduced replacements.

- Preserve required output formats.
- Preserve opening-book support and random opening support where applicable.
- Native arena is for engine-version A/B and promotion testing.
- Stockfish-anchored absolute Elo remains a Python tournament responsibility.
- Before replacing a harness, inventory capabilities and document any
  deliberate difference.

## Documentation

- `CLAUDE.md` and `AGENTS.md` must have identical bodies; only the title line
  may differ.
- `docs/testing.md` is the canonical deterministic-test catalog.
- `docs/regression-testing.md` is the canonical statistical-strength policy.
- `docs/build.md` is the canonical build reference.
- `docs/release-process.md` defines release evidence.
- `docs/syzygy.md` defines tablebase availability and validation.
- All English technical prose follows the mirrored writing-rules skills under
  `.agents/skills/writing-rules/` and `.claude/skills/writing-rules/`.

When documentation and executable behavior disagree, fix one of them in the
same change. Do not leave stale instructions as historical folklore.
