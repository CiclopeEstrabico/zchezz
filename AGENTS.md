# Zchezz — Project Guide

## Overview

Zchezz is a chess engine written in C with NNUE evaluation, targeting both native platforms (Windows/Linux/Android) and WebAssembly for browser play. The engine communicates via the UCI protocol.

**Current engine version: `engine/c/zchezz_v402/`.** NNUE HalfKP-4Bucket
(README.md § NNUE) with a TRAINED Gen-1 network plus the v4.02 search rework
(stable-TT policy, packed TT entries, VNNI eval kernel, GA-tuned pruning
constants — see README.md § Search). Measured: **+157 ±20 Elo vs v401**,
**−146 ±22 vs v314** (800 games, 100 ms/move). Phase 1 checks pass (perft
37/37, UCI extended 119/120 with known non-blocking T3.2c).

`engine/c/zchezz_v403/` is the LC0-training working copy (branch
`v403-lc0-training`, tracked since commit bb26755); its engine code is identical
to v402 - do not release from it, and do not edit v402's files expecting them
there.

For architecture, file structure, and build instructions, see `README.md` — this file is rules only.

## CRITICAL DEVELOPMENT RULES

### 1. NO REGRESSIONS — EVER
- Every code change MUST be validated with regression tests before merging.
- **Minimum regression test:** 500 games at 100ms + 200 games at 200ms, 14 concurrency, vs the previous stable version (e.g., v313 vs v312). Calculate combined ELO from all 700 games.
- If ANY test shows regression, **STOP**, diagnose, fix, and restart ALL tests from scratch.

### 2. Versioning
- Each new version gets a new folder: `engine/c/zchezz_vXXX/`
- Version number increments: v312 → v313 → v314, etc.
- NEVER modify a released version's code — create a new version folder instead.

### 3. Testing Workflow (Iterate Until Goal Is Met)

Every new version MUST pass ALL phases in order before deployment.
If ANY phase fails → fix → restart from Phase 1.

#### Phase 1: Pre-Flight Sanity Checks (before ANY games)
These checks catch regressions in seconds. Run after every recompile.

- **Perft:** `python tests/test_perft.py vXXX` — must pass 37/37
- **UCI test:** `python tests/test_uci_extended.py vXXX` — verify handshake, NNUE, search, TB, MultiPV, threads
- **50-Position NPS + Eval Benchmark:** `python tests/bench_nps.py`
  - Tests opening (10), middlegame (15), endgame 6/5/4/3/2-piece (20), insufficient material (5)
  - NPS: compare vXXX-TB vs vXXX-noTB vs previous stable. Fail if >5% drop in opening/middle
  - Eval sanity: startpos ≈ 0cp (±50), KQK > +1000cp, KvK/KBvK/KNvK = 0cp, won endgames > threshold
  - Note: endgame NPS is NOT comparable (TB prunes subtrees → fewer nodes → lower measured NPS)

#### Phase 2: Quick Validation (50 games)
- Fast check that a fix doesn't make things worse
- 50 games at 300ms, 4 concurrency, TB vs noTB or vXXX vs previous

#### Phase 3: TB Effectiveness (200 games from openings)
- 200 games: vXXX-TB vs vXXX-noTB, 300ms movetime, **4 concurrency** (TB disk I/O)
- Target: TB ≥ noTB (ELO difference includes zero or positive in CI)
- If TB is hurting → fix TB integration → restart from Phase 1

#### Phase 4: Endgame TB Test (200 games from /endgames/)
- 200 games from `/endgames/` EPD positions (exchange colors)
- vXXX-TB vs vXXX-noTB, 300ms movetime, 4 concurrency
- TB should show strongest advantage here (fewer pieces = more TB hits)

#### Phase 5: Regression Test (700 games vs previous stable)
- **Stage A:** 500 games at 100ms + 200 games at 200ms, 14 concurrency
- vXXX vs v(XXX-1), no TB for both (pure engine comparison)
- Target: no regression (ELO ≥ -10)
- **Stage B:** Multi-thread verification
  - 100 games: 1 concurrent, Threads=1 vs Threads=4, 100ms movetime
  - Verify no crashes, hangs, or data races
  - Both thread counts should play at similar strength (±20 ELO)

#### Phase 6: WASM Bundle Verification
- Compile: `build_wasm.bat` from engine version directory
- Open `zchezz_bundle.html` in browser and verify:
  - Engine doesn't freeze after opening book moves
  - Analysis MultiPV (4-5 lines) updates ALL lines at depth 14+
  - Infinite search mode progresses smoothly through depths
  - Single PV analysis works correctly

#### Phase 7: Documentation & Cleanup
- Update `README.md` with version changes
- Ensure all source files have comprehensive comments
- Clean up unused generated files

#### Phase 8: Git Commit & Push
- Final git commit with version number and key changes
- Push to GitHub — only final tested version goes to engine folder
- Verify `index.html` (GitHub Pages) is updated via `build_wasm.bat`

#### Phase 9 (EXTRA): ELO Calibration vs Stockfish
- Uses `run_tournament.py` with Stockfish as anchor (known ELO ~2800)
- 600 games at 100ms + 200 games at 200ms, 14 concurrency
- Calculates absolute ELO rating for the engine
- This is informational — does not block deployment

### 4. Use Idle Time Productively — CODE REVIEW IS FUNDAMENTAL
- While tests are running, **review the engine code line by line**.
- Understand how EVERY function works step by step.
- Look for bugs in TB integration, NNUE accumulator management, thread safety, TT handling, browser integration and so on.
- Check for off-by-one errors, missing guards, race conditions.
- This is NOT optional — it's the most valuable use of idle time and catches bugs before they cost 200-game test runs.

### 5. Git & GitHub Deployment
- Only the FINAL tested version goes to the engine folder on GitHub.
- Commit message must include: version number, key changes.
- `index.html` in repo root is the GitHub Pages deployment (auto-updated by `build_wasm.bat`).
- **Before pushing:** verify `zchezz_bundle.html` works in browser (game play + analysis + MultiPV).

### 6. WASM Bundle Compilation
- Run `build_wasm.bat` from the engine version directory.
- This calls: `emcc` (compile C → WASM) → `bundle.py` (merge HTML + JS + WASM + weights) → copy to `index.html`.
- WASM builds use `-DNO_TABLEBASES -DNO_BOOK` (no file I/O in browser).
- After building, **test in browser**: game play, opening book, analysis mode, MultiPV (4-5 lines at depth 14+).
- The browser version should be mobile first but also work in desktop.

### 7. Documentation & Reporting
- Document all changes in the `README.md` file with clarity and details.
- Document all changes in the engine with clarity and details. We want no regressions in future work.

### 8. CONFIGURATION BLOCK AT THE TOP OF EVERY TOOL — MANDATORY

**Everything reachable from the command line MUST also be settable from a
documented configuration block at the top of the file, and the CLI default MUST
BE that constant — not a copy of it.** A flag with no documented default in the
config block is a defect; so is a default that is written twice and can drift.

**Python:**
```python
# ═══════════════ CONFIGURATION ═══════════════
GAMES       = 500      # number of games to play
MOVETIME_MS = 100      # per-move budget, milliseconds
CONCURRENCY = 14       # parallel games (NOT threads per game)
OUT_PGN     = None     # path to PGN output, None = disabled
# ═════════════════════════════════════════════

p.add_argument("--games",     type=int, default=GAMES)
p.add_argument("--movetime",  type=int, default=MOVETIME_MS)
```
Never `default=500`. The literal lives in the config block, once.

**C:**
```c
/* ═══════════════ CONFIGURATION ═══════════════
 * Defaults for every CLI option.  The parser overrides these; nothing
 * may be configurable without appearing here with a comment and units. */
#define CFG_GAMES        500   /* number of games to play              */
#define CFG_MOVETIME_MS  100   /* per-move budget, milliseconds        */
#define CFG_THREADS        0   /* 0 = auto-detect cores                */
/* ════════════════════════════════════════════ */
```

Applies to: `tests/*.py`, `train/*.py`, `train/labeling/*.py`, and every
`engine/c/tools/*.c`.

### 9. Native and non-native paths are a BLEND, not a replacement

The native C tools (`engine/c/tools/selfplay.c`, `engine/c/tools/arena.c`) are a *faster path
for the same job*, never a reduced one. When a native tool replaces a Python harness, it must
keep that harness's capabilities, and the Python harness stays.

- **Output formats are not either/or.** Self-play and arena must both be able to emit the
  packed `.bin` (for training) **and** standard `.pgn` (for GUIs, external tools, and
  comparison with the Python harnesses). Losing PGN because the native path prefers `.bin` is
  a regression.
- **Openings are not optional.** Any game-playing tool must support opening books
  (`openings/lines/*.pgn`, `openings/positions/*.epd`, `openings/book.bin`) and random opening
  plies. Starting every game from the identical start position narrows the training
  distribution badly.
- **Division of labour:** a *native* tournament is for A/B between engine VERSIONS (the SPRT
  promotion gate). Calibrated absolute-ELO benchmarking against a Stockfish anchor
  (2800/2900/...) stays in `tests/run_tournament.py` — do not duplicate it in C.
- Before replacing any harness, inventory what the old one did and state, item by item,
  whether the new one covers it, deliberately does not (with the reason), or still needs it.

### 10. TRAINING-DATA NAMING CONVENTION — `result`, `cp`, `wdl`

Every training dataset (parquet columns, and the `.bin` record in
`engine/c/tools/sample.h`) uses exactly these three names, with exactly
these meanings. Do not invent a fourth.

| Name | Meaning | Range |
|---|---|---|
| `result` | the REAL game outcome, WHITE-relative | **0.0 / 0.5 / 1.0** (0 = Black won, 0.5 = draw, 1 = White won) |
| `cp` | evaluation in centipawns, WHITE-relative | int |
| `wdl` | `sigmoid(cp / 320)` — a pure function of `cp` | 0..1, WHITE-relative |

All three are WHITE-RELATIVE and `result`/`wdl` share the SAME 0..1 scale — this matters
because the training target is a CONVEX combination (below): a `result` on a different scale
(e.g. -1..1) would push the target outside the sigmoid's range at lambda=1 and break the BCE
loss (the trainer detects and converts a -1..1 column rather than training on it), and because
one STM flip (`x -> 1-x`) applied downstream must be correct for all three at once.

`result` is written as a NUMBER in new datasets; the legacy PGN strings (`'1-0'` etc.) are
still accepted when reading. The `.bin` record (`sample.h`) keeps its own
internally-consistent convention: `eval_cp` and `game_result` (`+1/0/-1`) are both
STM-relative there, and `dataset.py` converts `game_result` to the 0..1 probability at read
time.

**`wdl` IS NOT AN OUTCOME.** It is a transform of the evaluation, stored for convenience. The
temperature 320 is the same constant as `nnue.c`'s `_nnL3B * 320.0f` output scale and
`train/train_nnue.py`'s `CP_TO_WDL_T`; changing one without the others silently rescales
everything.

**The training target is a blend, computed AT TRAINING TIME, never baked into a dataset:**

```
target = lambda * result_prob + (1 - lambda) * wdl
```

`lambda` is set PER DATASET in the `DATASETS` block at the top of `train/train_nnue.py`.
lambda=0 trusts the labelling engine's evaluation; lambda=1 trusts only the real game outcome.

**Why the blend must not live in the dataset:** baking it in freezes lambda at generation
time, and lambda is exactly the knob you anneal across bootstrap generations. It also went
wrong in practice — a labeling script blended the stored `wdl` with `cp`, i.e. `f(cp)` with
`f(cp)`, so lambda silently did nothing; and another synthesised `result` from `cp`, turning
"ground truth" into the labelling engine's own opinion.

**Because `wdl` is derivable from `cp`, they can rot apart.** The trainer therefore
recomputes `wdl` from `cp` whenever `cp` is present, uses a stored `wdl` only when `cp` is
absent, and reports any dataset where the two disagree. Missing columns are not fatal:
whichever term exists is used alone, and the fallback is counted and printed.

## CRITICAL INVARIANTS — landmines, not documentation

These are the "if you touch X, Y must hold or Z breaks" facts every future change must
respect. Full explanations live in `README.md` § NNUE architecture and § Per-instance
objects — this list is just the tripwire.

- **NNUE concat order is always `[stm, opp]`.** Swapping it makes the engine evaluate winning
  positions as lost.
- **A king move that crosses its own perspective's king-bucket boundary invalidates every
  feature of that perspective.** `nnue_push_na` cannot rebuild on the spot (`board_make` calls
  it pre-move), so it only raises a per-frame dirty flag; the dirty flag must live on the
  accumulator stack frame (not a flat field) so it survives further plies and pops correctly.
  See README.md § Lazy bucket refresh.
- **TT probe happens BEFORE TB probe** (Stockfish ordering) — do not reorder.
- **Selfplay SHARES one `TTable`** between the two colors of a game and clears it between
  games with a physical `tt_clear()`, never `tt_new_generation()` — a generation bump does not
  stop a stale repetition-draw score from game N leaking into game N+1.
- **Arena ISOLATES a `TTable` per player** — they are adversaries; sharing would leak search
  information from one side to the other.
- **Lazy SMP helper threads share the main thread's `TTable` pointer** — this is intentional,
  do not give helpers their own table.
- **Only the main thread increments `TT_GEN`** — a helper doing so is a race condition.

## Coding Conventions

- C11 standard, no C++ features
- All search state is `_Thread_local` (for Lazy SMP)
- Mutable NNUE state is per-thread via `NnueAccum *` in Board struct
- WASM build uses static (no TLS, single-threaded)
- Comments in English, user-facing strings may be in Portuguese
- Piece encoding: COL_W=8, COL_B=16, type 1-6 (P,N,B,R,Q,K)
- Square encoding: a8=0, h1=63 (rank 0 = rank 8, file 0 = file a)
- Test scripts: engine path configured in run_tournament.py config block

## Technical English - writing-rules skill

All English prose in this repository (code comments, docstrings, commit
messages, help and error strings, Markdown docs) follows the
**writing-rules** skill, a domain-neutral style guide adapted from
ASD-STE100 (Simplified Technical English). Canonical copy, usable by any
LLM or agent: `.agents/skills/writing-rules/SKILL.md`. Identical copy for
Claude Code auto-discovery: `.claude/skills/writing-rules/SKILL.md`. Load
the skill BEFORE writing or reviewing any such text, and keep the two
copies identical when editing either one.

## Naming Convention — `tests/` and `train/`

`tests/` and `train/` are **flat and version-less** — one toolset tracking the current
engine, unlike `engine/c/zchezz_vXXX/` which is version-suffixed. **Never create a
`vNNN` subfolder inside `tests/` or `train/`.**

**`tests/`:**
| Prefix | Meaning |
|---|---|
| `test_*` | Pass/fail correctness check |
| `bench_*` | Performance measurement |
| `run_*` | Harness that plays games or runs a long job |
| `debug_*` | One-off scratch script (gitignored, not part of the suite) |
| bare noun (e.g. `elo_calc.py`) | Shared library, or a fixture generator (e.g. `make_random_nnu4.py`) |

**`train/`:**
| Form | Meaning |
|---|---|
| bare noun (`encoding`, `model`, `dataset`) | Library module, imported by other scripts |
| `verb_noun` (`train_nnue`, `export_nnu4`, `check_parity`) | Executable script |

When adding a new script, name it by what it does using this table — don't invent a new
prefix.

## Per-instance objects — `TTable` and `NnueNet`

`TTable` (`search.h`) and `NnueNet` (`nnue.h`) are structs with `create`/`destroy`
constructors, not bare global arrays — this is what lets `engine/c/tools/selfplay.c` and
`engine/c/tools/arena.c` run multiple independent searches/networks in one process. The UCI
binary still allocates one process-wide default of each (`g_tt`, `g_nnue_net`), so normal
single-engine usage is unaffected. See § CRITICAL INVARIANTS above for the TT-sharing rules
that follow from this, and README.md § Per-instance objects for the full rationale.

Everything else — architecture, file structure, and build instructions — lives in
`README.md`. Do not duplicate it here; if it drifts, fix it there.
