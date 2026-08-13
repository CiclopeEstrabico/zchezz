# Zchezz — Project Guide

## Overview

Zchezz is a chess engine written in C with NNUE evaluation, targeting both native platforms (Windows/Linux/Android) and WebAssembly for browser play. The engine communicates via the UCI protocol.

**Current engine version: `engine/c/zchezz_v400/`.** Its NNUE architecture (HalfKP-4Bucket, see README.md § NNUE) is implemented and structurally validated. The network is **being trained** in two stages (`STAGE` at the top of `train/train_nnue.py`): a `warmup` from random weights on crude high-volume data, then a `finetune` on the strong deeply-evaluated sets. Until a finetuned NNU4 file is exported and passes Phase 1, the deployed/playable engine stays v3.14, and Phase 2–9 strength numbers for v4.00 mean nothing — an undertrained net loses by construction.

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

**Comments and docstrings explain the PRESENT, not the past.** A reader opens a
file to learn what it does, what to set, and how to run it — not to learn what a
previous version got wrong. Write:

- **what the tool does**, in one paragraph, first;
- **how to run it**, with a real command line;
- **what to set**, naming the constants in the order that matters;
- **the traps that are still live**: an invariant that will break something if
  violated ("keep this 320, it must match `nnue.c`"), stated as a rule.

Do NOT write: "this used to be X", "an earlier draft had a bug", "v3.14 did it
differently", "kept for the record", "this was measured on 47M rows", or a
paragraph arguing with a decision that is already made. If a past mistake taught
a rule, keep the RULE and drop the story. Comments in English (CLAUDE.md
§ Coding Conventions); user-facing strings may be in Portuguese.

### 8. CONFIGURATION BLOCK FIRST, CLI SECOND — MANDATORY

**The configuration block at the top of the file IS the interface.** Running the
tool with NO arguments at all must work and must do exactly what those constants
say. The command line only OVERRIDES them, for scripted or one-off runs. Every
value reachable from the command line has a documented constant with units; a
flag with no constant is a defect, and so is a default written twice, since the
two copies will drift.

Rules, all of them enforced by review:

1. **The CLI default IS the constant**, never a copy of the literal.
2. **No knob hidden in the body of the code.** A timeout, a port, an output
   path, a seed, a sampling fraction — if the code reads it, the top of the file
   declares it.
3. **`--show-config`**: every tool prints its resolved settings and exits,
   running nothing and writing nothing. This is how you check what a bare run
   will do before starting a job that takes hours.
4. **One name per concept, everywhere.** The shared vocabulary lives in ONE
   place — `utils/cliconf.py`, section "SHARED CONFIGURATION VOCABULARY" — and
   is not copied into individual files. A `DEFAULT_` prefix marks a knob
   specific to one tool.
5. **Old invocations keep working.** Where a script had a positional form
   (`test_perft.py v400`), it is translated to the equivalent flag.

**Python** — declare each option once, next to its constant, and let
`utils/cliconf.py` build the parser from that list:

```python
# ═══════════════ CONFIGURATION ═══════════════
GAMES       = 500      # number of games to play
MOVETIME_MS = 100      # per-move budget, milliseconds
CONCURRENCY = 14       # parallel games (NOT threads per game)
SAVE_PGN    = False    # write a PGN of every game
# ═════════════════════════════════════════════

sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))
from cliconf import override_from_cli

CLI = [
    ("GAMES",       "--games",       int,  "number of games to play"),
    ("MOVETIME_MS", "--movetime",    int,  "per-move budget, ms"),
    ("CONCURRENCY", "--concurrency", int,  "parallel games"),
    ("SAVE_PGN",    "--pgn",         bool, "write a PGN of every game"),
]

if __name__ == "__main__":
    override_from_cli(globals(), CLI, description=__doc__)
    main()
```

This gives `--games N`, `--pgn`/`--no-pgn` for every bool, repeatable flags for
every list, `--show-config`, and a `--help` that prints the value the config
block actually holds. Call it right after the config block and BEFORE anything
derived from it (timestamps, output paths, `os.makedirs`, validation).

A hand-written `argparse` is fine when a tool needs richer syntax — it must
still read its defaults from the constants and still provide `--show-config`.

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

Applies to: `tests/*.py`, `train/*.py`, `train/labeling/*.py`, `utils/*.py`, and
every `engine/c/tools/*.c`.

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

## Naming Convention — `tests/`, `train/` and `utils/`

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

**`utils/`:** helpers shared by BOTH `tests/` and `train/`, plus standalone
maintenance scripts. `cliconf.py` (config-block + CLI plumbing, and the one copy
of the shared configuration vocabulary — rule 8) and `kill_ghosts.py` live here.
Anything used by only one of the two folders belongs in that folder instead.

When adding a new script, name it by what it does using this table — don't invent a new
prefix.

### One tool per job — don't fork a script to change a setting

A second script that differs from an existing one only by its constants is a
maintenance trap: the machinery drifts between the copies. Add the case to the
existing tool's config block instead. In particular:

- **positions in, positions out** → `train/labeling/process_positions.py`. It
  reads `.epd`, `.pgn`, `.bin` and `.parquet`, writes any combination of
  `parquet`/`bin`/`epd`/`pgn`, and every filter is optional, so it is also the
  format converter (`--filters none`). Do not write a new one-off converter.
- **games between two engine versions** → `run_arena.py` (SPRT gate) or
  `run_tournament.py` (cross-table / anchored Elo).
- **training data from self-play** → `selfplay.exe` via `run_selfplay_native.py`
  (fast path) or `run_selfplay.py` (any UCI engine).

## Per-instance objects — `TTable` and `NnueNet`

`TTable` (`search.h`) and `NnueNet` (`nnue.h`) are structs with `create`/`destroy`
constructors, not bare global arrays — this is what lets `engine/c/tools/selfplay.c` and
`engine/c/tools/arena.c` run multiple independent searches/networks in one process. The UCI
binary still allocates one process-wide default of each (`g_tt`, `g_nnue_net`), so normal
single-engine usage is unaffected. See § CRITICAL INVARIANTS above for the TT-sharing rules
that follow from this, and README.md § Per-instance objects for the full rationale.

Everything else — architecture, file structure, and build instructions — lives in
`README.md`. Do not duplicate it here; if it drifts, fix it there.
