# Testing

`tests/run_tests.py` is the canonical deterministic test entry point.

Correctness, protocol conformance, deterministic regression, performance, web
delivery, and playing strength are different kinds of evidence. Do not use one
as a substitute for another.

## Canonical commands

Run from the repository root.

```bash
python tests/run_tests.py smoke --version v403 --baseline v402
python tests/run_tests.py full --version v403 --baseline v402 --keep-going
python tests/run_tests.py web --version v403 --baseline v402 --keep-going
python tests/run_tests.py regression --version v403 --baseline v402
python tests/run_tests.py release --version v403 --baseline v402 --keep-going
python tests/run_tests.py smoke --version v403 --baseline v402 --list
```

On Windows, the runner calls `mingw32-make`. On POSIX systems it calls `make`.

## Result states

- **PASS** — the command actually ran and returned success.
- **FAIL** — a required command ran and returned failure.
- **WARN** — an optional command ran and returned failure. Read its log.
- **SKIP** — the command did not run because a declared prerequisite was
  unavailable, such as a missing WebAssembly template or unsupported platform.
  A skip is not evidence that the feature works.

The runner writes `summary.json`, `summary.txt`, and one log per step under
`artifacts/tests/<profile>-<timestamp>/`.

## Profiles

### Smoke

Purpose: catch repository, build, board-state, NNUE, move-generation, and basic
UCI defects quickly.

Run after every engine recompile and after build/test infrastructure changes.

### Full

Purpose: run the deterministic native suite before review or merge.

It contains the smoke gates, then full perft, extended UCI testing,
documentation contracts, and infrastructure unit tests. It does not run web
checks. This avoids testing generated web artifacts before they are built.

### Web

Purpose: validate the browser-delivery path.

It builds the bundle first, then runs static bundle checks, opening-book
legality checks, and browser E2E when the required inputs/tools exist.

### Regression

Purpose: run the short candidate-vs-baseline H2H entry point.

This is an early warning only. Use `docs/regression-testing.md` for a promotion
decision.

### Release

Purpose: collect the deterministic native, web, sanitizer, and provenance
evidence used for a release review.

Platform-limited steps may report SKIP. A release claim must state those skips.
A strength-affecting release also needs the statistical evidence described in
`docs/regression-testing.md`.

## Gate catalog

Each gate below states what a PASS means.

### Gate: repository contracts

Command:

```bash
python tools/check_repo.py
```

Checks:

- required repository files exist;
- Python files parse;
- CLAUDE/AGENTS and writing-rule mirrors do not drift;
- no new machine-specific `C:\Zchezz` roots are introduced;
- piece-set directories are structurally complete when present;
- the shared Makefile uses the active v403 default and supports both Fathom and
  clean-checkout no-tablebase builds.

PASS means the repository pre-flight policy is internally consistent. It does
not compile the engine.

### Gate: agent instructions

Command:

```bash
python -m pytest tests/test_agent_instructions.py -q
```

Checks:

- `AGENTS.md` and `CLAUDE.md` exist;
- their rule bodies are identical;
- major sections are not duplicated;
- mirrored writing-rule skills are identical.

PASS means cross-agent instructions cannot silently diverge.

### Gate: repository contracts pytest

Command:

```bash
python -m pytest tests/test_repository_contracts.py -q
```

Checks executable repository policy, including:

- required v403 tracked core files;
- Fathom source/header pair consistency;
- no new absolute repository-root debt;
- Makefile compiler-selection policy;
- active engine default;
- optional piece-set completeness.

PASS means objective repository conventions hold.

### Gate: native build

Command on Windows:

```bat
mingw32-make -C engine/build ENGINE=v403 native
```

Command on POSIX:

```bash
make -C engine/build ENGINE=v403 native
```

The Makefile includes Fathom automatically only when both required local
Fathom files are available. Otherwise it compiles with `NO_TABLEBASES`.

PASS means the candidate native engine compiled. Use `build-info` to record
whether the binary was tablebase-enabled.

### Gate: NNUE artifact

Command:

```bash
python tools/check_nnue.py --version v403
```

Checks:

- `NNU4` magic;
- expected HalfKP-4Bucket dimensions;
- minimum binary length;
- stored epoch/scales readability;
- SHA-256 reporting.

PASS means the file has the expected NNU4 structural contract. It does not prove
playing strength.

### Gate: C invariants

Command on Windows:

```bat
mingw32-make -C engine/build ENGINE=v403 test-c
```

Checks directly against board/NNUE code:

- mailbox ↔ bitboard consistency;
- white/black/all occupancies;
- king-square caches;
- incremental ↔ recomputed Zobrist hash;
- full make/unmake restoration;
- castling sequence;
- en passant;
- promotion;
- 50-move and insufficient-material draw checks;
- incremental NNUE ↔ clean rebuild;
- HalfKP feature indices and king-bucket mapping.

PASS means these internal state representations agree for the exercised
sequences.

### Gate: perft smoke

Command used by the runner:

```bash
python tests/test_perft.py --version v403 --max-depth 3
```

Checks all configured perft positions but limits each to depth 3.

PASS requires exact reference node counts for every executed depth. It is a
fast move-generation/make-unmake gate.

### Gate: UCI smoke

Command:

```bash
python tests/test_uci_extended.py --version v403 --only T1 --only T2
```

Checks UCI handshake, required option inventory, readiness, position loading,
and common `go` modes.

PASS means the basic external engine protocol works.

### Gate: golden engine contracts

Command:

```bash
python -m pytest tests/test_engine_golden.py -q
```

Checks deterministic externally visible contracts stored in
`tests/data/golden_engine.json`, such as required UCI option names.

PASS means the protected deterministic inventory did not drift.

### Gate: perft full

Command:

```bash
python tests/test_perft.py --version v403
```

Runs every configured reference depth. This gate runs once in `full`; it is not
duplicated by the smoke command because smoke stops at depth 3.

PASS requires exact node counts.

### Gate: UCI extended

Command:

```bash
python tests/test_uci_extended.py --version v403
```

Groups:

- **T1** — handshake and required UCI options;
- **T2** — position and search commands;
- **T3** — Syzygy functional probing when a tablebase directory is available;
- **T4** — opening-book option behavior, with file-specific checks only when a
  book exists;
- **T5** — MultiPV;
- **T6** — Threads/Lazy-SMP command behavior;
- **T7** — supported UCI option setting;
- **T8** — engine extension commands (`bench`, `d`, `eval`);
- **T9** — crash/stress sequences and rapid state changes.

The suite intentionally avoids fixed NPS thresholds and diagnostic-message
wording. Those are machine/implementation dependent and belong in performance
or logging tests, not protocol correctness.

PASS means all applicable functional assertions passed. Environmental groups
can report SKIP.

### Gate: documentation contracts

Command:

```bash
python -m pytest tests/test_documentation.py -q
```

Checks that canonical commands and runner gates are documented, documented
repository paths exist when they are asserted as literal tracked paths, and
engine contract identifiers are unique.

PASS means documentation and executable test topology agree on objective facts.

### Gate: infrastructure unit tests

Command:

```bash
python -m pytest tests/test_repo_paths.py tests/test_quality_tools.py tests/test_test_runner.py tests/test_repo_policy.py tests/test_wasm_wiring.py -q
```

Checks numeric version resolution, quality-tool parsers, runner wiring, and
repository-policy helpers without launching the chess engine.

PASS means the testing infrastructure itself satisfies its unit contracts.

### Gate: bundle

Command:

```bash
python tests/run_tests.py web --version v403 --list
```

The underlying Make target is `bundle`; it consumes the shared template and writes
the generated bundle into the selected engine version directory.

Prerequisites:

- Emscripten (`emcc`);
- the tracked shared template `engine/build/zchezz_wasm.html`.

If `emcc` is unavailable locally, the runner reports SKIP instead of claiming success.
The shared template is tracked repository infrastructure and is separately enforced.

PASS means JS/WASM and NNUE data were bundled successfully.

### Gate: browser static

Runs only after a bundle exists.

Checks the generated HTML for:

- embedded WASM/NNUE payload markers;
- engine bootstrap code;
- opening-book block when the UI exposes one;
- absence of obvious unresolved bundle placeholders.

PASS is a static packaging check, not browser execution.

### Gate: book contracts

Runs after the web artifact exists.

Checks every parsed opening-book FEN and every configured move for chess
legality. Engine-quality scoring is optional and is not part of the
deterministic release gate.

PASS means the embedded opening-book entries are structurally legal.

### Gate: browser e2e

Uses Playwright in headless mode.

PASS means the delivered page starts and satisfies the browser scenarios
implemented in `tests/test_browser.py`.

If Playwright or the bundle is unavailable, the runner reports SKIP.

### Gate: ASan+UBSan build

Runs on supported POSIX toolchains in the release profile and CI.

PASS means the candidate compiled with AddressSanitizer and
UndefinedBehaviorSanitizer instrumentation.

### Gate: ASan+UBSan UCI smoke

Executes T1/T2 against the sanitized executable.

PASS means the basic UCI/search sequence completed under sanitizer
instrumentation without a sanitizer-detected failure.

### Gate: release manifest

Command:

```bash
python tools/release_manifest.py --version v403
```

Records Git state, compiler/platform information, engine and NNUE hashes,
Fathom availability, and the most recent available test summaries.

PASS means provenance metadata was written. It does not replace any preceding
test.

### Gate: quick H2H

The `regression` profile launches candidate and baseline with paired test
configuration through `tests/run_tournament_quick.py`.

PASS/FAIL of the harness means the tournament completed successfully. The
strength interpretation must use W/D/L, Elo uncertainty, and the procedure in
`docs/regression-testing.md`.

## Failure handling

For a required FAIL:

1. Open the corresponding log.
2. Identify whether the failure is engine logic, test logic, environment, or
   repository infrastructure.
3. Do not convert a real failure into a skip.
4. Fix the smallest correct layer.
5. Re-run the focused test.
6. Re-run the profile that exposed the defect.

For a WARN, decide whether the optional check should be promoted to required,
fixed, or removed. Do not leave recurring warnings unexplained.

For a SKIP, record the missing prerequisite. If the changed feature depends on
that skipped gate, obtain evidence on another supported environment before
release.
