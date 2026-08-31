# Zchezz Repository Professionalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic engineering infrastructure to the active v403 checkout without changing chess-engine behavior.

**Architecture:** A path module owns repository discovery. The unified runner composes existing and new deterministic checks, then writes reports below an ignored artifact root. Native checks validate engine state through a separate harness. Documentation and CI invoke the same local commands.

**Tech Stack:** Python 3, pytest, C11, GNU Make, GitHub Actions, Playwright, Emscripten.

**Spec:** `docs/professionalization_plan.md`

## Global Constraints

- Work in the current `v403-lc0-training` checkout.
- Preserve tracked, ignored, and untracked local resources unless the user authorized a clear destination.
- Do not change `engine/c/zchezz_v403/*.c` or `*.h` without a reproduced engine defect and regression test.
- Write all repository prose in English under the writing-rules contract.
- Use numeric engine-version ordering. Do not duplicate repository root discovery.
- Store generated test output below ignored `artifacts/`.

---

### Task 1: Preserve the baseline and remove the obsolete remote branch

**Files:**
- Create: `artifacts/baseline/v403-lc0-training.json`
- Modify: none
- Test: manual Git-state comparison

- [ ] Record the branch, commit, status, ignored and untracked resource names, engine directories, and relevant resource hashes.
- [ ] Confirm that the saved manifest contains no machine-specific absolute paths.
- [ ] Delete only `origin/403_testsuits`, which the user explicitly authorized.
- [ ] Compare the pre-change resource inventory with the final inventory.

### Task 2: Add the repository path and test foundation

**Files:**
- Create: `utils/repo_paths.py`, `tests/run_tests.py`, `tests/helpers/uci_engine.py`
- Create: `tests/test_repo_paths.py`, `tests/test_test_runner.py`
- Test: `python -m pytest tests/test_repo_paths.py tests/test_test_runner.py -q`

- [ ] Write a failing test that expects numeric engine ordering and root discovery.
- [ ] Run the test. Confirm that it fails because the module does not exist.
- [ ] Implement the smallest path API that satisfies the test.
- [ ] Write a failing test that expects the runner to list profiles and produce a structured result.
- [ ] Implement the runner and UCI helper.
- [ ] Run both tests and confirm that they pass.

### Task 3: Add deterministic contracts and validation tools

**Files:**
- Create: `tests/test_repository_contracts.py`, `tests/test_documentation.py`, `tests/test_agent_instructions.py`
- Create: `tools/check_repo.py`, `tools/check_nnue.py`, `tools/check_epd.py`, `tools/release_manifest.py`
- Create: `tests/data/golden_engine.json`, `tests/test_engine_golden.py`
- Test: `python -m pytest tests/test_agent_instructions.py tests/test_repository_contracts.py tests/test_documentation.py tests/test_engine_golden.py -q`

- [ ] Write failing tests for synchronized agent rules, real documented paths, and explicit golden data.
- [ ] Run the tests and confirm that each failure describes the missing contract.
- [ ] Implement the checks without hard-coded local paths or self-matching forbidden-string checks.
- [ ] Run the contract test command and confirm that it passes.

### Task 4: Add build and native invariant gates

**Files:**
- Create: `engine/c/tests/test_engine_invariants.c`
- Modify: `engine/build/Makefile`, `.gitignore`
- Test: `make -C engine/build ENGINE=v403 test-c`

- [ ] Build the existing native target before changing the Makefile and record its command and result.
- [ ] Add a native harness for make/unmake, Zobrist recomputation, board occupancy, and NNUE rebuild checks.
- [ ] Add debug, sanitizer, and invariant Make targets without changing release optimization defaults.
- [ ] Run the invariant target and sanitizer UCI smoke.
- [ ] Compare the v403 C and header hashes with the baseline.

### Task 5: Document the contracts and automate deterministic CI

**Files:**
- Create: `.github/workflows/deterministic.yml`, `.github/workflows/regression.yml`
- Create: `docs/architecture.md`, `docs/build.md`, `docs/testing.md`, `docs/regression-testing.md`, `docs/engine-contracts.md`, `docs/nnue.md`, `docs/syzygy.md`, `docs/wasm.md`, `docs/repository-layout.md`, `docs/release-process.md`, `docs/generated-artifacts.md`
- Create: `THIRD_PARTY_NOTICES.md`, `pyproject.toml`
- Modify: `AGENTS.md`, `CLAUDE.md`, `README.md` or `Readme.md`
- Test: `python -m pytest tests/test_documentation.py tests/test_agent_instructions.py -q`

- [ ] Write failing documentation tests for documented commands and linked local paths.
- [ ] Implement focused documentation and synchronized operational rules.
- [ ] Configure CI to call `tests/run_tests.py`, not duplicate test logic.
- [ ] Run documentation and instruction tests and confirm that they pass.

### Task 6: Integrate, validate, and publish

**Files:**
- Modify: only files required by Tasks 1 to 5
- Test: `python tests/run_tests.py smoke --version v403 --baseline v402 --keep-going`

- [ ] Run smoke, full, web, and release profiles. Capture reports below `artifacts/tests/`.
- [ ] Run direct build, native invariant, and sanitizer commands when the runner cannot invoke them.
- [ ] Classify every failure as introduced, pre-existing, or environment-limited. Fix only introduced infrastructure defects.
- [ ] Review Git status, diff, untracked and ignored inventories, and v403 core hashes.
- [ ] Commit validated changes on `v403-lc0-training` and push the active branch.
