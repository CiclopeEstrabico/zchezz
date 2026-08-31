# Testing

Zchezz separates different kinds of evidence. Correctness tests, deterministic regression, performance measurements, and Elo/SPRT do not answer the same question.

## Canonical commands

Run these commands from the repository root:

```bash
python tests/run_tests.py smoke
python tests/run_tests.py full
python tests/run_tests.py web
python tests/run_tests.py regression
python tests/run_tests.py release
python tests/run_tests.py smoke --list
```

Use `--version v403` to select an engine explicitly. If omitted, `utils/repo_paths.py` selects the numerically highest `engine/c/zchezz_v*` directory.

## Profiles

### Smoke

Use after engine-source changes. It runs repository contracts, native invariants, reduced-depth perft, a UCI smoke set, and deterministic golden contracts.

### Full

Use before review of an engine change. It adds full perft, the extended UCI suite, documentation contracts, and other deterministic integration tests.

### Web

Builds the WASM bundle and runs static/browser checks.

### Regression

Runs the quick head-to-head entry point. The promotion decision should use the documented statistical procedure in `docs/regression-testing.md`.

### Release

Runs deterministic gates, web checks, sanitizer build, and release-manifest generation. Statistical promotion evidence must already exist for strength-affecting changes.

## Test evidence

- Correctness: legal moves and state transitions are valid.
- Protocol: the external UCI contract is satisfied.
- Native invariants: internal representations agree.
- Golden regression: deterministic behavior changed or did not change.
- Performance: speed/resource use changed.
- Statistical regression: strength probably changed or did not change.
- Browser/WASM: delivered browser behavior works.
- Repository contracts: project rules and documentation did not drift.

## Bug workflow

For a logic defect:

1. Reproduce the defect.
2. Add a regression test when practical.
3. Confirm that the test fails before the fix.
4. Implement the smallest correct fix.
5. Run the focused test.
6. Run the appropriate profile.
7. Run strength testing only when the change can affect playing strength.

## Candidate and baseline selection

`--version` selects the candidate. `--baseline` selects the comparison engine.
When omitted, the runner uses the numerically newest engine older than the candidate.
For this branch, `v403` therefore defaults to `v402`.

```bash
python tests/run_tests.py regression --version v403 --baseline v402
```

