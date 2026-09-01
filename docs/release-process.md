# Release Process

A release is an evidence package, not only a successful compile.

## 1. Identify candidate and baseline

Record:

- candidate version and Git SHA;
- stable baseline version and Git SHA;
- NNUE hashes;
- relevant local runtime resources.

For this branch, the normal candidate/baseline pair is v403/v402.

## 2. Deterministic native validation

Run:

```bash
python tests/run_tests.py smoke --version v403 --baseline v402
python tests/run_tests.py full --version v403 --baseline v402 --keep-going
```

Required FAIL entries must be resolved.

## 3. Web validation

For web-affecting changes:

```bash
python tests/run_tests.py web --version v403 --baseline v402 --keep-going
```

A SKIP means the web feature was not validated. Obtain evidence on an
environment with Emscripten, the tracked shared HTML template, and Playwright before
making a web-release claim.

## 4. Sanitizers

The `release` profile runs ASan/UBSan on supported POSIX environments.

Windows may skip this step. The corresponding GitHub Actions Linux job supplies
the sanitizer evidence.

## 5. Playing strength

If the change can affect playing strength, follow
`docs/regression-testing.md`.

Do not substitute a deterministic release profile for Elo/SPRT evidence.

## 6. Release profile

```bash
python tests/run_tests.py release --version v403 --baseline v402 --keep-going
```

Review `summary.json` and all WARN/SKIP entries.

## 7. Provenance manifest

`tools/release_manifest.py` records:

- Git SHA and branch;
- dirty/clean working-tree state;
- platform/Python/compiler;
- engine path and SHA-256;
- NNUE path and SHA-256;
- Fathom file availability;
- latest discovered test summaries;
- optional regression artifact reference.

Keep this manifest with the release evidence.

## 8. Source immutability

Do not edit an already released version directory to implement the next engine
change. Create a new version directory and compare it against the stable
baseline.

Repository/test/documentation fixes that do not change released engine source
may be applied on the active development branch, but still require the
appropriate deterministic profile.

## 9. Final review

Before promotion:

- no unexplained required FAIL;
- no ignored WARN;
- every relevant SKIP has replacement evidence or a documented limitation;
- source hashes changed only where intended;
- ignored/local resources were not deleted or accidentally staged;
- documentation describes the commands that actually ran.
