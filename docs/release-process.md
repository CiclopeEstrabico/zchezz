# Release Process

A release is evidence plus an artifact, not only a version folder.

## Gate

1. Select the candidate version explicitly.
2. Run `python tests/run_tests.py release --version vXXX`.
3. For strength-affecting changes, complete the statistical promotion gate in `docs/regression-testing.md`.
4. Verify the standalone browser bundle.
5. Generate a release manifest with `python tools/release_manifest.py --version vXXX --regression-run <id>`.
6. Review generated hashes and test reports.
7. Tag/release only the candidate that produced the recorded evidence.

The manifest links source, executable, NNUE network, and regression evidence.

