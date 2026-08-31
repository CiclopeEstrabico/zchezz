# Generated Artifact Policy

Generated files must have one declared role.

- `artifacts/tests/` — ignored test logs and summaries.
- `artifacts/regression/` — ignored/local regression manifests, PGNs, and reports unless deliberately attached to a release.
- `artifacts/releases/` — generated release provenance manifests.
- `zchezz_wasm.js`, `zchezz_wasm.wasm`, `zchezz_bundle.html` — build artifacts in the selected engine directory under the current build model.
- root `index.html` — committed deployment artifact under the current GitHub Pages model. A future CI deployment can replace this policy after equivalent coverage exists.
- NNUE weights inside a release folder — versioned model artifact, identified by SHA-256.

Do not write benchmark screenshots, temporary executables, PGNs, or logs into source directories unless an existing compatibility workflow requires it.

