# WebAssembly

The browser build uses the shared `engine/build/Makefile`.

## Shared source template

The canonical browser UI source is:

```text
engine/build/zchezz_wasm.html
```

Do not duplicate this template into every engine version. v314 and v401
historically contain the same template blob; migration is performed by:

```bash
python tools/promote_wasm_template.py
```

The tool verifies the historical Git blob before copying it. The resulting
`engine/build/zchezz_wasm.html` must be committed so clean CI checkouts have it.

Generated outputs remain version-specific:

```text
engine/c/zchezz_v403/zchezz_wasm.js
engine/c/zchezz_v403/zchezz_wasm.wasm
engine/c/zchezz_v403/zchezz_bundle.html
```

## WASM API contract

WebAssembly defines `NO_TABLEBASES` and `NO_BOOK` for native file I/O.

The browser worker calls `nnue_reset_global`, so the Makefile exports
`_nnue_reset_global`. `tests/test_wasm_wiring.py` protects this boundary.

## Canonical web profile

```bash
python tests/run_tests.py web --version v403 --baseline v402 --keep-going
```

Order:

1. compile JS/WASM with Emscripten;
2. bundle the shared template with version-specific JS/WASM/NNUE;
3. static bundle checks;
4. embedded opening-book legality checks;
5. Playwright browser E2E against `zchezz_bundle.html`.

## Windows

One-time setup:

```bat
engine\build\setup_web_windows.bat
```

Later runs:

```bat
engine\build\build_wasm.bat v403
```

The setup script uses the official emsdk, installs Python web dependencies,
installs Playwright Chromium, verifies the template, and runs the web profile.

## CI

GitHub Actions provisions emsdk and Playwright and runs the same `web` profile.
A missing shared template is a repository defect, not a successful skip.
