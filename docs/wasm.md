# WebAssembly

The browser build uses the shared `engine/build/Makefile`.

## Shared source template

The canonical browser UI source is:

```text
engine/build/zchezz_wasm.html
```

Do not duplicate this template into every engine version. Generated JS, WASM,
and the final bundle remain version-specific.

## Browser/WASM API boundary

The browser worker calls exported C functions through Emscripten. This boundary
is a binary ABI and must be tested as such.

The v403 `SearchParams` layout in wasm32 is 44 bytes:

| Offset | Field | wasm32 size |
|---:|---|---:|
| 0 | `max_depth` | 4 |
| 4 | `start_depth` | 4 |
| 8 | `time_limit_ms` | 4 |
| 12 | `node_limit` (`long`) | 4 |
| 16 | `multi_pv` | 4 |
| 20 | `threads` | 4 |
| 24 | `stop` pointer | 4 |
| 28 | `search_state` pointer | 4 |
| 32 | `info_cb` pointer | 4 |
| 36 | `tt` pointer | 4 |
| 40 | `mpv_share_budget` | 4 |

Browser code must allocate the full 44 bytes. Pointer/function-pointer fields
not supplied by JS are explicitly zeroed. `search_best_sret` still sanitizes
these fields defensively, but that is not a substitute for allocating the
correct struct size.

Run:

```bat
python tools\repair_wasm_searchparams.py
python tools\repair_wasm_searchparams.py --check
```

`tests/test_wasm_wiring.py` prevents this ABI from silently regressing.

The browser worker uses `nnue_reset_global`, so the Emscripten export list must
contain `_nnue_reset_global`.

## Canonical web profile

```bash
python tests/run_tests.py web --version v403 --baseline v402 --keep-going
```

The sequence is:

1. compile JS/WASM;
2. bundle the shared template with version-specific JS/WASM/NNUE;
3. static bundle validation;
4. opening-book legality validation;
5. Playwright browser E2E against `zchezz_bundle.html`.

## E2E timing policy

Browser correctness tests must not use short fixed sleeps as performance
thresholds.

The application can intentionally spend up to about 10 seconds on one browser
search. Therefore the E2E suite waits for observable state transitions:

- `window.zchezzSearch` becomes available;
- analysis PV/score becomes non-empty;
- MultiPV line 2 becomes non-empty;
- game history gains the engine reply.

The default E2E condition timeout is 30 seconds. This timeout detects a hang;
it is not an NPS benchmark.

The clock-mode regression test disables the opening-book lookup inside the test
page before the move. This directly exercises the engine-search path that used
to freeze after the book was exhausted, without depending on the current book
contents.

## Windows

One-time setup:

```bat
engine\build\setup_web_windows.bat
```

Later runs:

```bat
engine\build\build_wasm.bat v403
```

## CI

GitHub Actions provisions Emscripten and Playwright and runs the same canonical
`web` profile.
