# WASM Blob-worker loading

The offline Zchezz bundle runs Emscripten-generated JavaScript inside a Web
Worker created from a Blob URL.

Newer Emscripten runtimes can still resolve the `.wasm` filename through
`Module.locateFile()`. Inside a Blob worker, the default script prefix is a
`blob:` URL; treating that prefix like a normal directory can produce:

```text
Failed to execute 'open' on 'XMLHttpRequest': Invalid URL
```

The worker already receives the complete WASM ArrayBuffer. Its initialization
therefore creates a temporary `application/wasm` Blob URL from those bytes and
returns that URL from `locateFile()` for `.wasm` lookups.

Repair and verify:

```bat
python tools\repair_wasm_blob_worker.py
python tools\repair_wasm_blob_worker.py --check
python -m pytest tests\test_wasm_blob_worker.py -q
```

Then rebuild the bundle:

```bat
python tests\run_tests.py web --version v403 --baseline v402 --keep-going
```

An old `zchezz_bundle.html` still contains the old worker and must not be used
to validate this repair.
