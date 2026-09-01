# Build

The shared build entry point is `engine/build/Makefile`.

## Native build

Windows:

```bat
mingw32-make -C engine/build ENGINE=v403 build-info
mingw32-make -C engine/build ENGINE=v403 native
```

Linux/macOS with GNU Make:

```bash
make -C engine/build ENGINE=v403 build-info
make -C engine/build ENGINE=v403 native
```

`build-info` prints the selected engine, compiler, and Fathom state.

## Compiler selection

GNU Make supplies a built-in `CC=cc`. The Makefile replaces only that built-in
default with GCC. A compiler selected by the caller remains authoritative.

Examples:

```bash
make -C engine/build ENGINE=v403 CC=gcc native
make -C engine/build ENGINE=v403 CC=clang ARCH_FLAGS=-mavx2 STATIC_FLAG= native
```

CI uses this to compile the same candidate with GCC and Clang.

## Architecture flags

The local production default is:

```text
-mavxvnni -mavx2
```

Hosted CI can override `ARCH_FLAGS`, for example to `-mavx2`, when the runner
CPU contract does not guarantee VNNI.

Changing architecture flags changes the binary/toolchain contract. Do not
interpret an NPS difference caused by a different ISA as an engine-algorithm
regression.

## Syzygy / Fathom

Native tablebase support requires both local Fathom source and header files in
the selected engine directory.

The Makefile behaves as follows:

- both files present → include Fathom and enable tablebases;
- either file absent → compile with `NO_TABLEBASES`.

Use the strict gate when a tablebase-capable build is required:

```bat
mingw32-make -C engine/build ENGINE=v403 require-tablebases
```

This must fail if the Fathom pair is unavailable.

See `docs/syzygy.md`.

## Debug build

```bash
make -C engine/build ENGINE=v403 debug
```

Uses low optimization plus stronger warnings.

## Sanitizer build

```bash
make -C engine/build ENGINE=v403 sanitize
```

The release profile treats sanitizer execution as a supported-POSIX gate. A
Windows environment may report this step as SKIP; GitHub Actions supplies the
Linux sanitizer evidence.

## Native invariant harness

```bat
mingw32-make -C engine/build ENGINE=v403 test-c
```

This compiles and executes `engine/c/tests/test_engine_invariants.c`.

## WebAssembly

The `wasm` target requires Emscripten. The `bundle` target additionally uses
the tracked shared template `engine/build/zchezz_wasm.html`.

```bash
make -C engine/build ENGINE=v403 bundle
```

WebAssembly always defines `NO_TABLEBASES` and `NO_BOOK` for native file I/O.
The WASM export list includes `_nnue_reset_global`, matching the browser worker.
See `docs/wasm.md` and `docs/testing.md`.

## Safe cleanup

```bat
mingw32-make -C engine/build ENGINE=v403 clean
```

Cleanup delegates to `engine/build/clean_generated.py`. It removes only known
generated build outputs. It does not traverse or delete ignored datasets,
checkpoints, tablebases, openings, local engines, or other user resources.
