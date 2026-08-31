# Build System

`engine/build/Makefile` is the shared build entry point. `ENGINE=v403` selects `engine/c/zchezz_v403/`.

## Native release

```bash
make -C engine/build ENGINE=v403 native
```

On Windows with MinGW, use `mingw32-make` with the same arguments.

## Debug

```bash
make -C engine/build ENGINE=v403 debug
```

The debug build uses low optimization, debug symbols, and broad compiler warnings. Warnings are visible but are not globally promoted to errors until the existing codebase is clean enough for that policy.

## Sanitizers

```bash
make -C engine/build ENGINE=v403 sanitize
```

The sanitizer build uses AddressSanitizer and UndefinedBehaviorSanitizer. Do not combine it with static linking.

ThreadSanitizer is a separate future/optional job because Lazy SMP intentionally shares selected structures and requires careful suppression/interpretation.

## Native invariant harness

```bash
make -C engine/build ENGINE=v403 test-c
```

This compiles and runs `engine/c/tests/test_engine_invariants.c` against the selected engine sources and NNUE weights.

## WASM

```bash
make -C engine/build ENGINE=v403 wasm
make -C engine/build ENGINE=v403 bundle
```

WASM remains single-threaded and excludes file-based tablebases/book access by compile-time flags.

