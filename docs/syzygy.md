# Syzygy Integration

Syzygy support has separate build-time and runtime availability.

Native tablebase support requires both `tbprobe.c` and `tbprobe.h` in the
selected engine directory. These Fathom files are local-only in the current
repository. If either is absent, the shared Makefile builds the native engine
with `NO_TABLEBASES` so a clean checkout remains buildable.

Use:

`make -C engine/build ENGINE=v403 require-tablebases`

when a build must prove Fathom is available.

A tablebase-enabled binary still needs a configured Syzygy directory. Tests
that require tablebase files must report an explicit skip when those files are
unavailable.

The WebAssembly build always defines `NO_TABLEBASES`.

The engine square convention is `a8=0` through `h1=63`. Mapping to Fathom is a
correctness boundary and should be protected by direct fixtures and search
tests.
