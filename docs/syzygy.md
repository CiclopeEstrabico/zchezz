# Syzygy Integration

Syzygy has separate build-time and runtime prerequisites.

## Build-time Fathom availability

The native build enables tablebases only when the selected engine directory
contains the complete Fathom source/header pair expected by the Makefile.

A clean checkout that does not contain local Fathom files remains buildable:
the Makefile defines `NO_TABLEBASES`.

Check the state with:

```bat
mingw32-make -C engine/build ENGINE=v403 build-info
```

Require tablebase capability with:

```bat
mingw32-make -C engine/build ENGINE=v403 require-tablebases
```

A PASS from `require-tablebases` means the Fathom sources required by the
native build are present. It does not prove that runtime tablebase files exist.

## Runtime tablebase availability

A tablebase-capable binary still needs a configured Syzygy directory.

`tests/test_uci_extended.py` group T3 behaves as follows:

- missing tablebase directory → SKIP with an explicit reason;
- available directory → set `SyzygyPath`, require readiness, run tablebase
  positions, and require functional probe evidence (`tbhits`) where applicable.

The test does not require a particular human-readable "loaded" message because
diagnostic wording and stdout/stderr routing are not part of the UCI contract.

## WebAssembly

WebAssembly always builds without native tablebases.

## Square convention

Zchezz uses `a8=0` through `h1=63`. Mapping to Fathom's convention is a
correctness boundary. Keep direct mapping fixtures and end-to-end probe tests
when changing this code.

## Release claims

Do not claim native Syzygy coverage from a clean no-Fathom CI build. A release
that changes tablebase code must include evidence from an environment where:

1. `require-tablebases` passes;
2. the required tablebase files exist;
3. UCI T3 runs rather than skips.
