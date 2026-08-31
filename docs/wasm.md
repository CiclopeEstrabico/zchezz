# WebAssembly and Browser Verification

The browser engine is a single-threaded WebAssembly build. It excludes file-based Syzygy and opening-book code through compile-time flags.

The web quality gate checks:

1. bundle generation succeeds;
2. the page loads without unexpected console/page errors;
3. WASM and NNUE initialization complete;
4. game mode can move beyond the opening book;
5. analysis starts and stops;
6. MultiPV changes produce additional lines;
7. new-game/reset clears state;
8. clocks and promotion UI behave correctly;
9. the standalone bundle works without runtime network dependencies.

Browser automation should be headless by default and save diagnostic screenshots on failures.

