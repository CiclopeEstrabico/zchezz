# Syzygy Integration

Syzygy support is optional at runtime and absent from the WebAssembly build.

Tests that require tablebase files must report an explicit skip if the configured tablebase directory is unavailable. They must not silently convert a tablebase test into a non-tablebase test.

The engine's square convention is `a8=0` through `h1=63`. Mapping to external tablebase conventions is a correctness boundary and should be protected by direct fixtures in addition to full search tests.

