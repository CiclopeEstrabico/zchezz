# NNUE Engineering Notes

The v403 core uses an NNU4 HalfKP-4Bucket network.

Compiled dimensions are 2560 input features per perspective, 512 L1 outputs, 1024 concatenated L2 inputs, 32 L2 outputs, and one scalar output. Mutable accumulator state is per thread. Network weights are immutable after loading.

Critical contracts:

- feature encoding and king buckets must match the trainer/exporter exactly;
- kings are not direct features;
- concat order is `[stm, opp]`;
- a king-bucket transition marks one perspective dirty;
- lazy rebuild occurs from the post-move board before that perspective is evaluated;
- incremental evaluation must equal a full rebuild.

Use `python tools/check_nnue.py` for NNU4 header, dimension, length, and SHA-256 sanity. Use the native invariant harness for incremental/full-rebuild equivalence.

