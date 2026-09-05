# Zchezz v5.00 strength experiments

Branch: `v500-strength`.

Hard constraints for the current campaign:

- Start from the v4.03 HalfKP/4-king-bucket architecture.
- Keep the NNUE first stage fully incremental (lazy perspective rebuild on king-bucket crossing is allowed).
- Do not reintroduce the 31 manual v3.14 features in this campaign.
- Use v3.14 as the strength reference.
- Separate fixed-node tree-quality tests from fixed-time/NPS tests.
- Promote changes only after paired-opening head-to-head evidence.

Initial experiment families:

1. NNUE output-scale sweep at fixed nodes.
2. Search ablations at fixed nodes (quiet SEE pruning and pruning-margin families).
3. Search-parameter retuning against the v5.00 NNUE.
4. If evaluation remains the limiting factor, retrain the full-accumulator NNUE and consider a wider post-accumulator hidden layer.
