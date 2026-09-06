# v5.00 strength status

Last updated: 2026-09-06.

## Accepted baseline

The current v5.00 strength baseline is the existing NNU4 HalfKP network plus the promoted lazy HalfKP accumulator.

Architecture remains:

- HalfKP with 4 king buckets.
- `2560 -> 512`, SCReLU.
- Dual-perspective concat `1024 -> 32 -> 1`.
- Full incremental accumulator with lazy materialization.
- No reintroduction of the 31 manual v3.14 features.

The lazy accumulator is retained because it preserves fixed-node semantics while improving throughput. Scalar and AVX2 paths were parity-tested after the portability fix.

## Current strength gap

Against v3.14:

- 50k nodes, 512 games: approximately `-154.7 +/- 30.9 Elo`.
- 100 ms, 512 games: W/D/L `144/82/286`, `-98.95 +/- 28.48 Elo`.

Therefore v5.00 is not yet stronger than v3.14. The remaining deficit is not explained by throughput alone; quality per node is also materially lower.

## Corrected 100k teacher-target sweep

All candidates started from the accepted v5 weights and used the repaired v3.14 teacher corpus. Black-to-move teacher labels were corrected before training.

| Result blend lambda | vs current v5 @ 50k | vs v3.14 @ 50k | Decision |
| ---: | ---: | ---: | --- |
| 0.00 | `-112.55 +/- 42.48` | `-239.99 +/- 46.96` | Reject |
| 0.05 | `-109.33 +/- 41.70` | `-219.16 +/- 45.28` | Reject |
| 0.15 | `-51.96 +/- 40.19` | `-176.66 +/- 44.94` | Reject |

No network from this sweep is promoted. The existing NNUE weights remain the v5.00 baseline.

## Rejected throughput screens

- Combined dual-perspective lazy materialization: no measurable NPS gain; reject.
- AVX2 L2 8-output unroll: about 6.7% lower NPS; reject.
- AVX-VNNI-only build: unsupported on the reference runner; reject as a portable baseline.

## Next priority

Do not continue 100k v3.14 teacher fine-tuning with the same recipe. The next experiments should target the quality-per-node deficit, with priority on training/runtime parity and then network/search capacity. Preserve HalfKP, king buckets, full accumulator, and the no-manual-feature constraint until explicitly changed.
