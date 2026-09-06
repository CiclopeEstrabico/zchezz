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

All candidates started from the accepted v5 weights and used the repaired v3.14 teacher corpus. Black-to-move teacher labels were corrected before training. The source contained 100,000 rows, including 49,686 repaired Black-to-move rows.

| Result blend lambda | vs current v5 @ 50k | vs v3.14 @ 50k | Decision |
| ---: | ---: | ---: | --- |
| 0.00 | W/D/L `76/34/146`, `-97.48 +/- 40.95` | W/D/L `56/27/173`, `-171.48 +/- 44.55` | Reject |
| 0.05 | W/D/L `74/30/152`, `-109.33 +/- 41.70` | W/D/L `37/39/180`, `-219.16 +/- 45.28` | Reject |
| 0.15 | W/D/L `93/32/131`, `-51.96 +/- 40.19` | W/D/L `55/26/175`, `-176.66 +/- 44.94` | Reject |

No network from this sweep is promoted. The existing NNUE weights remain the v5.00 baseline. The 0.15 blend was the least harmful of these three experiments, but it still lost materially to the accepted baseline.

## Rejected throughput screens

- Combined dual-perspective lazy materialization: no measurable NPS gain; reject.
- AVX2 L2 8-output unroll: about 6.7% lower NPS; reject.
- AVX-VNNI-only build: unsupported on the reference runner; reject as a portable baseline.

## Next priority

Do not continue 100k v3.14 teacher fine-tuning with the same recipe. The next experiments should target the quality-per-node deficit.

Priority order:

1. Make the Python/QAT forward path match the C integer inference path as closely as possible, ideally bit-exact where practical.
2. Revalidate import/export and training identity before another large training run.
3. Retrain with a substantially larger, well-labelled corpus once training/runtime parity is trustworthy.
4. If the training path is healthy but capacity still limits strength, test hidden layer 2 width `32 -> 64`, then `128` only if justified.
5. Revisit search improvements after the evaluation path is no longer the dominant uncertainty.

Preserve HalfKP, king buckets, full accumulator, and the no-manual-feature constraint until explicitly changed.