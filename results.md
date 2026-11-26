# Measured results

## Fictional headline pairs

Each order uses the same 120 fictional article families, 240 titles and a review budget of 60 titles.

| Routing policy | Original false passes / 120 | Reversed false passes / 120 |
| --- | ---: | ---: |
| Single A.X | 21 | 10 |
| Four A.X roles | 2 | 1 |
| Roles + exact quotation | 2 | 1 |
| Roles + A.X verifier | 0 | 1 |
| Roles + Phi verifier | 0 | 1 |

The primary comparison is Phi minus the A.X extra verifier on original order.

- Original order: 0.00 percentage points, paired 95% interval [0.00, 0.83].
- Reversed order: 0.00 percentage points, paired 95% interval [-1.67, 0.00].

These intervals include zero. The low false-pass rate after either extra verifier creates a floor on the controlled fictional data.

## H100 verifier runtime

Results use 48 fictional titles and three repetitions per cell. Timings cover one verifier, with input preparation and token decoding in the measured service loop. The p95 is a batch latency.

| Verifier | Batch | Schedule | Titles/s | p95 batch ms | Peak allocated GiB | Decision changes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A.X | 1 | original | 11.31 | 106.4 | 13.59 | 0 |
| A.X | 1 | length_sorted | 11.46 | 103.9 | 13.59 | 0 |
| A.X | 4 | original | 27.14 | 154.4 | 13.71 | 0 |
| A.X | 4 | length_sorted | 28.62 | 151.6 | 13.71 | 0 |
| A.X | 8 | original | 46.13 | 180.3 | 13.86 | 0 |
| A.X | 8 | length_sorted | 46.51 | 177.6 | 13.86 | 0 |
| A.X | 16 | original | 69.04 | 241.8 | 14.16 | 0 |
| A.X | 16 | length_sorted | 69.70 | 238.1 | 14.16 | 0 |
| A.X | 32 | original | 83.83 | 337.0 | 14.76 | 0 |
| A.X | 32 | length_sorted | 85.14 | 332.2 | 14.73 | 0 |
| Phi | 1 | original | 10.60 | 111.9 | 7.23 | 0 |
| Phi | 1 | length_sorted | 10.65 | 109.1 | 7.23 | 0 |
| Phi | 4 | original | 26.44 | 156.8 | 7.41 | 0 |
| Phi | 4 | length_sorted | 27.86 | 153.6 | 7.41 | 1 |
| Phi | 8 | original | 46.68 | 173.9 | 7.64 | 0 |
| Phi | 8 | length_sorted | 48.09 | 174.5 | 7.64 | 1 |
| Phi | 16 | original | 74.47 | 216.1 | 8.11 | 0 |
| Phi | 16 | length_sorted | 75.48 | 216.5 | 8.11 | 1 |
| Phi | 32 | original | 88.22 | 329.6 | 9.03 | 0 |
| Phi | 32 | length_sorted | 90.37 | 316.1 | 8.93 | 0 |

## SDPA microbenchmark

Synthetic key/value tensors match the A.X attention dimensions. See `results/attention.json` for available implementations, CUDA event timings and error against the FP32 math reference.

End-to-end verifier timings and kernel microbenchmark timings capture different workloads. The benchmark comparison does not establish a production latency improvement.
