# Encoder feasibility pilot results

Protocol registered before training: [pilot protocol](pilot_protocol.md).

Full training pool, fixed validation split, seed 42, five epochs, float32 MPS eager attention. Test data was not evaluated.

Hardware: Apple M4, 24 GiB physical memory, Apple MPS.

| Model | Best validation micro-F1 | Best epoch | Training + validation (min) | Sampled RSS (GiB) | Sampled MPS driver (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| bert | 0.9072 | 5 | 1.86 | 0.66 | 3.22 |
| roberta | 0.8936 | 5 | 1.73 | 0.79 | 3.21 |
| modernbert | 0.8669 | 5 | 2.13 | 1.02 | 3.59 |

Working model selected by the preregistered rule: **bert**.

Within 0.01 absolute F1 of the highest score, prefer shorter summed training and validation time. All candidates completed and checkpoint reload reproduced their saved validation predictions.

## Per-type validation F1

| Model | Activity | PhysicalObject | Process | Property | State |
| --- | ---: | ---: | ---: | ---: | ---: |
| modernbert | 0.9347 | 0.8339 | 0.9412 | 0.8571 | 0.8921 |
| bert | 0.9709 | 0.8805 | 0.9811 | 0.7500 | 0.9155 |
| roberta | 0.9372 | 0.8718 | 0.9259 | 0.8571 | 0.9155 |

## Interpretation and limits

- Single-seed, short-budget selection; not proof of statistical superiority or final five-seed active-learning results.
- Fixed common hyperparameters and five epochs may not reveal each candidate's best achievable performance.
- Validation model/epoch selection makes these scores unsuitable as an unbiased final performance estimate.
- Property has only four validation mentions; its score is particularly unstable.
- Timing excludes download/loading, checkpoint I/O, and reload verification; it includes first-step device kernel setup.
- Sampled memory is not an exact peak. RSS and MPS driver allocations share unified memory; do not add them.
- MPS/eager speed comparisons may differ on CUDA or with FlashAttention. Inherent pretrained/tokenizer differences remain.
- Next: keep the chosen working model fixed, define acquisition protocol, and run the repeated sampling benchmark.

Machine-readable metrics, epoch history, software versions, and protocol/source/split checksums: [comparison.json](../benchmarks/pilot-v1/comparison.json).
