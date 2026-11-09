# Boundary proposer baseline

Recorded before the Phase 1 batched candidate-assembly rewrite.

- Date: 2026-07-27
- Platform: macOS 14.4, Apple arm64
- PyTorch: 2.8.0
- Device: CPU (`torch.cuda.is_available() == False`; MPS is available)
- Commit: unavailable because this workspace copy has no `.git` metadata
- Command: `PYTHONPATH=. python benchmarks/benchmark_boundary_proposer.py`

## Existing proposer

| Shape | Median latency |
| --- | ---: |
| B=1, L=64, Q=4 | 5.295 ms |
| B=8, L=128, Q=8 | 77.664 ms |
| B=16, L=256, Q=8 | 183.553 ms |

The existing single-query dedup helper matched its Python reference for all
benchmark shapes. CUDA synchronization counts and the CUDA 8× speed target
cannot be measured on this host; the CUDA-only regression test skips when CUDA
is unavailable.

The richer `scripts/boundary_baseline.py` probe records forward/backward head
latency, candidate count, peak memory, and pre-injection proposal recall. Run
the same command before and after future proposal changes on the target
training GPU:

```bash
PYTHONPATH=. python scripts/boundary_baseline.py --device cuda
```

## Phase 1 result on the same CPU

| Shape | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| B=1, L=64, Q=4 | 5.295 ms | 1.681 ms | 3.15× |
| B=8, L=128, Q=8 | 77.664 ms | 9.291 ms | 8.36× |
| B=16, L=256, Q=8 | 183.553 ms | 29.176 ms | 6.29× |

The representative B=8 shape clears the 8× phase target on CPU. Candidate
assembly equivalence passes across randomized ties, masks, gold injection, and
capacity pressure. The CUDA no-host-sync gate remains pending on CUDA hardware.

The post-change full-head probe also completed on MPS (B=2, L=128, Q=8):
26.908 ms median forward, 25.695 ms median forward+backward, 128 candidates per
query, and 0.9 MiB reported current allocation. Random untrained weights yielded
zero proposal recall, as expected; learned-recall gates require a training/dev
dataset.

## Phases 4–7 local verification

On 2026-07-27, the post-roadmap MPS probe used:

```bash
PYTHONPATH=. python scripts/boundary_baseline.py \
  --device mps --warmup 3 --iterations 10
```

It measured 17.640 ms median forward latency, 19.907 ms median
forward-plus-backward latency, 128 candidates per query, and 0.9 MiB reported
current allocation. The synthetic untrained model again produced zero
pre-injection oracle recall.

The broad boundary regression gate completed with 218 passed, 1 skipped, and no
failures. The skip and the remaining no-host-sync/performance gate require CUDA;
learned recall, length-bucket F1, absent-query precision, and calibration gates
require a representative training/dev dataset.
