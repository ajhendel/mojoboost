# Apple M4 early stopping on the GPU: gradient source and validation scorer, 2026-08-15

The measurement behind `train_gpu_with_valid`'s two switches. Before this run
the early-stopping trainer computed every round's gradients on the host and
walked each tree over every training row on the host to keep Float64 raw
scores current, whichever scorer was chosen, because the module said early
stopping needed those host raw scores. It does not: the stopping rule reads
only the validation loss. `train_gpu_with_valid` now takes `train_gpu`'s
`objective_source` and, under AUTO, generates a plain run's gradients on the
device exactly as `train_gpu` does. This record measures that, and measures
the host versus device validation scorer beside it.

## Environment

Same machine, toolchain, and dataset generator as
`apple_m4_crossover_2026-08-15.md`, run in the same window (load 2.5 to 5).
The validation block is drawn from the same stream after the training block.
`n_estimators=200`, `early_stopping_rounds=10`, otherwise library defaults.

```sh
pixi run bench-valid-scoring ROWS VALID_ROWS 50 reg|binary 3
```

Five arms interleaved: `cpu` is `train_with_valid`; the GPU arms are
`train_gpu_with_valid` with host or device gradients (`hg`/`dg`,
`objective_source`) crossed with the host or device scorer (`hs`/`ds`,
`valid_scoring`). Each arm's kept tree count and final validation loss are
printed beside its time. Raw outputs are in
`apple_m4_valid_scoring_2026-08-15/`.

## Results

| problem | arm | min s | spread | trees kept | valid loss | vs cpu |
|---|---|---:|---:|---:|---:|---:|
| 500k / 100k x 50 reg | cpu | 12.170 | 18.8% | 200 | 0.0027574 | |
| | hg-hs (what shipped) | 8.918 | 2.3% | 200 | 0.0028134 | 1.36x |
| | dg-hs | 5.361 | 4.0% | 200 | 0.0028132 | 2.27x |
| | hg-ds | 8.341 | 1.9% | 200 | 0.0028134 | 1.46x |
| | dg-ds | 4.995 | 0.7% | 200 | 0.0028132 | 2.44x |
| 1M / 200k x 50 reg | cpu | 27.162 | 6.8% | 200 | 0.0027595 | |
| | hg-hs | 17.702 | 17.4% | 200 | 0.0027752 | 1.53x |
| | dg-hs | 10.509 | 15.0% | 200 | 0.0027752 | 2.58x |
| | hg-ds | 16.859 | 7.4% | 200 | 0.0027752 | 1.61x |
| | dg-ds | 9.575 | 16.7% | 200 | 0.0027752 | 2.84x |
| 500k / 100k x 50 binary | cpu | 7.479 | 2.0% | 74 | 0.290345 | |
| | hg-hs | 4.734 | 1.8% | 85 | 0.290319 | 1.58x |
| | dg-hs | 2.792 | 1.6% | 85 | 0.290319 | 2.68x |
| | hg-ds | 4.504 | 23.0% | 85 | 0.290319 | 1.66x |
| | dg-ds | 2.650 | 1.1% | 85 | 0.290319 | 2.82x |

Every `vs cpu` delta is `resolved` by the driver's spread rule.

## Reading

- Device gradients over host gradients, same scorer: 1.66x at 500k, 1.68x at
  1M, 1.70x on binary. That is the per-tree host `predict_row` walk and the
  per-round gradient upload leaving the loop; it is the same saving the
  bagged record measures for the same reason.
- The kept tree count is identical across the four GPU arms in every problem
  (200, 200, 85), so neither switch changed a stopping decision here. The
  regression runs hit the 200-round cap; the binary run stopped on the rule at
  85 on every GPU arm and at 74 on the CPU, which is the Float32-histogram
  difference between the backends deciding a different tree sequence, not the
  scorer: `hs` and `ds` agree with each other exactly.
- Device scorer over host scorer, same gradients: 6% to 9%. Real and
  resolved, but small: on this problem the validation walk is a tenth of the
  round.

## What changed because of this, and what did not

`train_gpu_with_valid` gained `objective_source` and takes the device
gradient path under AUTO for a plain run, mirroring `train_gpu`; a bagged
early-stopping run takes the device round too (see the bagged record); GOSS
keeps the host path. `tests/parallel/test_gpu_valid.mojo` pins that the
stopping decision and the kept models agree with the host path and the CPU
trainer.

The validation scorer default stays the host walk. The measured gain is 6%
to 9%, and the device scorer accumulates the running validation scores in
Float32, which can order two rounds inside Float32 noise of each other
differently than the host does and pick a different `best_iteration`; that
did not happen in these three problems, but the gain is not large enough to
buy the risk by default. `VALID_SCORE_DEVICE` (or
`MOJOTREES_GPU_VALID_SCORING=device`) opts in.

## Caveats

Same as the crossover record: a shared, non-idle machine, seed 0 only. The
regression problems ran to the 200-round cap, so their stopping decisions
are trivially equal; the binary problem is the one that exercised the rule.
