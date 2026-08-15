# Apple M4 bagged rounds: host path versus device round, 2026-08-15

The measurement behind `_routes_all_rows` in `src/mojotrees/train_gpu.mojo`:
under row bagging, does the device round (gradients generated on the device,
every row's raw score advanced by `GpuTreeRouter` after each tree) beat the
host-gradient path it replaces (host Float64 raw scores, host gradients
uploaded per round, one `predict_row` per training row per tree)? Before this
run a bagged `train_gpu` under `OBJECTIVE_SOURCE_AUTO` took the host path and
only an explicit `OBJECTIVE_SOURCE_DEVICE` took the device round.

## Environment

Same machine, toolchain, dataset generator, and library defaults as
`apple_m4_crossover_2026-08-15.md`, run in the same window (load 4 to 8, a
peer's extension build running for the 500k row). Bagging: `subsample=0.8`,
every tree, seed 0, so both GPU arms and the CPU arm draw identical bags.

```sh
pixi run bench-train-gpu ROWS 50 reg 3 cpu,gpu-obj-host,gpu-obj-device 0 0.8
```

Three arms interleaved: `cpu` is `train`; `gpu-obj-host` is `train_gpu` with
`objective_source=OBJECTIVE_SOURCE_HOST`; `gpu-obj-device` is
`objective_source=OBJECTIVE_SOURCE_DEVICE`. Raw outputs are in
`apple_m4_bagged_round_2026-08-15/`.

## Results

| rows x features | CPU min s | host path min s (spread) | device round min s (spread) | device over host | device over CPU | host-path loss | device-round loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 200,000 x 50 | 2.814 | 2.866 (2.5%) | 2.054 (5.8%) | 1.40x | 1.37x resolved | 0.0037211 | 0.0037195 |
| 500,000 x 50 | 7.136 | 5.087 (14.9%) | 2.796 (36.6%) | 1.82x | 2.55x resolved | 0.0036904185 | 0.0036904182 |
| 1,000,000 x 50 | 14.165 | 9.318 (4.9%) | 4.409 (21.1%) | 2.11x | 3.21x resolved | 0.0035770185 | 0.0035770190 |

## Reading

- The device round beats the host path at every size, and by more as the
  size grows: the host path's per-tree `predict_row` walk over all n rows and
  its per-round gradient upload are the two host-side per-row costs a bagged
  run was still paying, and they scale with n.
- The two arms' training losses agree to the seventh significant digit at
  500k and 1M and to the fourth at 200k, which is the Float32-level agreement
  the device path is specified to (`GpuTreeRouter` walks integer bins, so
  the trees are the same; only the Float32 raw-score accumulation differs).
- The bagged crossover against the CPU is at or below 200,000 x 50 (1.37x
  resolved there), lower than the unbagged one, because bagging removes 20%
  of the rows from each tree's histograms without removing any of the CPU
  trainer's fixed per-tree work.

## What changed because of this

`_routes_all_rows` now answers True for a bagged run under AUTO as well as
under an explicit device request; `OBJECTIVE_SOURCE_HOST` (or
`MOJOTREES_GPU_OBJECTIVE=host`) is the switch back. GOSS is unchanged: its
sample is a ranking of the gradients, and ranking Float32 device gradients
can pick a different row at the threshold than the CPU trainer does, so it
keeps the host path and identical sampling. `tests/test_gpu_training.mojo`'s
bagged tests (tree shapes identical to the CPU trainer, predictions within
1e-3) pass on the device round.

## Caveats

Same as the crossover record: a shared, non-idle machine, seed 0 only, 100
rounds. The 500k row's device arm spread (36.6%) came from one slow first
repeat during a peer's build; its minimum and its two later repeats agree
to 1%.
