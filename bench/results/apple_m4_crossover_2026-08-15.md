# Apple M4 CPU versus GPU training crossover, 2026-08-15

The measurement behind `crossover_rules()` in `src/mojotrees/device_policy.mojo`
(policy version 2): where end-to-end GPU training first beats the multicore
CPU trainer on the development M4, by shape and objective. This is a
development result on a shared, non-idle machine, not a production
performance claim; the caveats section says exactly how non-idle.

## Environment

- Apple M4, 10 physical CPU cores, 10 GPU cores, 16 GB memory
- macOS 26.5.2 (25F84), arm64
- Mojo 1.0.0, MAX 26.5.0; `DeviceContext` reports `api=metal`, `arch_name=4-metal4`
- Library defaults (LightGBM-matched): 100 trees, 31 leaves, learning rate 0.1,
  255 bins, L2 regularization 1.0; `device="gpu"` arm is `train_gpu` under
  `SPLIT_SEARCH_AUTO` and `OBJECTIVE_SOURCE_AUTO`, i.e. what a caller gets
- CPU arm is `train` with the default worker pool (`MOJOTREES_NUM_WORKERS` unset)
- Dataset: bench_train_gpu.mojo's deterministic synthetic problem, seed 0, so
  every row of this table trains on the same data as the Aug 14 large-scaling
  record for the same shape
- Machine NOT idle: other sessions were building and testing throughout
  (load average 2.5 to 5, briefly 10 to 18); see caveats

Command, per row (`repeats` interleaves the two arms inside one process):

```sh
pixi run bench-train-gpu ROWS FEATURES reg|binary REPEATS cpu,gpu 0
```

The driver reports each arm's minimum, its spread (max/min minus one) over the
repeats, the speedup of minima, and a verdict: `resolved` when the delta
exceeds the wider of the two arms' own spreads, `indistinguishable` otherwise.
Raw outputs are in `apple_m4_crossover_2026-08-15/`.

## Results

Four sweeps. Sweep 1 ran at load 2.5 to 3.8. Sweep 2 was aborted after four
configurations when the load reached 18 (a peer's build plus a peer's test
suite); its rows are listed for completeness and were not used. Sweep 3 reran
the bracket at load 3 to 5 with five repeats per configuration and added the
binary objective and two more shapes. Sweep 4, at load 2.4 to 3.3, reran the
two binary rows sweep 3 could not resolve and added 250,000 x 100.

| sweep | rows | features | objective | cells | repeats | CPU min s | CPU spread % | GPU min s | GPU spread % | GPU speedup | verdict | CPU loss | GPU loss |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 200,000 | 50 | binary | 10M | 3 | 3.052 | 18.0 | 2.516 | 13.1 | 1.21x | indistinguishable | 0.280105 | 0.280176 |
| 1 | 100,000 | 20 | reg | 2M | 3 | 0.993 | 29.6 | 1.347 | 28.1 | 0.74x | resolved | 0.003806 | 0.003788 |
| 1 | 50,000 | 50 | reg | 2.5M | 3 | 1.137 | 25.7 | 2.234 | 15.4 | 0.51x | resolved | 0.003888 | 0.003926 |
| 1 | 50,000 | 100 | reg | 5M | 3 | 1.898 | 30.0 | 2.917 | 12.3 | 0.65x | resolved | 0.003782 | 0.003793 |
| 1 | 100,000 | 50 | reg | 5M | 3 | 1.710 | 3.3 | 3.163 | 1.4 | 0.54x | resolved | 0.003858 | 0.003820 |
| 1 | 100,000 | 100 | reg | 10M | 3 | 3.378 | 11.5 | 3.886 | 3.3 | 0.87x | resolved | 0.003615 | 0.003604 |
| 1 | 200,000 | 50 | reg | 10M | 3 | 2.845 | 19.8 | 2.085 | 38.1 | 1.36x | indistinguishable | 0.003605 | 0.003605 |
| 1 | 500,000 | 50 | reg | 25M | 3 | 6.705 | 10.5 | 2.654 | 0.8 | 2.53x | resolved | 0.003615 | 0.003476 |
| 2 | 500,000 | 20 | reg | 10M | 5 | 4.504 | 65.7 | 1.798 | 60.9 | 2.51x | indistinguishable | 0.003573 | 0.003617 |
| 2 | 250,000 | 50 | reg | 12.5M | 5 | 7.069 | 238.8 | 9.129 | 112.4 | 0.77x | indistinguishable | 0.003577 | 0.003577 |
| 2 | 300,000 | 50 | reg | 15M | 5 | 3.948 | 27.2 | 2.273 | 33.8 | 1.74x | resolved | 0.003544 | 0.003549 |
| 2 | 200,000 | 100 | reg | 20M | 5 | 4.404 | 53.7 | 3.122 | 33.3 | 1.41x | indistinguishable | 0.003599 | 0.003603 |
| 3 | 200,000 | 50 | binary | 10M | 5 | 2.885 | 48.4 | 2.072 | 38.5 | 1.39x | indistinguishable | 0.280105 | 0.280176 |
| 3 | 500,000 | 50 | binary | 25M | 5 | 7.840 | 28.1 | 2.540 | 69.2 | 3.09x | indistinguishable | 0.286389 | 0.286339 |
| 3 | 1,000,000 | 50 | binary | 50M | 3 | 16.328 | 124.6 | 5.837 | 98.5 | 2.80x | indistinguishable | 0.288565 | 0.288545 |
| 3 | 100,000 | 50 | reg | 5M | 5 | 1.616 | 11.9 | 2.523 | 28.1 | 0.64x | resolved | 0.003858 | 0.003820 |
| 3 | 100,000 | 100 | reg | 10M | 5 | 3.393 | 18.8 | 2.771 | 3.6 | 1.22x | indistinguishable | 0.003615 | 0.003604 |
| 3 | 200,000 | 50 | reg | 10M | 5 | 2.966 | 14.3 | 2.369 | 17.1 | 1.25x | resolved | 0.003605 | 0.003605 |
| 3 | 500,000 | 20 | reg | 10M | 5 | 6.777 | 15.8 | 2.774 | 33.9 | 2.44x | resolved | 0.003573 | 0.003617 |
| 3 | 300,000 | 50 | reg | 15M | 5 | 3.928 | 62.0 | 2.214 | 52.1 | 1.77x | indistinguishable | 0.003544 | 0.003549 |
| 3 | 200,000 | 100 | reg | 20M | 5 | 5.596 | 23.7 | 3.091 | 40.6 | 1.81x | resolved | 0.003599 | 0.003603 |
| 3 | 1,000,000 | 20 | reg | 20M | 3 | 12.903 | 6.4 | 2.713 | 65.0 | 4.76x | resolved | 0.003466 | 0.003473 |
| 3 | 500,000 | 50 | reg | 25M | 5 | 8.386 | 52.8 | 2.598 | 14.6 | 3.23x | resolved | 0.003615 | 0.003476 |
| 3 | 1,000,000 | 50 | reg | 50M | 3 | 23.699 | 43.5 | 7.182 | 49.4 | 3.30x | resolved | 0.003518 | 0.003456 |
| 4 | 500,000 | 50 | binary | 25M | 5 | 5.608 | 54.9 | 2.509 | 5.7 | 2.23x | resolved | 0.286389 | 0.286339 |
| 4 | 1,000,000 | 50 | binary | 50M | 3 | 15.487 | 4.7 | 4.283 | 9.5 | 3.62x | resolved | 0.288565 | 0.288545 |
| 4 | 250,000 | 100 | reg | 25M | 5 | 6.743 | 12.5 | 3.191 | 15.3 | 2.11x | resolved | 0.003601 | 0.003603 |

The 1,000,000 x 50 regression point also has an idle-machine measurement in
`apple_m4_large_scaling_2026-08-14.md`: 2.62x over three seeds at load 1.3.

## Reading

- Below 10 million cells the GPU trainer loses at every shape tried (0.51x to
  0.87x). At these sizes a tree is a few dozen launches over small data, and the
  fixed cost per node (measured elsewhere at about 257 us) is what the run pays
  for; the CPU trainer's histogram loops finish inside that.
- Around 10 million cells the verdict depends on shape. 500,000 x 20 won 2.4x
  resolved; 200,000 x 50 won 1.25x resolved in sweep 3 and was inside the noise
  in sweep 1; 100,000 x 100 was inside the noise in both. Cells alone do not
  order the marginal shapes: rows do more for the GPU than features do.
- From 25 million cells and 200,000 rows up the GPU won by more than the noise
  floor at every regression shape in every sweep (2.5x to 4.8x), and the Aug 14
  idle-machine run agrees at 50 million cells.
- Binary logistic tracks regression at every size: 1.2x to 1.4x at 10
  million cells (inside the noise), then 2.2x resolved at 25 million and 3.6x
  resolved at 50 million in sweep 4. Sweep 3's two large binary rows are
  marked `indistinguishable` by the driver's own rule only because the CPU
  arm's spread reached 69% and 125% during a peer's build; even there every
  interleaved pair had the GPU ahead by 2x or more and the GPU arm's slowest
  repeat beat the CPU arm's fastest (4.30 s against 7.84 s; 11.6 s against
  16.3 s). The sweep 4 reruns settle it.
- 250,000 x 100 (25 million cells, the wide end of the claimed region) won
  2.1x resolved in sweep 4, so the cell floor holds at 100 features as well
  as at 20 and 50.
- Training losses agree between the two arms to Float32-level tolerance at
  every shape (fourth significant digit or better), so no row is a speed bought
  with fit.

## What the rules take from this

Two rules, `apple-m4-metal-l2-dense` and `apple-m4-metal-binary-dense`, each
scoped to Metal on an M4 generation part, one output, its own objective, and
requiring both 25,000,000 cells and 200,000 rows. The cell floor is the first
resolved win, not the estimated break-even (which sits nearer 10 to 15 million
for row-heavy shapes). The row floor is the smallest row count measured
winning; nothing narrower and taller than 200,000 x 100 was measured, so a
wide, short matrix (for example 100,000 x 250) is not claimed. Both
objectives were resolved at both 25 and 50 million cells. Multiclass,
other objectives, and other hardware are not claimed.

## Caveats

- The machine was shared with other sessions running Mojo builds and GPU tests
  for the whole window. Interleaving the arms inside one process is what makes
  the ratios usable at all; the absolute times are not comparable across rows
  or with the Aug 14 record. External CPU load inflates the CPU arm more than
  the GPU arm, so a ratio here is an upper bound on the idle-machine ratio;
  the sweep 1 versus sweep 3 comparison at 500,000 x 50 (2.53x at load 3,
  3.23x at load 4.5) shows the size of that effect. The 25 million cell floor
  keeps a 2x margin over that inflation.
- The CPU worker pool was left at its default rather than pinned.
- Only seed 0. The Aug 14 record covers seeds 0 to 2 at 1 and 5 million rows.
- 100 rounds only; a longer run amortizes the GPU's per-fit setup further.
