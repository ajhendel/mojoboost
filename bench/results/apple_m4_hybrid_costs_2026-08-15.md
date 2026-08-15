# Hybrid cost calibration, Apple M4, 2026-08-15

Experiment E1 of `docs/design/HYBRID_TRAINING.md` §9, re-run with the two
coefficients the 2026-08-14 calibration did not have: the host's
*scattered* accumulation rate and the *fixed* price of a host build. All
ten `HybridCosts` coefficients come from this one run — this machine's
timings drift several-fold between time windows (bench/README.md), so a
model must not mix runs. Measured by `bench/bench_hybrid_costs.mojo`
(`pixi run bench-hybrid-costs`) at 500,000 rows x 50 features x 255 bins,
five interleaved trials, minimum reported, on an otherwise idle machine.
These are the numbers `HybridCosts.apple_m4()` cites.

```
hybrid cost calibration: 500000 rows x 50 features x 255 bins, 5 trials of 50 reps

raw minima (nanoseconds):
  synchronize         20560
  launch              62128
  row readback        653240 ( 1953.125 KiB )
  convert             127800
  device build_leaf   4256600
  host replica build  29482800
  host replica build, scattered leaf 895800 ( 7812 rows, gap 64 )
  host replica build, scattered leaf 310600 ( 976 rows, gap 512 )
  host replica build, scattered leaf 57800 ( 122 rows, gap 4096 )
  host replica build, one row 41420
  host float64 build  16353000
  host zero pass      140
  host partition      7686800

HybridCosts coefficients (nanos per thousand units):
  launch_nanos                = 62128
  sync_nanos                  = 20560
  transfer_nanos_per_kib      = 334
  device_nanos_per_krow_slot  = 160
  host_nanos_per_krow_slot    = 1178
  host_fixed_nanos            = 41280
  host rate at row gap 64 = 2187
  host rate at row gap 512 = 5516
  host rate at row gap 4096 = 2685
  host_scatter_nanos_per_krow_slot = 5516 (max over gaps)
  host_partition_nanos_per_krow = 15374
  host_zero_nanos_per_kcell   = 11
  convert_nanos_per_kcell     = 10024
```

## What changed since 2026-08-14, and why

- **The per-range readback exists.** `DeviceBuffer.create_sub_buffer` plus
  `enqueue_copy` moves exactly one node's window of the active-row buffer
  (`GpuHistogramBuilder.readback_range`), which HYBRID_TRAINING.md §3 had
  assumed was not expressible. Measured on this machine it costs ~45–50 µs
  flat — the synchronize (~25 µs) plus a copy enqueue (~20 µs) — whatever
  the buffer size (20k, 100k, 1M rows) or the window (500 or 4000 rows).
  The whole-permutation snapshot it replaces cost 280 µs at 20k rows and
  2.6 ms at 1M. So the grower now plans every leaf as `ROWS_DEVICE_COPY`,
  charges each leaf only its own transfer, and takes no snapshot; the
  `DECLINE_SNAPSHOT_NOT_PAID` regime that kept hybrid inert above ~150k
  active rows is gone. (The copy-enqueue overhead is not a separate
  coefficient: the device path pays the same enqueue on its histogram
  download, so it cancels in the comparison.)

- **The host rate depends on the node's row density.** Bins are
  feature-major, so a small leaf of a large dataset reads one cache line
  (and, thinner still, one prefetch/TLB miss) per bin. The contiguous root
  accumulates at 1,178 ns per thousand (row, slot) pairs; a leaf of every
  64th row at 2,187, of every 512th at 5,516, of every 4096th at 2,685 (122
  rows — a small sample, and the fixed term dominates it). Priced with the
  contiguous rate alone, the scheduler elected leaves at 500k–1M rows whose
  real host cost was 2–3x the model's, and those fits lost. The model now
  ramps linearly from the contiguous rate at gap 1 to the largest measured
  scattered rate at gap 256 and stays there (`host_slot_nanos_per_k`),
  which sits above every gap this run measured.

- **A host build has a fixed price**: 41 µs for a one-row build, which is
  the quantization dispatch, the per-feature task fan-out, and the
  dequantization pass. Two thirds of the device's launch, so it is not
  negligible against the ~260 µs device fixed cost the substitution
  targets; it is now `host_fixed_nanos`. (Fusing the two dispatches would
  roughly halve it and is the obvious next lever on the host side.)

- **The device-gradient path is reachable.** `stage_from_device` reads the
  device's exact Float32 gradients back into the staging buffers once per
  tree (`8 * n_rows` bytes and one synchronize), so the unbagged built-in
  objectives — which generate gradients on the device — get the same
  replica the upload path gets. Verified bit-identical in
  `tests/parallel/test_hybrid_replica.mojo` and in every arm below.

## Where the scheduler now places work

`MOJOTREES_HYBRID_TRACE=1` on this shape (50 features, 255 bins, 127
leaves, `replica`, forced host split search) elects per tree roughly:
every split at 6k rows, most at 20k, a minority at 100k, ~5 at 500k, and
**none at 1M**, where every leaf is either large (device wins on
throughput) or so thin that the scattered host rate and the fixed term
exceed the device's fixed cost. That is the model being conservative
where the 2026-08-14 model was wrong, not a limit of the mechanism.

Note that the workload-aware AUTO split-search policy already puts runs
above roughly 250k rows on **device** split search on this machine, where
hybrid scheduling is inapplicable by design (`DECLINE_NO_HOST_PARENT`: no
host-resident parent histogram to subtract from). Hybrid's domain is the
host-search regime, which is exactly where the device's fixed per-node
costs are the largest fraction of a tree.

## End-to-end effect

Interleaved off/replica arms in one process, predictions bit-identical in
every trial (checked in-run). 50 features x 255 bins, `num_leaves` 127, 20
rounds. Timings on this machine drift several-fold between windows and
several sessions share it; the arms below are the ones taken while it was
otherwise idle, and each row is one interleaved pair.

Bagged (`bagging_fraction` 0.8, host-staged gradients — the path the first
integration reached):

```
20k   1.29x  1.17x  1.25x
100k  0.98x  1.05x  1.01x
```

Unbagged squared error (device-generated gradients — newly reachable):

```
20k   1.17x  1.15x  1.12x
100k  1.08x  1.07x  0.95x
```

Above that the AUTO policy takes device split search and both arms are the
same code; with host search forced at 500k the scheduler elects ~5 leaves a
tree and at 1M none, and neither is distinguishable from the pure-device
arm on this machine (their arms fell in contended windows and are not
reported as numbers).

`MOJOTREES_HYBRID_GUARD_TRANSFER=1` (the design's transfer-dominates guard,
which under a per-range readback refuses exactly the small leaves the
comparison exists to move) measured 0.83–1.10x on the bagged 20k shape
against 1.22–1.47x with it off in the same window; the grower leaves it
off.
