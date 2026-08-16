# Hybrid cost calibration, Apple M4, 2026-08-14

> **Status note, 2026-08-16.** This is a measured result and it stands as
> measured. The code it calibrated is gone: the hybrid CPU/GPU leaf
> scheduler (`src/mojotrees/hybrid_leaf_scheduler.mojo`), its `HybridCosts`
> model, `bench/bench_hybrid_costs.mojo`, and the `MOJOTREES_HYBRID_*`
> variables were deleted on 2026-08-16, because the device-resident tree
> plane now beats the host path at every shape measured, all resolved under
> rule M0 (2.2x at 50,000 rows, 44 percent at 250,000, 24 percent at
> 1,000,000: `bench/results/session3_2026-08-16/RESULTS.md`). The
> coefficients below remain the only recorded per-launch, per-cell and
> per-row-slot cost model for this machine, and `phase_profile.mojo` still
> reports in the units they are stated in.


Experiment E1 of `docs/design/HYBRID_TRAINING.md` §9: the eight
`HybridCosts` coefficients, measured by `bench/bench_hybrid_costs.mojo`
(`pixi run bench-hybrid-costs`) at 500,000 rows x 50 features x 255 bins,
five interleaved trials, minimum reported. These are the numbers
`HybridCosts.apple_m4()` cites.

```
hybrid cost calibration: 500000 rows x 50 features x 255 bins, 5 trials of 50 reps

raw minima (nanoseconds):
  synchronize         20320
  launch              59153
  row readback        627080 ( 1953.125 KiB )
  convert             129060
  device build_leaf   2919800
  host replica build  33571200
  host float64 build  20052400
  host zero pass      160
  host partition      7742600

HybridCosts coefficients (nanos per thousand units):
  launch_nanos                = 59153
  sync_nanos                  = 20320
  transfer_nanos_per_kib      = 321
  device_nanos_per_krow_slot  = 107
  host_nanos_per_krow_slot    = 1343
  host_partition_nanos_per_krow = 15485
  host_zero_nanos_per_kcell   = 13
  convert_nanos_per_kcell     = 10122
```

Notes, in the order they changed the integration:

- **The host replica accumulates at 1343 ns per thousand (row, slot)
  pairs**, 12.6x the device's 107 but within 1.7x of the Float64 host
  builder (20.1ms vs 33.6ms on the calibration node). An earlier replica
  that re-quantized per (row, feature) ran at 12,154 — twenty times the
  Float64 builder — which is why `_accumulate_replica` hoists the
  quantization to once per row.
- **The device's fixed per-node cost on this shape is ~257us** (launch
  59us + download 48us + synchronize 20us + conversion 129us), which is
  what a host build of a small leaf avoids. The modelled host/device
  crossover at 50 active features sits near 4,000 rows.
- **The whole-permutation snapshot costs ~627us at 500k rows** through
  `GpuHistogramBuilder.snapshot_rows` (the `download_rows` path builds its
  list a row at a time and measured ~3x that). At 500k active rows no
  single leaf's saving covers a snapshot, so the scheduler's
  `DECLINE_SNAPSHOT_NOT_PAID` keeps hybrid scheduling inert there; it
  engages as the active-row count (dataset, or bag) drops toward ~150k and
  below, which is also where the GPU trainer's fixed costs hurt it most.
- **`host_partition_nanos_per_krow` = 15485** measured serially over the
  full 500k window. At that rate a host mirror of every split's partition
  (the maintenance in HYBRID_TRAINING.md §8.3 step 4) would cost tens of
  milliseconds per tree at this scale, so the integration in
  `grow_tree_gpu` retakes a fresh snapshot per host-elected split instead
  of maintaining one — the alternative §9 E6 names. Acceptance already
  requires each such leaf to pay for a full snapshot, so a host placement
  can never make a tree slower than the pure-device path.

## End-to-end effect

Interleaved off/replica arms in one process, same day, on the intersection
hybrid scheduling serves (host-staged gradients — here a bagged run):
20,000 rows x 50 features x 255 bins, `num_leaves` 127, 20 rounds,
`bagging_fraction` 0.8. Predictions bit-identical in every trial.

```
trial 0 off_s 1.553 replica_s 1.283 speedup 1.211
trial 1 off_s 1.555 replica_s 1.294 speedup 1.202
trial 2 off_s 1.557 replica_s 1.294 speedup 1.204
```

An unbagged built-in objective keeps its gradients device-resident, where
the scheduler correctly declines (`DECLINE_GRADIENTS_ON_DEVICE`); the same
shape measured there is flat (0.99x-1.02x), which is the no-op the default
must be.

Timings on this machine drift several-fold between time windows
(bench/README.md); coefficients from different runs should not be mixed.
