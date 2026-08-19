# GPU packed bins: wired, measured, and INERT

Date 2026-08-19, Apple M4, covertype 581,012 x 54 over 7 classes, 100 trees,
`max_depth=8`, `num_leaves=256`, arms interleaved in one run, three repeats.
Same shape as `GPU_ROW_COMPACTION_2026-08-19.md` so the two are comparable.

| arm | median s | min | max | multi_logloss |
| --- | --- | --- | --- | --- |
| mojotrees-gpu | 46.101 | 45.388 | 51.268 | 0.46406132921105236 |
| mojotrees-gpu-packed | 47.198 | 45.663 | 50.351 | 0.46406132921105236 |

`packed/baseline = 1.0238`, and **the ranges overlap**, so by
`PROFILE_PROTOCOL` M0 this is INDISTINGUISHABLE and not a resolved loss.

## The result is not "packing does not pay". It is "nothing reads the pack".

`GpuActiveRows.set_packed_bins` had no caller anywhere in `src/`, `tests/` or
`bench/`. It was wired into both `train_gpu` overloads behind
`MOJOTREES_GPU_PACKED_BINS` to make it measurable at all.

Two reach checks were run and **only the second one mattered**.

**Check 1, the env path. PASSED, and it was not enough.** `set_packed_bins`
refuses an all-width-8 table, so on 255-bin synthetic data the arm must raise.

    MOJOTREES_GPU_PACKED_BINS=1 bench-train-gpu 20000 20 reg 5
        -> raises "a packed bin layout at eight bits throughout is the
           feature-major matrix already on the device"
    (unset)
        -> trains, gpu_train_s median 0.771

That proves the switch reaches `set_packed_bins`. It proves nothing about
whether any kernel then reads what `set_packed_bins` produced.

**Check 2, the sabotage. FAILED, and it is the finding.** One feature's width
was narrowed by one bit. A short width silently truncates a bin id and a
truncated id is a legal id, so any consumer of the packed buffer must produce
a different model.

    MOJOTREES_GPU_PACKED_SABOTAGE=1 -> multi_logloss 0.46406132921105236
    baseline                        -> multi_logloss 0.46406132921105236

Bit-identical to seventeen digits under deliberate corruption. **Nothing on
the `train_gpu` path reads the packed buffer.** The arm allocates it, packs
into it, and every histogram reads the dense matrix as before, which is
exactly what a 2.4 percent slowdown with no accuracy movement looks like.

`tests/test_gpu_packed_bins.mojo` passes 20 of 20. The packing machinery is
correct in isolation. It has no consumer in training.

## What this costs the density hypothesis

Two of the three structural differences from CatBoost's compressed index have
now been measured and neither moved the number.

- Row compaction rearranges the cell: 1.535x slower leaf-wise, 0.757x
  symmetric. Both moved the BINNED MATRIX, which CatBoost never moves.
- Packed bins narrows the cell: unmeasurable, because no consumer exists.

So the 3.9 GB against their 480 MB is still unexplained, and the next question
is not another layout arm. It is **which kernel reads what**, because this
repository has now found six stages built and unreachable and two of them were
found by an arm measuring its own cost against nothing.

## Standing rule, restated because it just earned its keep again

An env-path reach check is NOT a reach check. Reaching the setter is not
reaching the consumer. Sabotage the data and watch the model move, or the
number is a fact about an allocation.
