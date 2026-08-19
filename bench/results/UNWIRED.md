# Built, tested, documented, and unreachable

**The question this file answers: why does implementing a competitor's
algorithm keep failing to make us faster?**

It is not that the algorithms are hard. We have implemented most of them.
It is that we build them as INDEPENDENT OPTIONS rather than as STAGES OF A
PIPELINE, so each one passes its own tests, reads the raw bin matrix on its
own, and never meets the others. LightGBM bins, then bundles, then builds its
multi-value bin OVER THE BUNDLES, and each stage consumes the last one's
output. We bin, and then offer a menu.

Found on 2026-08-19, four instances in one day. Each was fully implemented
with docstrings and tests.

| machinery | state | what it needed |
|---|---|---|
| **EFB** (`efb.mojo`) | implemented, correct, **default off** | measured 1.10-1.13x on covtype; default stays off for a real reason (`CLOSED_LANES.md`), so this one is a decision rather than a gap |
| **Row-major view over bundles** | implemented; **bundling silently destroyed it** | `bundle_dense` returned a fresh `BinnedMatrix` whose constructors set `row_stride = 0`, so every bundled fit degraded to feature-major and the two best CPU layouts could never compose. FIXED 2026-08-19 |
| **Compact bin addressing** (`build_packed_offsets`, `has_packed_offsets`) | implemented; **zero callers in `src/`, `bindings/` or `tests/`** | `feature_bins` / `bin_offset` are populated only as a side effect of `build_row_major`, and the only reader is the row-major kernel's PRIVATE accumulator. The main histogram is still rectangular |
| **GPU packed bin loads** (`packed_bin_loads`) | field exists, read in several places, **nothing anywhere sets it true** | consequently `plan_packed_window_for` has no callers and `WINDOW_STORAGE_NOT_BYTES` is an unreachable decline reason |

## Why the compact histogram is the expensive one

Covtype has **2,262 real bins** across 53 used features (LightGBM prints
exactly this: `Total Bins 2262`). Our rectangular histogram allocates
`n_features * n_bins` = 54 x 255 = **13,770 slots**, and each cell is 24
bytes across three planes (`_gh` pair plus a separate `_count`), so the
working set is about **330 KB**. LightGBM's is `bin_offset[f] + b` over 2,262
bins at 16 bytes with no count plane: about **37 KB, and L1-resident**.

That is a **9x working-set difference on the benchmark we lose**, and it is
the most credible remaining explanation for the part of the gap that bundling
and layout do not close. It is also a coherent explanation for why the
row-major arm lost at `row_stride = 54`: it scattered into 330 KB while the
feature-major control touched one 255-bin column at a time.

Two independent competitors carry TWO planes, not three: CatBoost's
`statCount = 2` for regression, LightGBM's `kHistEntrySize = 2 * sizeof(double)`
with counts recovered as `num_data / sum_hessian`. Ours is 1.5x both.

## The rule this implies

**A stage that cannot be reached from a default fit is not done.** A test
that constructs the machinery directly and asserts it works proves the
machinery works; it does not prove anything about the library. Before
claiming a competitor's algorithm is implemented, name the default fit that
reaches it, or say plainly that none does.

The reach discipline in `mojotrees-verify-reach-not-output` was learned for
CHANGES that do not execute. This is the same failure for FEATURES that do
not execute, and it is bigger, because a feature can sit unreachable for
months while its tests stay green.
