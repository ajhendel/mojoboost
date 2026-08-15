"""Deterministic row bagging.

LightGBM's `bagging_fraction` / `bagging_freq` / `bagging_seed`. Every
`freq` boosting rounds a new bag of rows is drawn, and the trees of the
rounds in between are grown on that same bag. Within tree growth the bag is
the dataset: histograms, row counts, `min_data_in_leaf`, leaf values, and
quantile/L1 leaf renewal see bagged rows only. Everything outside tree
growth stays on the full dataset, as in LightGBM:

- the base score is computed from every row, bagging or not
- after each tree, raw scores are updated for every row, in-bag or not, so
  out-of-bag rows carry correct gradients into later rounds

A bag is a list of row indices in ascending order, and it is the only thing
bagging materializes: gradients, hessians, and the binned matrix are never
copied or compacted, and tree growth consumes the index list it already
builds for its root node.

Sampling
--------
Each row is kept independently with probability `fraction` (Bernoulli, no
replacement), so a bag holds Binomial(n_rows, fraction) rows rather than
exactly fraction * n_rows. These are LightGBM's `BaggingHelper` semantics.
The draw is uniform over rows and ignores `sample_weight`: a heavy row is no
likelier to be drawn, it just carries its weight into the gradients of
whichever bag holds it. Zero-weight rows take part in the draw and
contribute nothing once drawn, exactly as they contribute nothing to a
full-data tree.

RNG
---
Counter-based splitmix64, not a sequential stream. The draw for row r of
bag b is

    stream = splitmix64(seed_bits ^ (b * GOLDEN))
    u(r)   = splitmix64(stream + r) >> 11, scaled by 2^-53   in [0, 1)

so a row's draw depends only on (seed, bag index, row index). Nothing
carries between rows, rounds, or backends, which is what lets the CPU and
GPU trainers sample identical rows and makes any one bag reproducible
without replaying the bags before it.

INTENTIONAL DIFFERENCES FROM LightGBM
-------------------------------------
- LightGBM draws from a 15-bit linear-congruential stream
  (`Random::NextFloat`, 32768 distinct values) seeded per 1024-row block so
  that threads stay reproducible. splitmix64 gives 53 bits of resolution and
  needs no blocking, so bags do not match LightGBM's row for row at equal
  seeds. The distribution and the resampling schedule do match.
- A draw that selects no rows falls back to the single row with the smallest
  draw value, so a bag is never empty and an unlucky draw cannot yield a
  degenerate round. LightGBM has no such guard.
"""

from .rng import GOLDEN, splitmix64, uniform

# LightGBM's bagging_seed default.
comptime DEFAULT_BAGGING_SEED = 3


@fieldwise_init
struct BaggingParams(Copyable, Movable):
    """Row bagging configuration.

    `fraction` in (0, 1] is the per-row keep probability, `freq` the number
    of rounds one bag is reused for (0 disables bagging), `seed` the RNG
    seed. Bagging is off when `freq <= 0` or `fraction >= 1`, which is how
    LightGBM resolves the same parameter combination.
    """

    var fraction: Float64
    var freq: Int
    var seed: Int

    @staticmethod
    def disabled() -> BaggingParams:
        """No bagging: every tree sees every row. LightGBM's defaults."""
        return BaggingParams(1.0, 0, DEFAULT_BAGGING_SEED)


def _stream(seed: Int, bag_index: Int) -> UInt64:
    """Start of the counter stream for one bag. The sign bit is masked off
    so negative seeds are accepted (as in LightGBM) without relying on
    signed-to-unsigned conversion."""
    return splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ (UInt64(bag_index) * GOLDEN)
    )


def bagging_enabled(params: BaggingParams) -> Bool:
    return params.freq > 0 and params.fraction < 1.0


def check_bagging(params: BaggingParams) raises:
    if not (params.fraction > 0.0 and params.fraction <= 1.0):
        raise Error("bagging_fraction must be in (0, 1]")
    if params.freq < 0:
        raise Error("bagging_freq must be nonnegative")


def sample_rows(
    params: BaggingParams, n_rows: Int, bag_index: Int, mut rows: List[Int]
) raises:
    """Draw bag number `bag_index` into `rows` (cleared first), ascending.

    Deterministic in (seed, bag_index, n_rows) alone: two callers agreeing
    on those three sample exactly the same rows, on whichever backend and
    whatever they sampled before.
    """
    if n_rows < 1:
        raise Error("n_rows must be positive")
    if bag_index < 0:
        raise Error("bag_index must be nonnegative")
    check_bagging(params)

    var stream = _stream(params.seed, bag_index)
    rows.clear()
    var min_draw = 2.0
    var min_row = 0
    for r in range(n_rows):
        var u = uniform(stream + UInt64(r))
        if u < params.fraction:
            rows.append(r)
        if u < min_draw:
            min_draw = u
            min_row = r
    # Never hand back an empty bag (see the module docstring).
    if len(rows) == 0:
        rows.append(min_row)


def refresh_bag(
    mut bag: List[Int],
    params: BaggingParams,
    n_rows: Int,
    iteration: Int,
) raises:
    """Redraw `bag` if `iteration` starts a new bag, leave it as it is if
    not.

    Bags change on the iterations where `iteration % freq == 0`, LightGBM's
    schedule, so the bag in force at iteration i is bag number `i // freq`.
    `bag` stays empty while bagging is disabled, and an empty bag means "all
    rows" everywhere downstream.
    """
    if not bagging_enabled(params):
        return
    if iteration % params.freq != 0:
        return
    sample_rows(params, n_rows, iteration // params.freq, bag)
