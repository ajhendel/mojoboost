"""Gradient-based One-Side Sampling (GOSS).

GOSS trains each boosting round on a subset of the rows: every row whose
gradient magnitude is large is kept, and a fixed fraction of the remaining
(small-gradient) rows is sampled. The sampled rows carry a compensation
multiplier on their gradient and hessian so that the histogram of the subset
still estimates the histogram of the full dataset.

LightGBM semantics, matched here:

- Row importance is `|grad * hess|`, not `|grad|` as in the GOSS paper.
  Sample weights are already folded into grad and hess, so weighted rows are
  ranked by their weighted contribution and zero-weight rows rank last.
- `top_k = int(n * top_rate)` (at least 1) rows are kept. The threshold is the
  `top_k`-th largest importance and the test is `importance >= threshold`, so
  ties can keep more than `top_k` rows.
- `other_k = int(n * other_rate)` small-gradient rows are sampled, in one
  forward pass, with the running probability `rest_need / rest_all`; that
  draws exactly `other_k` of them when no importance ties inflate the kept
  set.
- The multiplier is `(n - top_k) / other_k`, applied to both the gradient and
  the hessian of each sampled small-gradient row.
- Sampling is skipped for the first `int(1 / learning_rate)` rounds
  (LightGBM's GOSS warmup); `warmup_rounds` overrides that.
- The random stream is LightGBM's `Random`: a 32-bit LCG reseeded every round
  with `seed + round`, so a run is reproducible and independent of thread
  count.

Intentional differences from LightGBM:

- LightGBM partitions rows into blocks and gives each block its own random
  stream, so its selection depends on the block layout; mojoboost makes one
  serial pass over all rows with a single stream. Selected counts, the
  threshold rule, and the multiplier are the same, the individual rows drawn
  are not.
- `top_rate + other_rate <= 0` is rejected instead of silently training on a
  single row.
- If the running `rest_all` reaches zero while rows are still needed, the
  remaining small-gradient rows are taken outright rather than dividing by
  zero.
"""


@fieldwise_init
struct GossParams(Copyable, Movable):
    """GOSS configuration. Disabled by default: ordinary GBDT trains on
    every row."""

    var enabled: Bool
    var top_rate: Float64
    var other_rate: Float64
    var seed: Int
    var warmup_rounds: Int

    @staticmethod
    def disabled() -> GossParams:
        """Ordinary GBDT (the library default)."""
        return GossParams(False, 0.2, 0.1, 3, -1)

    @staticmethod
    def enable(
        top_rate: Float64 = 0.2,
        other_rate: Float64 = 0.1,
        seed: Int = 3,
        warmup_rounds: Int = -1,
    ) -> GossParams:
        """GOSS with LightGBM's defaults (`top_rate` 0.2, `other_rate` 0.1,
        `bagging_seed` 3). `warmup_rounds` of -1 means LightGBM's automatic
        `int(1 / learning_rate)` rounds of full-data training."""
        return GossParams(True, top_rate, other_rate, seed, warmup_rounds)

    def validate(self) raises:
        """Reject out-of-range rates. Comparisons are written so that NaN
        rates are rejected too."""
        if not self.enabled:
            return
        if not (self.top_rate >= 0.0 and self.top_rate <= 1.0):
            raise Error("goss top_rate must be in [0, 1]")
        if not (self.other_rate >= 0.0 and self.other_rate <= 1.0):
            raise Error("goss other_rate must be in [0, 1]")
        if not (self.top_rate + self.other_rate <= 1.0):
            raise Error("goss top_rate + other_rate must not exceed 1")
        if not (self.top_rate + self.other_rate > 0.0):
            raise Error("goss top_rate + other_rate must be positive")
        if self.seed < 0:
            raise Error("goss seed must be nonnegative")
        if self.warmup_rounds < -1:
            raise Error("goss warmup_rounds must be -1 (auto) or nonnegative")

    def warmup(self, learning_rate: Float64) -> Int:
        """Rounds trained on the full dataset before sampling starts."""
        if self.warmup_rounds >= 0:
            return self.warmup_rounds
        if learning_rate <= 0.0:
            return 0
        return Int(1.0 / learning_rate)

    def active(self, round: Int, learning_rate: Float64) -> Bool:
        return self.enabled and round >= self.warmup(learning_rate)


@fieldwise_init
struct GossSelection(Copyable, Movable):
    """One round's sampled rows.

    `rows` is ascending; `scale[i]` is the compensation multiplier for
    `rows[i]`, 1.0 for a kept high-gradient row and `multiplier` for a
    sampled low-gradient row. An empty `rows` means "train on every row":
    that is what GOSS returns while disabled or warming up.
    """

    var rows: List[Int]
    var scale: List[Float64]
    var multiplier: Float64
    var n_top: Int
    var n_other: Int

    @staticmethod
    def all_rows() -> GossSelection:
        """The no-sampling selection."""
        return GossSelection(List[Int](), List[Float64](), 1.0, 0, 0)


struct _GossRandom(Movable):
    """LightGBM's `Random`: a 32-bit linear congruential generator with the
    MSVC constants, returning the top 15 bits of the state. `next_float`
    therefore yields multiples of 1/32768 in [0, 1)."""

    var state: UInt32

    def __init__(out self, seed: Int):
        self.state = UInt32(seed & 0xFFFFFFFF)

    def next_int15(mut self) -> Int:
        self.state = self.state * 214013 + 2531011
        return Int((self.state >> 16) & 0x7FFF)

    def next_float(mut self) -> Float64:
        return Float64(self.next_int15()) / 32768.0


def _median3(a: Float64, b: Float64, c: Float64) -> Float64:
    if a < b:
        if b < c:
            return b
        return c if a < c else a
    if a < c:
        return a
    return c if b < c else b


def _select_kth_desc(mut values: List[Float64], k: Int) -> Float64:
    """The value at 0-based position `k` of `values` sorted descending,
    by quickselect with a median-of-three pivot. Deterministic (no random
    pivots) and destructive: `values` is left partially ordered. This is
    LightGBM's `ArrayArgs::ArgMaxAtK` role."""
    var lo = 0
    var hi = len(values) - 1
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var pivot = _median3(values[lo], values[mid], values[hi])
        var i = lo
        var j = hi
        # Hoare partition into (> pivot) then (< pivot). The pivot value is
        # present in [lo, hi], so neither scan can run past the ends.
        while i <= j:
            while values[i] > pivot:
                i += 1
            while values[j] < pivot:
                j -= 1
            if i <= j:
                var t = values[i]
                values[i] = values[j]
                values[j] = t
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            # Everything in (j, i) equals the pivot, including position k.
            return pivot
    return values[lo]


def goss_importance(
    grad: List[Float64], hess: List[Float64]
) raises -> List[Float64]:
    """Per-row sampling importance `|grad * hess|`, LightGBM's GOSS score."""
    if len(grad) != len(hess):
        raise Error("gradient/hessian length must match")
    var imp = List[Float64](capacity=len(grad))
    for r in range(len(grad)):
        imp.append(abs(grad[r] * hess[r]))
    return imp^


def goss_select(
    importance: List[Float64], params: GossParams, round: Int
) raises -> GossSelection:
    """Choose this round's rows from per-row importance. The RNG is reseeded
    with `seed + round`, so the selection depends only on the importance
    values, the rates, the seed, and the round index."""
    params.validate()
    var n = len(importance)
    if n == 0:
        raise Error("goss needs at least one row")

    var top_k = Int(Float64(n) * params.top_rate)
    if top_k < 1:
        top_k = 1
    if top_k > n:
        top_k = n
    var other_k = Int(Float64(n) * params.other_rate)
    if other_k > n - top_k:
        other_k = n - top_k

    var scratch = importance.copy()
    var threshold = _select_kth_desc(scratch, top_k - 1)

    var multiplier = 1.0
    if other_k > 0:
        multiplier = Float64(n - top_k) / Float64(other_k)

    var rng = _GossRandom(params.seed + round)
    var rows = List[Int](capacity=top_k + other_k)
    var scale = List[Float64](capacity=top_k + other_k)
    var n_top = 0
    var n_other = 0
    for r in range(n):
        if importance[r] >= threshold:
            rows.append(r)
            scale.append(1.0)
            n_top += 1
        else:
            var rest_need = other_k - n_other
            # Small-gradient rows still to come, this one included.
            var rest_all = (n - r) - (top_k - n_top)
            var prob: Float64
            if rest_all <= 0:
                prob = 1.0 if rest_need > 0 else 0.0
            else:
                prob = Float64(rest_need) / Float64(rest_all)
            # Drawn unconditionally so the stream does not depend on prob.
            if rng.next_float() < prob:
                rows.append(r)
                scale.append(multiplier)
                n_other += 1
    return GossSelection(rows^, scale^, multiplier, n_top, n_other)


def goss_select_round(
    params: GossParams,
    round: Int,
    learning_rate: Float64,
    grad: List[Float64],
    hess: List[Float64],
) raises -> GossSelection:
    """This round's selection for a single-output objective, or the
    all-rows selection while GOSS is disabled or warming up."""
    if not params.active(round, learning_rate):
        params.validate()
        return GossSelection.all_rows()
    return goss_select(goss_importance(grad, hess), params, round)


def apply_goss_scaling(
    selection: GossSelection, mut grad: List[Float64], mut hess: List[Float64]
):
    """Scale up the sampled low-gradient rows in place. Gradients are
    refilled from scratch every round, so mutating them here is safe."""
    for i in range(len(selection.rows)):
        var s = selection.scale[i]
        if s != 1.0:
            var r = selection.rows[i]
            grad[r] *= s
            hess[r] *= s


def goss_round(
    mut rows: List[Int],
    mut grad: List[Float64],
    mut hess: List[Float64],
    params: GossParams,
    round: Int,
    learning_rate: Float64,
) raises:
    """Run one round's sampling for a single-output objective: replace
    `rows` with the sampled rows and scale their gradients in place.

    `rows` is the same row list row bagging fills, and the two never both
    apply (the trainer rejects that combination), so leaving it untouched
    while GOSS is disabled or warming up keeps a bag a bag and an empty list
    meaning "every row".
    """
    if not params.active(round, learning_rate):
        params.validate()
        return
    var selection = goss_select(goss_importance(grad, hess), params, round)
    apply_goss_scaling(selection, grad, hess)
    rows = selection.rows.copy()
