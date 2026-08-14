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

Coverage: `train`, `train_with_valid`, `train_multiclass`,
`train_multiclass_with_valid`, `fit`, `fit_multiclass`, and the GPU
trainers `train_gpu` and `train_multiclass_gpu` all take a `goss`
parameter. `train_custom` and `train_ranker` do not, matching where row
bagging is wired today: the custom-objective trainers grow trees on every
row, and the ranker samples whole queries rather than rows.

RNG
---
Counter-based splitmix64, as in bagging.mojo, not a running stream. The draw
for row r of round i is

    stream = splitmix64(seed_bits ^ (i * GOLDEN))
    u(r)   = splitmix64(stream + r) >> 11, scaled by 2^-53   in [0, 1)

so a row's draw depends only on (seed, round, row): not on how many rows
before it were drawn, not on what earlier rounds drew, and not on thread
count. The forward pass still walks rows in order, because the acceptance
probability depends on how many rows have already been taken; only the
random numbers are independent of that walk.

Intentional differences from LightGBM:

- LightGBM partitions rows into blocks, gives each block its own 15-bit LCG
  seeded from `bagging_seed`, and draws sequentially within a block, so its
  sample depends on the block layout and its per-round streams are strongly
  correlated (consecutive round seeds start an LCG at nearby states).
  mojoboost draws from the counter-based stream above instead. The threshold
  rule, the sampled counts, and the multiplier are the same; the individual
  small-gradient rows drawn are not. This is the same trade bagging.mojo
  makes, for the same reason.
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


# Counter-based splitmix64, the same scheme bagging.mojo draws its bags
# from (and sampling.mojo its feature sets), duplicated here rather than
# shared so each sampler owns its stream.
comptime _GOLDEN = UInt64(0x9E3779B97F4A7C15)
comptime _TWO_POW_NEG_53 = 1.0 / 9007199254740992.0


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + _GOLDEN
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    """Uniform in [0, 1) with 53 significant bits, from a counter value."""
    return Float64(_splitmix64(counter) >> 11) * _TWO_POW_NEG_53


def _stream(seed: Int, round: Int) -> UInt64:
    """Start of the counter stream for one round's sample. The sign bit is
    masked off so the arithmetic never depends on signed-to-unsigned
    conversion."""
    return _splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ (UInt64(round) * _GOLDEN)
    )


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
    """Choose this round's rows from per-row importance. Every draw comes
    from the counter stream keyed by `(seed, round, row)`, so the selection
    depends only on the importance values, the rates, the seed, and the
    round index."""
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

    var stream = _stream(params.seed, round)
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
            # One draw per row, keyed by the row index, so a row's draw does
            # not depend on how many rows before it happened to be drawn.
            if _uniform(stream + UInt64(r)) < prob:
                rows.append(r)
                scale.append(multiplier)
                n_other += 1
    return GossSelection(rows^, scale^, multiplier, n_top, n_other)


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
