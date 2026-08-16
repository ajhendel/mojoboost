"""The multi-column label contract.

**This file is the finding, not the gradients.** Every training entry point in
mojotrees -- `boosting.train`, `boosting.fill_grad_hess`,
`objective.train_custom`, the C API, `bindings/_mojotrees.mojo`, the sklearn
wrapper -- takes the label as `List[Float64] target`, one number per row. Two
of the three CatBoost objectives this lane was sent for cannot be spelled in
that contract at all:

| objective     | label columns    | approx dimension | fits `List[Float64]`? |
|---------------|------------------|------------------|-----------------------|
| `Cox`         | 1, signed        | 1                | yes                   |
| `SurvivalAft` | 2 (lower, upper) | 1                | no                    |
| `MultiRMSE`   | T                | T                | no                    |

`TargetMatrix` is that contract widened by exactly one integer: a flat
row-major `List[Float64]` of `n_rows * n_targets` plus the `n_targets` that
says how to read it. A one-column `TargetMatrix` is the old contract and
`from_single` builds it without copying semantics changing, so nothing that
exists today has to move.

**Nothing upstream reaches this yet, and that is the honest status.** Widening
the ingestion path -- Python, C API, the marshaller, the sklearn `fit(X, y)`
signature -- is a change to files this lane does not own. Until that happens,
`SurvivalAft` and `MultiRMSE` are reachable from Mojo and from nowhere else.

Layout, and why row-major
-------------------------
`values[r * n_targets + t]`. CatBoost's own layout is the transpose,
`TConstArrayRef<TConstArrayRef<float>> target` indexed `[dim][row]`
(`catboost/libs/metrics/metric.h`), which suits its column-at-a-time metric
loops. Row-major suits ours: the `MultiRMSE` derivative reads all `T` targets
and all `T` raw scores at one row, a contiguous run of `T` in each, and it is
the layout `boosting.train_multiclass` already uses for `raw[r * n_classes +
k]`. One layout for both multi-output paths beats matching CatBoost's
transpose in a file no CatBoost code ever reads.

Determinism
-----------
Nothing here is order-dependent or parallel; a `TargetMatrix` is a validated
container. The row-major choice is what makes the *consumers* deterministic:
a per-row gradient loop over a row-major matrix splits into contiguous row
blocks with no cross-block reads, which is the property
`boosting.fill_grad_hess` already relies on to be bit-identical at every
`MOJOTREES_NUM_WORKERS`.
"""

from std.math import isfinite

# The unbounded sentinel in a `SurvivalAft` interval, in either column.
# CatBoost's, verbatim: `TSurvivalAftError::CalcDers` tests `target[1] == -1`
# and `target[0] == -1`, and `TSurvivalAftMetric::EvalSingleThread` maps `-1`
# to `+infinity` in both columns. It is a literal comparison against -1 in
# CatBoost too, not a "negative means missing" rule.
comptime AFT_UNBOUNDED = -1.0


struct TargetMatrix(Copyable, Movable):
    """`n_rows` by `n_targets` labels, flat and row-major.

    The invariant is `len(values) == n_rows * n_targets` and `n_targets >= 1`,
    established in `__init__` and never rechecked, so a consumer's inner loop
    can index without bounds arithmetic.
    """

    var values: List[Float64]
    var n_rows: Int
    var n_targets: Int

    def __init__(
        out self, var values: List[Float64], n_targets: Int
    ) raises:
        if n_targets < 1:
            raise Error("n_targets must be at least 1")
        if len(values) % n_targets != 0:
            raise Error(
                String(
                    "target values length ",
                    len(values),
                    " is not a multiple of n_targets ",
                    n_targets,
                )
            )
        self.n_targets = n_targets
        self.n_rows = len(values) // n_targets
        self.values = values^

    @staticmethod
    def from_single(target: List[Float64]) raises -> TargetMatrix:
        """The one-column contract as a `TargetMatrix`. Every existing caller
        is this case, which is why widening costs nothing that exists."""
        return TargetMatrix(target.copy(), 1)

    @always_inline
    def get(self, row: Int, t: Int) raises -> Float64:
        return self.values[row * self.n_targets + t]

    def column(self, t: Int) raises -> List[Float64]:
        """One target as a dense column. For metrics and for handing a single
        plane to a single-output grower; not on any per-row hot path."""
        if t < 0 or t >= self.n_targets:
            raise Error("target column out of range")
        var out = List[Float64](capacity=self.n_rows)
        for r in range(self.n_rows):
            out.append(self.values[r * self.n_targets + t])
        return out^

    def check_rows(self, n_rows: Int) raises:
        """The row count must match the design matrix."""
        if self.n_rows != n_rows:
            raise Error(
                String(
                    "target has ",
                    self.n_rows,
                    " rows, data has ",
                    n_rows,
                )
            )

    def check_finite(self) raises:
        """Every entry finite. `MultiRMSEWithMissingValues` is the one
        objective that wants NaN and it validates itself instead."""
        for i in range(len(self.values)):
            if not isfinite(self.values[i]):
                raise Error(
                    String(
                        "target entry ",
                        i,
                        " is not finite; only"
                        " MultiRMSEWithMissingValues accepts NaN",
                    )
                )

    def check_survival_aft(self) raises:
        """The two-column `SurvivalAft` contract, checked once per fit.

        CatBoost checks none of this and the failures are silent. Its branch
        order is `target[0] == target[1]` first, so a row of `(-1, -1)` is
        read as an *exact event at time -1* and goes straight into
        `FastLogf(-1)`. Both bounds unbounded is not a censoring pattern, it
        is a row that says nothing, and it is refused here.

        Positivity is the other one: `InverseMonotoneTransform` takes
        `log(target)`, so every bound that is not the sentinel has to be
        strictly positive.
        """
        if self.n_targets != 2:
            raise Error(
                String(
                    "SurvivalAft takes two label columns (lower, upper), got ",
                    self.n_targets,
                )
            )
        for r in range(self.n_rows):
            var lo = self.values[2 * r]
            var hi = self.values[2 * r + 1]
            if not isfinite(lo) or not isfinite(hi):
                raise Error(
                    String("SurvivalAft bound at row ", r, " is not finite")
                )
            if lo == AFT_UNBOUNDED and hi == AFT_UNBOUNDED:
                raise Error(
                    String(
                        "SurvivalAft row ",
                        r,
                        " has both bounds unbounded (-1, -1); CatBoost reads"
                        " that as an exact event at time -1 and takes its"
                        " logarithm",
                    )
                )
            if lo != AFT_UNBOUNDED and not (lo > 0.0):
                raise Error(
                    String(
                        "SurvivalAft lower bound at row ",
                        r,
                        " must be positive or the -1 sentinel",
                    )
                )
            if hi != AFT_UNBOUNDED and not (hi > 0.0):
                raise Error(
                    String(
                        "SurvivalAft upper bound at row ",
                        r,
                        " must be positive or the -1 sentinel",
                    )
                )
            if lo != AFT_UNBOUNDED and hi != AFT_UNBOUNDED and hi < lo:
                raise Error(
                    String(
                        "SurvivalAft upper bound at row ",
                        r,
                        " is below its lower bound",
                    )
                )


def survival_aft_targets(
    lower: List[Float64], upper: List[Float64]
) raises -> TargetMatrix:
    """Interleave two bound columns into the two-column `SurvivalAft` target.

    The convenience that keeps callers from having to know the row-major
    layout. `AFT_UNBOUNDED` (-1) in `upper` is right-censored, in `lower` is
    left-censored, and equal bounds are an exact event.
    """
    if len(lower) != len(upper):
        raise Error("lower and upper must have the same length")
    var flat = List[Float64](capacity=2 * len(lower))
    for r in range(len(lower)):
        flat.append(lower[r])
        flat.append(upper[r])
    var out = TargetMatrix(flat^, 2)
    out.check_survival_aft()
    return out^


def multi_targets(columns: List[List[Float64]]) raises -> TargetMatrix:
    """Interleave `T` equal-length target columns into a `TargetMatrix`."""
    if len(columns) < 1:
        raise Error("multi_targets needs at least one column")
    var n = len(columns[0])
    for t in range(len(columns)):
        if len(columns[t]) != n:
            raise Error("every target column must have the same length")
    var flat = List[Float64](capacity=n * len(columns))
    for r in range(n):
        for t in range(len(columns)):
            flat.append(columns[t][r])
    return TargetMatrix(flat^, len(columns))
