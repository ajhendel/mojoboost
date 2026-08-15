"""The central validation layer: one authoritative check per rule.

Every trainer, dataset constructor, serializer, and binding in mojotrees
validates its inputs. Before this module those checks were re-derived at
each site, which is how `params._validate` and `callback.check_resettable`
came to hold the same nine range checks with two sets of wording, how three
modules each grew their own "weights must be finite, nonnegative, and sum to
something positive" loop, and how a rule that exists in one place (a CSC
column's row indices must ascend) is absent two calls later (a loaded tree's
children are bounds checked but never required to sit after their parent).

This module is the single home for those rules. It takes primitives, never
mojotrees structs, and imports nothing from the package. That is deliberate:
`boosting`, `trainset`, `serialize`, `sparse`, and the bindings are all meant
to call in here, so a dependency in the other direction would close a cycle
the first time one of them did. A caller unpacks its struct at the call site
and passes lengths and buffers.

What lives here, and what does not
----------------------------------
Here: shape and emptiness, numeric domains (finiteness, sign, positive
sums), label and class-code ranges, ranking group arithmetic, compressed
sparse structure, categorical code widths, allocation arithmetic that must
not overflow, node/tree/model size ceilings for data read off disk, tree
topology strong enough to bound a traversal, iteration and depth counts, and
the two resource rules (a control code is one of the codes, a teardown
released what it acquired).

Not here, and deliberately so:

- **Objective semantics.** That a Poisson target is nonnegative, that a
  gamma target is positive, that `alpha` sits in (0, 1) for quantile: those
  belong to `boosting._check_objective`, which knows which objective is
  running. Duplicating them here would mean two answers to one question.
- **Serialization grammar.** That the token after `mapper` is an integer,
  that a v4 file carries a covers flag, that category codes ascend within a
  feature: `serialize.mojo` owns the format and checks it token by token.
  What this module adds to the load path is the part the grammar cannot
  express, which is how large a count may be before it is allocated and
  whether the resulting tree can be walked in bounded time.
- **Device and policy decisions.** `device_policy.mojo` chooses a backend;
  nothing here has an opinion about hardware.

Where a check needs a value the caller has and this module does not (an
objective code, a mapper), the caller keeps the check. Where a check needs
only numbers, it moved here.

Determinism
-----------
Every failure raises `Error` with a message that names the rule, the
offending index, and the offending value, in that order, and every scan runs
front to back, so the same bad input always produces the same message. No
check warns, clamps, or repairs. A caller that wants tolerance gets it by
not calling.

Cost
----
The scans are O(n) in what they look at and allocate nothing except where a
signature says it returns a list. `check_features_finite` is the one check
that is O(n_rows * n_features); it is meant to run once, at binning, not per
round. The rest are O(rows), O(nnz), or O(nodes).
"""

from std.math import isfinite, isnan


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------
#
# Two different jobs wear the same clothing here, so they are named apart.
#
# The `MAX_*` values below are refusal thresholds for counts that arrive from
# outside: a header field in a model file, a shape handed across the Python
# binding, an offset array from a producer of compressed matrices. A count
# past one of these is not a large problem, it is a corrupt or hostile one,
# and the point of refusing early is that the count is about to be used as an
# allocation size. None of them bounds what a legitimate run may do; every
# one sits orders of magnitude above the largest real workload.
#
# `MAX_ALLOC_ELEMS` is the arithmetic ceiling rather than a policy one. Index
# arithmetic in this package is `Int`, which is 64-bit and wraps on overflow,
# so a product that would wrap must be caught before it is computed. Keeping
# every element count under 2^46 leaves every product of two of them, and
# every such product plus an offset, comfortably inside Int64.

comptime MAX_ALLOC_ELEMS: Int = 1 << 46
"""Largest element count any allocation in this package may request.

At 8 bytes an element that is 512 TiB, so no real allocation approaches it;
what it buys is that `a * b` for two checked counts cannot wrap Int64, which
is what makes `checked_mul` a total function rather than a guess."""

comptime MAX_FEATURES: Int = 1 << 31
"""Largest feature count. LightGBM indexes features with `int` and
`categorical.mojo` requires category codes to survive `static_cast<int>`, so
a wider feature axis could not round-trip through either."""

comptime MAX_ROWS: Int = 1 << 44
"""Largest row count. Bounded well under `MAX_ALLOC_ELEMS` so that
`n_rows * n_features` for any accepted shape is still checkable."""

comptime MAX_BIN_COUNT: Int = 256
"""Largest bin count per feature. The binned matrix stores bin ids as UInt8
(see binning.mojo), so 256 is a representational limit, not a policy."""

comptime MAX_NNZ: Int = 1 << 44
"""Largest stored-entry count in a compressed sparse matrix."""

comptime MAX_TREE_NODES: Int = 1 << 24
"""Largest node count in one tree read from a file. A tree with more than
sixteen million nodes is not a tree anyone trained; `num_leaves` is capped
far below this by every grower."""

comptime MAX_MODEL_TREES: Int = 1 << 22
"""Largest tree count in one model file. `num_iterations * num_class` for
any real ensemble is several orders of magnitude below this."""

comptime MAX_MODEL_NODES: Int = 1 << 30
"""Largest total node count across a model's trees. Per-tree and per-model
ceilings are both needed: many small trees exhaust memory as surely as one
enormous one, and only the running total catches that."""

comptime MAX_ITERATIONS: Int = 1 << 24
"""Largest boosting round count a run may be asked for."""

comptime MAX_DEPTH_LIMIT: Int = 1 << 20
"""Largest `max_depth` a caller may state. Depth is bounded by node count in
practice; this rejects a value so large it can only be a unit error."""

comptime MAX_CLASSES: Int = 1 << 20
"""Largest class count. One tree per class per round is the multiclass cost
model, so the class count multiplies the whole ensemble."""

comptime MAX_CATEGORY_CODE: Int = 1 << 31
"""Exclusive upper bound on a raw category code, matching
`categorical._MAX_CATEGORY`: LightGBM reads codes through
`static_cast<int>`, so a code at or above 2^31 cannot round-trip."""

comptime MAX_RELEVANCE: Int = 30
"""Largest graded relevance label. This must stay equal to
`ranking.MAX_RELEVANCE_LABEL`, which is the value `ranking.label_gain`
tabulates against; `label_gain` is `2^label - 1`, and the table it reads
from is sized `MAX_RELEVANCE_LABEL + 1`, so a label past it indexes off the
end of the gains rather than merely losing precision."""


# ---------------------------------------------------------------------------
# Allocation arithmetic
# ---------------------------------------------------------------------------


def check_alloc(n_elems: Int, what: String) raises:
    """An element count that is about to become an allocation size.

    Nonnegative and at most `MAX_ALLOC_ELEMS`. Call this on any count that
    came from outside the process before handing it to `List(capacity=...)`
    or `resize`: a negative count is a wrapped subtraction and an enormous
    one is a corrupt header, and both are cheaper to name here than to
    discover as an allocation failure with no context.
    """
    if n_elems < 0:
        raise Error(
            what, " cannot be negative, got ", n_elems
        )
    if n_elems > MAX_ALLOC_ELEMS:
        raise Error(
            what,
            " of ",
            n_elems,
            " exceeds the allocation ceiling of ",
            MAX_ALLOC_ELEMS,
            " elements",
        )


def checked_mul(a: Int, b: Int, what: String) raises -> Int:
    """`a * b` for two nonnegative counts, or an error if the product would
    pass `MAX_ALLOC_ELEMS`.

    The check is a division rather than a multiply-then-compare, because the
    multiply is exactly the operation that would wrap: `n_rows * n_features`
    for a shape a hostile caller chose can land on a small positive number
    that then sizes a buffer the rest of the code indexes past.
    """
    if a < 0 or b < 0:
        raise Error(
            what, " cannot be negative, got ", a, " and ", b
        )
    if a == 0 or b == 0:
        return 0
    if a > MAX_ALLOC_ELEMS // b:
        raise Error(
            what,
            " of ",
            a,
            " by ",
            b,
            " overflows the allocation ceiling of ",
            MAX_ALLOC_ELEMS,
            " elements",
        )
    return a * b


def checked_add(a: Int, b: Int, what: String) raises -> Int:
    """`a + b` for two nonnegative counts, bounded the same way."""
    if a < 0 or b < 0:
        raise Error(
            what, " cannot be negative, got ", a, " and ", b
        )
    if a > MAX_ALLOC_ELEMS - b:
        raise Error(
            what,
            " of ",
            a,
            " plus ",
            b,
            " overflows the allocation ceiling of ",
            MAX_ALLOC_ELEMS,
            " elements",
        )
    return a + b


def checked_cells(n_rows: Int, n_features: Int) raises -> Int:
    """`n_rows * n_features` for a shape that has already passed
    `check_shape`, as a checked product. This is the one allocation size
    every dense path derives, so it gets a name."""
    return checked_mul(n_rows, n_features, "matrix cell count")


# ---------------------------------------------------------------------------
# Dimensions and emptiness
# ---------------------------------------------------------------------------


def check_shape(n_rows: Int, n_features: Int) raises:
    """A feature matrix must have at least one row and one feature, and both
    axes must stay inside their ceilings.

    Emptiness is rejected rather than accommodated. A zero-row matrix has no
    quantiles to fit, a zero-feature matrix has no split to find, and both
    reach the histogram builders as loops that run zero times and return a
    model that predicts the base score with no explanation of why.
    """
    if n_rows < 1:
        raise Error("a feature matrix needs at least one row, got ", n_rows)
    if n_features < 1:
        raise Error(
            "a feature matrix needs at least one feature, got ", n_features
        )
    if n_rows > MAX_ROWS:
        raise Error(
            "row count of ", n_rows, " exceeds the limit of ", MAX_ROWS
        )
    if n_features > MAX_FEATURES:
        raise Error(
            "feature count of ",
            n_features,
            " exceeds the limit of ",
            MAX_FEATURES,
        )
    _ = checked_cells(n_rows, n_features)


def check_dense_matrix(n_values: Int, n_rows: Int, n_features: Int) raises:
    """A column-major dense buffer must hold exactly `n_rows * n_features`
    values for the shape it claims."""
    check_shape(n_rows, n_features)
    var cells = checked_cells(n_rows, n_features)
    if n_values != cells:
        raise Error(
            "features length must equal n_rows * n_features: got ",
            n_values,
            " values for ",
            n_rows,
            " by ",
            n_features,
            ", which needs ",
            cells,
        )


def check_column_length(n_values: Int, n_rows: Int, name: String) raises:
    """An optional per-row column is either absent (length 0) or has exactly
    one entry per row. Every dataset field except the matrix itself follows
    this rule, so they all check it the same way and report the same two
    numbers."""
    if n_values == 0:
        return
    if n_values != n_rows:
        raise Error(
            name,
            " must have one entry per row: got ",
            n_values,
            " for ",
            n_rows,
            " rows",
        )


def check_required_length(n_values: Int, n_rows: Int, name: String) raises:
    """A per-row column that is not optional. Same message shape as
    `check_column_length`, with absence rejected."""
    if n_values != n_rows:
        raise Error(
            name,
            " must have one entry per row: got ",
            n_values,
            " for ",
            n_rows,
            " rows",
        )


def check_row_index(r: Int, n_rows: Int) raises:
    if r < 0 or r >= n_rows:
        raise Error(
            "row index ", r, " out of range for ", n_rows, " rows"
        )


def check_feature_index(f: Int, n_features: Int) raises:
    if f < 0 or f >= n_features:
        raise Error(
            "feature index ", f, " out of range for ", n_features, " features"
        )


def check_ascending_rows(rows: List[Int], n_rows: Int) raises:
    """A row selection must be nonempty, in range, and strictly ascending.

    Ascending is not tidiness. A CSC column stores its row indices in
    ascending order, so a selection that reordered or repeated rows would
    either produce a matrix that violates that invariant or need a sort per
    column to repair it. Rejecting the selection says so once, where the
    caller can fix it, instead of making every later reader pay.
    """
    if len(rows) < 1:
        raise Error("a row selection needs at least one row")
    if len(rows) > n_rows:
        raise Error(
            "a strictly ascending selection of ",
            len(rows),
            " rows cannot come from ",
            n_rows,
            " rows",
        )
    for i in range(len(rows)):
        if rows[i] < 0 or rows[i] >= n_rows:
            raise Error(
                "selected row index ",
                rows[i],
                " at position ",
                i,
                " out of range for ",
                n_rows,
                " rows",
            )
        if i > 0 and rows[i] <= rows[i - 1]:
            raise Error(
                "selected rows must be strictly ascending: position ",
                i,
                " is ",
                rows[i],
                " after ",
                rows[i - 1],
            )


# ---------------------------------------------------------------------------
# Numeric domains
# ---------------------------------------------------------------------------
#
# Three domains, kept apart because the rules genuinely differ:
#
#   feature values   NaN allowed (it is the missing marker), infinity is not
#   labels, scores   neither NaN nor infinity
#   weights          neither, plus nonnegative, plus a positive total
#
# Collapsing them into one "must be finite" helper is what produced the
# earlier drift, where a feature matrix and a label vector were checked by
# the same predicate and one of the two was wrong.


def check_features_finite(
    values: List[Float64], n_rows: Int, n_features: Int
) raises:
    """Feature values may be NaN and may not be infinite.

    NaN is mojotrees's missing-value marker: the binner excludes it from the
    quantiles and reserves a bin for it (see binning.mojo). An infinity is
    neither a value nor a marker. It cannot sit between two quantile edges,
    the midpoint of an infinity and anything is not a number, and `_avoid_inf`
    exists precisely because an edge that reached infinity broke the
    comparison the transform depends on.

    Column-major order, so the report names the feature and the row the way
    the caller stored them. O(n_rows * n_features); this is the one check
    meant to run once, at binning.
    """
    check_dense_matrix(len(values), n_rows, n_features)
    for f in range(n_features):
        var base = f * n_rows
        for r in range(n_rows):
            var v = values[base + r]
            if not isfinite(v) and not isnan(v):
                raise Error(
                    "feature values must not be infinite (NaN is allowed and"
                    " means missing): feature ",
                    f,
                    " row ",
                    r,
                    " is ",
                    v,
                )


def check_sparse_values_finite(values: List[Float64]) raises:
    """The stored values of a compressed matrix, under the same rule as a
    dense matrix. Absent entries are the numerical value 0.0 and need no
    check; a stored NaN is missing and is allowed."""
    for i in range(len(values)):
        var v = values[i]
        if not isfinite(v) and not isnan(v):
            raise Error(
                "stored feature values must not be infinite (NaN is allowed"
                " and means missing): entry ",
                i,
                " is ",
                v,
            )


def check_finite_vector(values: List[Float64], name: String) raises:
    """A vector that must be finite throughout, with neither NaN nor
    infinity. Labels, init scores, raw scores, base scores, and metric
    outputs all live under this rule: a missing label has no defined
    contribution to a loss, and neither does an infinite one."""
    for i in range(len(values)):
        var v = values[i]
        if not isfinite(v):
            raise Error(
                name,
                " must be finite: entry ",
                i,
                " is ",
                v,
            )


def check_labels_finite(label: List[Float64], n_rows: Int) raises:
    """The label column: one finite value per row. Objective-specific range
    rules (Poisson nonnegative, gamma positive, cross entropy in [0, 1]) stay
    with the objective, which knows which one is running."""
    check_required_length(len(label), n_rows, "label")
    check_finite_vector(label, "label")


def check_weights(weight: List[Float64], n_rows: Int) raises -> Float64:
    """The sample-weight contract, and the total weight it implies.

    An empty vector means unweighted, and the total is the row count, which
    is what an unweighted mean divides by. Otherwise: one entry per row,
    every entry finite and nonnegative, and a positive sum.

    A zero weight is allowed and drops that row, exactly as LightGBM's does.
    An all-zero vector is not: it drops every row, so the run has nothing to
    fit and every weighted mean divides by zero. The returned total is the
    denominator every weighted statistic in the package needs, so returning
    it here is what keeps the sum from being recomputed with a different
    accumulation order somewhere else.
    """
    if len(weight) == 0:
        return Float64(n_rows)
    check_required_length(len(weight), n_rows, "sample_weight")
    var total = 0.0
    for r in range(n_rows):
        var w = weight[r]
        if not isfinite(w):
            raise Error("sample_weight must be finite: row ", r, " is ", w)
        if w < 0.0:
            raise Error(
                "sample_weight must be nonnegative: row ", r, " is ", w
            )
        total += w
    if total <= 0.0:
        raise Error(
            "sample_weight must have a positive sum, got ",
            total,
            "; an all-zero vector drops every row from training",
        )
    return total


def check_gradient_pair(
    grad: List[Float64], hess: List[Float64], n_rows: Int
) raises -> Float64:
    """A custom objective's gradients and hessians, and the total hessian.

    One of each per row, every value finite, every hessian nonnegative. The
    sign rule is not stylistic: a leaf value is `-G / (H + lambda)`, so a
    negative hessian can drive the denominator through zero and produce a
    leaf of arbitrary magnitude from a well-behaved gradient.

    A zero *total* is allowed here and is not an error. A converged custom
    objective can legitimately return all-zero curvature for a round, and the
    right answer to that is a root-only tree, not a raise. The total is
    returned rather than judged, so a caller that does need it positive says
    so with `check_positive_hessian_total`, and the sum is still accumulated
    only once and in one order.
    """
    check_required_length(len(grad), n_rows, "gradients")
    check_required_length(len(hess), n_rows, "hessians")
    var total = 0.0
    for r in range(n_rows):
        var g = grad[r]
        if not isfinite(g):
            raise Error("gradients must be finite: row ", r, " is ", g)
        var h = hess[r]
        if not isfinite(h):
            raise Error("hessians must be finite: row ", r, " is ", h)
        if h < 0.0:
            raise Error(
                "hessians must be nonnegative: row ",
                r,
                " is ",
                h,
                "; a negative hessian can drive the leaf denominator through"
                " zero",
            )
        total += h
    return total


def check_positive_hessian_total(total: Float64, where: String) raises:
    """The opt-in half of `check_gradient_pair`, for a caller whose next step
    divides by the total rather than by a per-node sum."""
    if not isfinite(total):
        raise Error(
            "hessian total at ", where, " must be finite, got ", total
        )
    if total <= 0.0:
        raise Error(
            "hessian total at ",
            where,
            " must be positive, got ",
            total,
            "; with a zero total no leaf value is defined",
        )


def check_positive_scalar(value: Float64, name: String) raises:
    if not isfinite(value):
        raise Error(name, " must be finite, got ", value)
    if value <= 0.0:
        raise Error(name, " must be positive, got ", value)


def check_nonnegative_scalar(value: Float64, name: String) raises:
    if not isfinite(value):
        raise Error(name, " must be finite, got ", value)
    if value < 0.0:
        raise Error(name, " must be nonnegative, got ", value)


def check_unit_fraction(value: Float64, name: String) raises:
    """A subsampling fraction, in (0, 1]. Zero selects nothing, which is a
    configuration that trains no tree rather than a small one."""
    if not isfinite(value):
        raise Error(name, " must be finite, got ", value)
    if value <= 0.0 or value > 1.0:
        raise Error(name, " must be in (0, 1], got ", value)


# ---------------------------------------------------------------------------
# Column quality: constant and all-missing features
# ---------------------------------------------------------------------------


@fieldwise_init
struct ColumnReport(Copyable, Movable):
    """What one scan of a feature column found.

    `n_missing` counts NaN. `n_finite` counts the rest. `is_constant` is true
    when every finite value is bit-equal to the first one, which includes the
    case of a single finite value and the case of none.

    A constant or all-missing column is not an error. LightGBM keeps such
    features, fits them zero bin edges, and never splits on them, and so does
    mojotrees. What it is, is the explanation a caller needs when a feature
    reports zero importance, so the scan returns its finding rather than
    raising on it. `check_columns_usable` is the one place that turns a
    finding into an error, and only for the case where every column is
    unusable and the run therefore cannot split on anything at all.
    """

    var n_missing: Int
    var n_finite: Int
    var is_constant: Bool

    def is_all_missing(self) -> Bool:
        return self.n_finite == 0

    def is_usable(self) -> Bool:
        """Whether a split could ever be found on this column: at least two
        distinct finite values."""
        return self.n_finite > 0 and not self.is_constant


def scan_column(
    values: List[Float64], n_rows: Int, feature: Int
) raises -> ColumnReport:
    """Scan one column of a column-major matrix. O(n_rows), no allocation.

    Infinities are not tolerated here either, so a caller that scans columns
    without having run `check_features_finite` still gets the same refusal
    with the same message.

    The slice bound is checked rather than assumed, because this is the one
    function here that reads a column without having been told the feature
    count and so cannot derive the bound from the shape.
    """
    if feature < 0:
        raise Error("feature index cannot be negative, got ", feature)
    var base = checked_mul(feature, n_rows, "column offset")
    if base + n_rows > len(values):
        raise Error(
            "feature ",
            feature,
            " needs values [",
            base,
            ", ",
            base + n_rows,
            ") but the matrix holds only ",
            len(values),
        )
    var n_missing = 0
    var n_finite = 0
    var first = 0.0
    var constant = True
    for r in range(n_rows):
        var v = values[base + r]
        if isnan(v):
            n_missing += 1
            continue
        if not isfinite(v):
            raise Error(
                "feature values must not be infinite (NaN is allowed and"
                " means missing): feature ",
                feature,
                " row ",
                r,
                " is ",
                v,
            )
        if n_finite == 0:
            first = v
        elif v != first:
            constant = False
        n_finite += 1
    return ColumnReport(n_missing, n_finite, constant)


def check_columns_usable(
    values: List[Float64], n_rows: Int, n_features: Int
) raises:
    """At least one feature must offer a split.

    Every individual column may be constant or entirely missing without
    complaint. All of them at once may not: there is no split anywhere in the
    matrix, every tree is a root, and the fitted model is the base score
    dressed as an ensemble. Reporting that at binning time costs one pass and
    saves the caller from reading a run of identical predictions and
    inferring the cause.
    """
    check_dense_matrix(len(values), n_rows, n_features)
    for f in range(n_features):
        var report = scan_column(values, n_rows, f)
        if report.is_usable():
            return
    raise Error(
        "no feature can be split on: all ",
        n_features,
        " columns are constant or entirely missing, so every tree would be a"
        " single leaf",
    )


# ---------------------------------------------------------------------------
# Labels, class codes, and class counts
# ---------------------------------------------------------------------------


def check_class_count(n_classes: Int) raises:
    """A softmax problem needs at least two classes, and the class count
    multiplies the ensemble (one tree per class per round), so it is capped
    like any other allocation driver."""
    if n_classes < 2:
        raise Error("n_classes must be at least 2, got ", n_classes)
    if n_classes > MAX_CLASSES:
        raise Error(
            "n_classes of ",
            n_classes,
            " exceeds the limit of ",
            MAX_CLASSES,
        )


def check_class_codes(
    label: List[Float64], n_classes: Int
) raises -> List[Int]:
    """Class codes from a float64 label column, which is how labels reach a
    dataset.

    Whole numbers in `[0, n_classes)` only. A fractional code is rejected
    rather than truncated: truncation would silently move a row into a
    neighboring class, and the caller who wrote 1.5 meant something this
    layer cannot guess. NaN and infinity fail the whole-number test first, so
    they are reported as the fractional value they are not.
    """
    check_class_count(n_classes)
    var out = List[Int](capacity=len(label))
    for r in range(len(label)):
        var v = label[r]
        if not isfinite(v) or v != Float64(Int(v)):
            raise Error(
                "class labels must be whole numbers: row ", r, " is ", v
            )
        var code = Int(v)
        if code < 0 or code >= n_classes:
            raise Error(
                "class label out of range: row ",
                r,
                " is ",
                code,
                ", which is outside [0, ",
                n_classes,
                ")",
            )
        out.append(code)
    return out^


def check_class_code_range(
    codes: List[Int], n_classes: Int, name: String = "class label"
) raises:
    """Integer class codes already decoded (the softmax trainers take
    `List[Int]`): every code in `[0, n_classes)`, reported the way
    `check_class_codes` reports the same fault on a float column. `name`
    lets a validation column say which column it is."""
    for r in range(len(codes)):
        var code = codes[r]
        if code < 0 or code >= n_classes:
            raise Error(
                name,
                " out of range: row ",
                r,
                " is ",
                code,
                ", which is outside [0, ",
                n_classes,
                ")",
            )


def check_classes_present(codes: List[Int], n_classes: Int) raises:
    """Every class must appear at least once.

    A class with no rows has no gradient anywhere, so its trees are grown
    from an all-zero objective and every one of them is a single leaf at the
    base score. The model then reports a class it cannot predict. LightGBM
    accepts this; mojotrees names the missing class instead, because the
    common cause is a label encoding that dropped a level.
    """
    check_class_count(n_classes)
    var seen = List[Bool](capacity=n_classes)
    seen.resize(n_classes, False)
    for r in range(len(codes)):
        var c = codes[r]
        if c < 0 or c >= n_classes:
            raise Error(
                "class label out of range: row ",
                r,
                " is ",
                c,
                ", which is outside [0, ",
                n_classes,
                ")",
            )
        seen[c] = True
    for k in range(n_classes):
        if not seen[k]:
            raise Error(
                "class ",
                k,
                " has no rows; a class with no gradient trains to a single"
                " leaf at the base score",
            )


def check_relevance_labels(label: List[Float64]) raises -> List[Int]:
    """Graded relevances from a float64 label column, for ranking.

    Whole numbers in `[0, MAX_RELEVANCE]`. The upper bound is where
    `label_gain`, which is `2^label - 1` evaluated in Float64, stops being
    exact; a larger label would contribute a gain that no longer matches the
    one an exact evaluation gives, and NDCG would drift by an amount that
    depends on the label rather than on the ranking.
    """
    var out = List[Int](capacity=len(label))
    for r in range(len(label)):
        var v = label[r]
        if not isfinite(v) or v != Float64(Int(v)):
            raise Error(
                "relevance labels must be whole numbers: row ", r, " is ", v
            )
        var g = Int(v)
        if g < 0:
            raise Error(
                "relevance labels must be nonnegative: row ", r, " is ", g
            )
        if g > MAX_RELEVANCE:
            raise Error(
                "relevance labels must be at most ",
                MAX_RELEVANCE,
                ": row ",
                r,
                " is ",
                g,
            )
        out.append(g)
    return out^


# ---------------------------------------------------------------------------
# Ranking groups
# ---------------------------------------------------------------------------


def check_group_counts(counts: List[Int], n_rows: Int) raises -> Int:
    """LightGBM's `group` array: one row count per query, in row order.

    Nonempty, every count positive, and the total exactly `n_rows`. The
    running total is accumulated through `checked_add`, so a hostile array of
    large counts is reported as an overflow at the entry that caused it
    rather than wrapping into a total that happens to match `n_rows`.

    Returns the query count, which is what a boundary array's length is
    derived from.
    """
    if len(counts) == 0:
        raise Error("group must contain at least one query")
    if len(counts) > n_rows:
        raise Error(
            "group has ",
            len(counts),
            " queries but the data has only ",
            n_rows,
            " rows, and every query needs at least one row",
        )
    var total = 0
    for q in range(len(counts)):
        if counts[q] < 1:
            raise Error(
                "group counts must be positive: query ",
                q,
                " has ",
                counts[q],
                " rows",
            )
        total = checked_add(total, counts[q], "group row total")
    if total != n_rows:
        raise Error(
            "group counts must sum to n_rows: they sum to ",
            total,
            " for ",
            n_rows,
            " rows",
        )
    return len(counts)


def check_group_boundaries(starts: List[Int], n_rows: Int) raises:
    """The same contract expressed as boundaries rather than counts: at least
    two entries, starting at 0, strictly increasing, ending at `n_rows`."""
    if len(starts) < 2:
        raise Error("group must contain at least one query")
    if starts[0] != 0:
        raise Error("group boundaries must start at row 0, got ", starts[0])
    for q in range(len(starts) - 1):
        if starts[q + 1] <= starts[q]:
            raise Error(
                "group boundaries must be strictly increasing: boundary ",
                q + 1,
                " is ",
                starts[q + 1],
                " after ",
                starts[q],
            )
    var last = starts[len(starts) - 1]
    if last != n_rows:
        raise Error(
            "group boundaries must end at n_rows: they end at ",
            last,
            " for ",
            n_rows,
            " rows",
        )


# ---------------------------------------------------------------------------
# Compressed sparse structure
# ---------------------------------------------------------------------------


def check_compressed(
    offsets: List[Int],
    indices: List[Int],
    n_values: Int,
    n_outer: Int,
    n_inner: Int,
    kind: String,
    outer: String,
    inner: String,
) raises:
    """CSC and CSR structural validation, for both orientations.

    `outer` is the compressed axis (columns for CSC, rows for CSR) and
    `inner` the indexed one. Every failure a malformed producer can hand over
    is caught before any index is dereferenced: wrong dimensions, a
    wrong-length or non-monotone offset array, an offset array that does not
    start at 0 or end at nnz, mismatched index and value arrays, an
    out-of-range index, and unsorted or duplicated indices within one outer
    slice.

    Strictly ascending inner indices is SciPy's canonical form
    (`sum_duplicates()` then `sort_indices()`), and it also rules out
    duplicate entries, which is what lets `lookup` be a binary search and
    lets a row subset stay canonical without a re-sort.
    """
    if n_outer < 1 or n_inner < 1:
        raise Error(
            kind,
            " matrix must have positive dimensions, got ",
            n_outer,
            " ",
            outer,
            "s by ",
            n_inner,
            " ",
            inner,
            "s",
        )
    check_alloc(n_values, kind + " stored-entry count")
    if n_values > MAX_NNZ:
        raise Error(
            kind,
            " matrix has ",
            n_values,
            " stored entries, above the limit of ",
            MAX_NNZ,
        )
    if len(offsets) != n_outer + 1:
        raise Error(
            kind,
            " offsets must have length n_",
            outer,
            "s + 1: got ",
            len(offsets),
            " for ",
            n_outer,
            " ",
            outer,
            "s",
        )
    if offsets[0] != 0:
        raise Error(kind, " offsets must start at 0, got ", offsets[0])
    if len(indices) != n_values:
        raise Error(
            kind,
            " indices and values must have equal length: got ",
            len(indices),
            " indices for ",
            n_values,
            " values",
        )
    if offsets[n_outer] != n_values:
        raise Error(
            kind,
            " offsets must end at nnz: they end at ",
            offsets[n_outer],
            " for ",
            n_values,
            " stored entries",
        )
    for k in range(n_outer):
        var lo = offsets[k]
        var hi = offsets[k + 1]
        if hi < lo:
            raise Error(
                kind,
                " offsets must be non-decreasing: ",
                outer,
                " ",
                k,
                " runs from ",
                lo,
                " to ",
                hi,
            )
        for i in range(lo, hi):
            if indices[i] < 0 or indices[i] >= n_inner:
                raise Error(
                    kind,
                    " ",
                    inner,
                    " index out of range: entry ",
                    i,
                    " is ",
                    indices[i],
                    ", which is outside [0, ",
                    n_inner,
                    ")",
                )
            if i > lo and indices[i] <= indices[i - 1]:
                raise Error(
                    kind,
                    " ",
                    inner,
                    " indices must be strictly ascending within each ",
                    outer,
                    ": entry ",
                    i,
                    " is ",
                    indices[i],
                    " after ",
                    indices[i - 1],
                )


def check_csc(
    col_offsets: List[Int],
    row_index: List[Int],
    n_values: Int,
    n_rows: Int,
    n_features: Int,
) raises:
    """CSC in the orientation mojotrees stores it: features compressed, rows
    indexed."""
    check_compressed(
        col_offsets,
        row_index,
        n_values,
        n_features,
        n_rows,
        "CSC",
        "column",
        "row",
    )


def check_csr(
    row_offsets: List[Int],
    col_index: List[Int],
    n_values: Int,
    n_rows: Int,
    n_features: Int,
) raises:
    """CSR in the orientation a caller hands it over: rows compressed,
    features indexed."""
    check_compressed(
        row_offsets,
        col_index,
        n_values,
        n_rows,
        n_features,
        "CSR",
        "row",
        "column",
    )


# ---------------------------------------------------------------------------
# Categorical declarations and codes
# ---------------------------------------------------------------------------


def check_categorical_features(
    features: List[Int], n_features: Int
) raises:
    """The categorical feature declaration: in range and without repeats.

    A repeat is rejected rather than deduplicated. Two entries for one
    feature mean the caller believes it declared something it did not, and
    the second entry is the one that would be silently dropped. O(k^2) in the
    declaration length, which is the feature count at worst and is fitted
    once per dataset.
    """
    for i in range(len(features)):
        var f = features[i]
        if f < 0 or f >= n_features:
            raise Error(
                "categorical feature index ",
                f,
                " at position ",
                i,
                " out of range for ",
                n_features,
                " features",
            )
        for j in range(i):
            if features[j] == f:
                raise Error(
                    "categorical feature index ",
                    f,
                    " listed twice, at positions ",
                    j,
                    " and ",
                    i,
                )


def check_category_code(value: Float64, feature: Int, row: Int) raises -> Int:
    """One raw category code, as a whole number below `MAX_CATEGORY_CODE`.

    Negative values and NaN are missing rather than codes and are the
    caller's to interpret before getting here, so this rejects them: it is
    the checked path, and a caller that wants the missing convention uses
    `categorical.bin_of`, which maps them to `UNKNOWN_BIN`. A fractional
    nonnegative value is rejected because rounding it would merge two
    categories without saying so.
    """
    if isnan(value):
        raise Error(
            "category code is NaN: feature ",
            feature,
            " row ",
            row,
            "; a missing category is routed by bin_of, not coded here",
        )
    if not isfinite(value):
        raise Error(
            "category code must be finite: feature ",
            feature,
            " row ",
            row,
            " is ",
            value,
        )
    if value < 0.0:
        raise Error(
            "category code must be nonnegative: feature ",
            feature,
            " row ",
            row,
            " is ",
            value,
        )
    if value >= Float64(MAX_CATEGORY_CODE):
        raise Error(
            "category code must be below ",
            MAX_CATEGORY_CODE,
            " to survive LightGBM's int cast: feature ",
            feature,
            " row ",
            row,
            " is ",
            value,
        )
    if value != Float64(Int(value)):
        raise Error(
            "category codes must be whole numbers: feature ",
            feature,
            " row ",
            row,
            " is ",
            value,
        )
    return Int(value)


def check_max_bin(max_bin: Int) raises:
    """The binned matrix stores bin ids as UInt8, so the per-feature bin
    count is at most 256, and a feature needs at least two bins to have an
    edge between them."""
    if max_bin < 2:
        raise Error("max_bin must be at least 2, got ", max_bin)
    if max_bin > MAX_BIN_COUNT:
        raise Error(
            "max_bin must be at most ",
            MAX_BIN_COUNT,
            " because bin ids are stored as UInt8, got ",
            max_bin,
        )


# ---------------------------------------------------------------------------
# Model, tree, and node size limits
# ---------------------------------------------------------------------------
#
# These are for counts read off disk, before the count is used as an
# allocation size. `serialize.mojo` owns whether a token is a well-formed
# integer; this owns whether the integer is a plausible size. The two are
# separate because the grammar is satisfied by any integer at all, including
# one that sizes a list larger than the machine.


def check_mapper_header(n_features: Int, n_bins: Int, n_edges: Int) raises:
    """A bin mapper's header, as read from a model file.

    The edge count is bounded by the shape rather than by a constant: a
    mapper fits at most `n_bins - 1` edges per feature, so any larger count
    describes a mapper no binning could have produced.
    """
    if n_features < 1 or n_features > MAX_FEATURES:
        raise Error(
            "corrupt mapper header: feature count of ",
            n_features,
            " is outside [1, ",
            MAX_FEATURES,
            "]",
        )
    check_max_bin(n_bins)
    check_alloc(n_edges, "mapper edge count")
    var ceiling = checked_mul(n_features, n_bins - 1, "mapper edge ceiling")
    if n_edges > ceiling:
        raise Error(
            "corrupt mapper header: ",
            n_edges,
            " edges for ",
            n_features,
            " features at ",
            n_bins,
            " bins, which allows at most ",
            ceiling,
        )


def check_tree_count(n_trees: Int) raises:
    if n_trees < 0:
        raise Error("corrupt tree count: ", n_trees)
    if n_trees > MAX_MODEL_TREES:
        raise Error(
            "model declares ",
            n_trees,
            " trees, above the limit of ",
            MAX_MODEL_TREES,
        )


def check_tree_header(n_nodes: Int, n_leaves: Int) raises:
    """One tree's declared sizes, before any of its arrays are allocated.

    A binary tree in which every internal node has two children has
    `2 * n_leaves - 1` nodes, so the leaf count is what bounds the node
    count. Both directions matter: a leaf count larger than the node count is
    impossible, and a node count far above the leaf count describes a tree
    with unreachable nodes.
    """
    if n_nodes < 1:
        raise Error("corrupt tree header: node count of ", n_nodes)
    if n_leaves < 1:
        raise Error("corrupt tree header: leaf count of ", n_leaves)
    if n_nodes > MAX_TREE_NODES:
        raise Error(
            "tree declares ",
            n_nodes,
            " nodes, above the limit of ",
            MAX_TREE_NODES,
        )
    if n_leaves > n_nodes:
        raise Error(
            "corrupt tree header: ",
            n_leaves,
            " leaves cannot fit in ",
            n_nodes,
            " nodes",
        )
    if n_nodes > 2 * n_leaves - 1:
        raise Error(
            "corrupt tree header: ",
            n_nodes,
            " nodes for ",
            n_leaves,
            " leaves, and a binary tree with that many leaves has at most ",
            2 * n_leaves - 1,
            " nodes",
        )


def check_model_nodes(total_nodes: Int) raises:
    """The running node total across an ensemble. Per-tree and per-model
    ceilings are both needed: four million trees of four nodes each exhaust
    memory as surely as one tree of sixteen million, and only the running
    total catches the first case."""
    check_alloc(total_nodes, "model node count")
    if total_nodes > MAX_MODEL_NODES:
        raise Error(
            "model holds ",
            total_nodes,
            " nodes in total, above the limit of ",
            MAX_MODEL_NODES,
        )


def check_tree_topology(
    feature: List[Int],
    left: List[Int],
    right: List[Int],
    n_nodes: Int,
    n_leaves: Int,
    n_features: Int,
) raises:
    """A loaded tree must be walkable in bounded time.

    Bounds checking the child indices is not enough, and this is the gap this
    function exists to close. Every grower in mojotrees appends nodes as it
    splits them, so a child always sits at a higher index than its parent,
    and both `predict_raw_row` and the contribution recursion rely on that:
    the first walks down with a `while` loop that has no visit budget, and the
    second recurses once per edge. A file whose node 3 names node 1 as its
    left child satisfies every bounds check in the reader and then hangs the
    first prediction, or overflows the stack. Requiring `child > parent`
    makes the child index a strict decrease in the remaining node count, so
    every walk terminates in at most `n_nodes` steps and every recursion is
    at most `n_nodes` deep.

    The rest is the structure that follows from it: an internal node
    (`feature >= 0`) has two distinct in-range children, a leaf
    (`feature < 0`) has none, every node except the root is some node's
    child exactly once, and the leaves counted match the header.
    """
    check_tree_header(n_nodes, n_leaves)
    if len(feature) != n_nodes:
        raise Error(
            "corrupt tree: ",
            len(feature),
            " split features for ",
            n_nodes,
            " nodes",
        )
    if len(left) != n_nodes or len(right) != n_nodes:
        raise Error(
            "corrupt tree: ",
            len(left),
            " left and ",
            len(right),
            " right child indices for ",
            n_nodes,
            " nodes",
        )

    var parents = List[Int](capacity=n_nodes)
    parents.resize(n_nodes, -1)
    var leaves = 0
    for i in range(n_nodes):
        var f = feature[i]
        if f >= n_features:
            raise Error(
                "corrupt tree: node ",
                i,
                " splits on feature ",
                f,
                ", which is outside [0, ",
                n_features,
                ")",
            )
        if f < 0:
            leaves += 1
            continue
        var l = left[i]
        var r = right[i]
        if l <= i or l >= n_nodes:
            raise Error(
                "corrupt tree: node ",
                i,
                " has left child ",
                l,
                ", and a child must sit after its parent and inside [0, ",
                n_nodes,
                ") so that a walk terminates",
            )
        if r <= i or r >= n_nodes:
            raise Error(
                "corrupt tree: node ",
                i,
                " has right child ",
                r,
                ", and a child must sit after its parent and inside [0, ",
                n_nodes,
                ") so that a walk terminates",
            )
        if l == r:
            raise Error(
                "corrupt tree: node ",
                i,
                " names node ",
                l,
                " as both children",
            )
        if parents[l] >= 0:
            raise Error(
                "corrupt tree: node ",
                l,
                " is a child of both node ",
                parents[l],
                " and node ",
                i,
            )
        if parents[r] >= 0:
            raise Error(
                "corrupt tree: node ",
                r,
                " is a child of both node ",
                parents[r],
                " and node ",
                i,
            )
        parents[l] = i
        parents[r] = i

    for i in range(1, n_nodes):
        if parents[i] < 0:
            raise Error(
                "corrupt tree: node ", i, " is unreachable from the root"
            )
    if leaves != n_leaves:
        raise Error(
            "corrupt tree: header declares ",
            n_leaves,
            " leaves but ",
            leaves,
            " nodes have no split feature",
        )


def tree_depth(
    feature: List[Int], left: List[Int], right: List[Int], n_nodes: Int
) raises -> Int:
    """The depth of the deepest leaf, in edges from the root.

    Requires `check_tree_topology` to have passed, and reads its guarantee
    directly: because every child sits after its parent, one forward pass
    computes each node's depth from its parent's, with no traversal and no
    stack. That is also the reason this is safe to call on a file that has
    only just been read.
    """
    if n_nodes < 1:
        return 0
    var depth = List[Int](capacity=n_nodes)
    depth.resize(n_nodes, 0)
    var deepest = 0
    for i in range(n_nodes):
        if depth[i] > deepest:
            deepest = depth[i]
        if feature[i] < 0:
            continue
        depth[left[i]] = depth[i] + 1
        depth[right[i]] = depth[i] + 1
    return deepest


# ---------------------------------------------------------------------------
# Depth, leaves, and iteration counts
# ---------------------------------------------------------------------------


def check_num_leaves(num_leaves: Int) raises:
    """A tree needs at least two leaves to have a split in it. The upper
    bound is the per-tree node ceiling read back through
    `n_nodes = 2 * n_leaves - 1`, so a `num_leaves` that would grow a tree no
    loader could read back is refused before the run rather than after."""
    if num_leaves < 2:
        raise Error("num_leaves must be at least 2, got ", num_leaves)
    var ceiling = (MAX_TREE_NODES + 1) // 2
    if num_leaves > ceiling:
        raise Error(
            "num_leaves of ",
            num_leaves,
            " exceeds the limit of ",
            ceiling,
            ", above which the tree could not be serialized",
        )


def check_max_depth(max_depth: Int) raises:
    """LightGBM's `max_depth`: values at or below 0 mean unlimited. The
    ceiling rejects a value so large it can only be a unit error, since depth
    is bounded by `num_leaves` long before it is reached."""
    if max_depth > MAX_DEPTH_LIMIT:
        raise Error(
            "max_depth of ",
            max_depth,
            " exceeds the limit of ",
            MAX_DEPTH_LIMIT,
            "; values at or below 0 mean unlimited",
        )


def check_depth_budget(depth: Int, max_depth: Int) raises:
    """The runtime guard a grower calls before descending. `max_depth <= 0`
    is unlimited in LightGBM's sense, but not in this layer's: the absolute
    ceiling still applies, because an unlimited depth is what an unbounded
    recursion looks like from inside."""
    if depth < 0:
        raise Error("node depth cannot be negative, got ", depth)
    if max_depth > 0 and depth > max_depth:
        raise Error(
            "node depth ", depth, " exceeds max_depth ", max_depth
        )
    if depth > MAX_DEPTH_LIMIT:
        raise Error(
            "node depth ",
            depth,
            " exceeds the absolute limit of ",
            MAX_DEPTH_LIMIT,
        )


def check_iterations(n_estimators: Int) raises:
    """The round count. Zero is legal and trains no trees, which is what
    LightGBM's `num_iterations=0` does and what continued training with
    nothing to add looks like."""
    if n_estimators < 0:
        raise Error(
            "num_iterations must be nonnegative, got ", n_estimators
        )
    if n_estimators > MAX_ITERATIONS:
        raise Error(
            "num_iterations of ",
            n_estimators,
            " exceeds the limit of ",
            MAX_ITERATIONS,
        )


def check_iteration_range(start: Int, end: Int, n_trees: Int) raises:
    """A half-open slice of an ensemble, as prediction and truncation use it.
    `end` at or below 0 means "to the end", which is LightGBM's convention
    for `num_iteration=0`."""
    if start < 0:
        raise Error("iteration range start must be nonnegative, got ", start)
    if start > n_trees:
        raise Error(
            "iteration range starts at ",
            start,
            " in a model with ",
            n_trees,
            " trees",
        )
    if end > n_trees:
        raise Error(
            "iteration range ends at ",
            end,
            " in a model with ",
            n_trees,
            " trees",
        )
    if end > 0 and end < start:
        raise Error(
            "iteration range ends at ", end, " before it starts at ", start
        )


def check_early_stopping_rounds(rounds: Int) raises:
    """Zero disables early stopping; a negative count is a unit error, since
    a patience of -1 reads as "stop before starting"."""
    if rounds < 0:
        raise Error(
            "early_stopping_rounds must be nonnegative, got ", rounds
        )
    if rounds > MAX_ITERATIONS:
        raise Error(
            "early_stopping_rounds of ",
            rounds,
            " exceeds the limit of ",
            MAX_ITERATIONS,
        )


# ---------------------------------------------------------------------------
# Booster hyperparameter ranges
# ---------------------------------------------------------------------------


def check_booster_ranges(
    n_estimators: Int,
    learning_rate: Float64,
    num_leaves: Int,
    max_depth: Int,
    min_data_in_leaf: Int,
    min_child_hess: Float64,
    lambda_l1: Float64,
    lambda_l2: Float64,
    feature_fraction: Float64,
    feature_fraction_bynode: Float64,
    feature_fraction_bylevel: Float64,
) raises:
    """The data-independent hyperparameter ranges, in one place.

    This is the exact set that `params._validate` checks when a parameter
    string is parsed and that `callback.check_resettable` checks again when a
    schedule rewrites a round's parameters. Two copies is one copy too many:
    they had already drifted, with `_validate` bounding
    `feature_fraction_bylevel` and `check_resettable` not, so a callback
    could set a bylevel fraction of 0.0 that the parser would have rejected.

    Scalars rather than a `BoosterParams`, so that this module stays free of
    package imports and both callers can hand over what they hold.
    Objective-dependent parameters (`alpha`, `fair_c`,
    `tweedie_variance_power`) are not here: their legal range depends on
    which objective is running, which is `boosting._check_objective`'s to
    know.
    """
    check_iterations(n_estimators)
    check_positive_scalar(learning_rate, "learning_rate")
    check_num_leaves(num_leaves)
    check_max_depth(max_depth)
    if min_data_in_leaf < 0:
        raise Error(
            "min_data_in_leaf must be nonnegative, got ", min_data_in_leaf
        )
    check_nonnegative_scalar(min_child_hess, "min_sum_hessian_in_leaf")
    check_nonnegative_scalar(lambda_l1, "lambda_l1")
    check_nonnegative_scalar(lambda_l2, "lambda_l2")
    check_unit_fraction(feature_fraction, "feature_fraction")
    check_unit_fraction(
        feature_fraction_bynode, "feature_fraction_bynode"
    )
    check_unit_fraction(
        feature_fraction_bylevel, "feature_fraction_bylevel"
    )


# ---------------------------------------------------------------------------
# Cancellation and resource cleanup
# ---------------------------------------------------------------------------


comptime CONTROL_CODES: Int = 3
"""How many callback control codes exist. The codes themselves
(`CONTINUE = 0`, `STOP = 1`, `ABORT = 2`) are named in callback.mojo; this
module knows only that they are the integers `[0, CONTROL_CODES)`, which is
all `check_control_code` needs and is what keeps the names in one place."""


def check_control_code(code: Int, phase: String, iteration: Int) raises -> Int:
    """A control code returned by a training callback.

    The training loop tests for `ABORT`, then for `STOP`, and treats
    everything else as `CONTINUE`. That last clause is the problem: a
    callback that returns 7, or that falls off the end of a branch and
    returns whatever a bridge defaulted to, keeps training and says nothing.
    A control code is the one value in the callback protocol the loop cannot
    sanity check from context, so it is checked explicitly.

    Returns the code, so a caller can write `var c = check_control_code(...)`
    and keep the comparison it already had.
    """
    if code < 0 or code >= CONTROL_CODES:
        raise Error(
            "training callback returned control code ",
            code,
            " in the ",
            phase,
            " phase of round ",
            iteration,
            "; the legal codes are 0 (continue), 1 (stop), and 2 (abort)",
        )
    return code


# `CancelToken` lives in `sequence.mojo`: one token for chunk drivers and
# training loops alike (`live()`, `cancel(reason)`, `is_cancelled()`,
# `why()`, `check(where)` are the spellings this module's callers use), and
# the package root exports it from there.


def check_cleanup_balanced(
    acquired: Int, released: Int, kind: String
) raises:
    """A teardown released exactly what it acquired.

    Called at the end of a scope that owns a countable resource: device
    buffers taken from a pool, staging slots, open shards. It is not a
    replacement for the teardown itself, which `gpu_runtime.close` and its
    neighbors own and order correctly. It is the assertion that the teardown
    finished, expressed as an error rather than as a leak that shows up as
    the next run's allocation failure.

    Releasing more than was acquired is reported too, and separately: a
    double release is a use-after-free in waiting, and reporting it as "not
    balanced" would let it read as the milder of the two.
    """
    if acquired < 0 or released < 0:
        raise Error(
            kind,
            " cleanup counts cannot be negative, got ",
            acquired,
            " acquired and ",
            released,
            " released",
        )
    if released > acquired:
        raise Error(
            kind,
            " was released more often than it was acquired: ",
            released,
            " releases for ",
            acquired,
            " acquisitions, which is a double release",
        )
    if released < acquired:
        raise Error(
            kind,
            " leaked at teardown: ",
            acquired - released,
            " of ",
            acquired,
            " acquisitions were never released",
        )


# ---------------------------------------------------------------------------
# Composite entry points
# ---------------------------------------------------------------------------
#
# Each of these is the whole contract of one existing call site, in the order
# that site checks it. They exist so that adopting this module is a one-line
# edit rather than a rewrite: a caller replaces its block of `if ... raise`
# with one call and keeps its own control flow around it. Nothing here adds a
# rule; every one is a call to the narrow checks above, and a caller that
# wants a different subset calls those directly.


def check_dataset_columns(
    n_rows: Int,
    n_features: Int,
    n_values: Int,
    n_label: Int,
    n_weight: Int,
    n_group: Int,
    n_init_score: Int,
    n_feature_names: Int,
) raises:
    """The length half of a dataset's construction contract.

    This is `trainset.Dataset.__init__`'s opening block, which validates
    every optional column against the row count at construction rather than
    at train time, so a mismatch is reported while the caller still knows
    which array they passed. The group *contents* are checked separately, by
    `check_group_counts`, because they need the counts themselves.
    """
    check_dense_matrix(n_values, n_rows, n_features)
    check_column_length(n_label, n_rows, "label")
    check_column_length(n_weight, n_rows, "weight")
    check_column_length(n_init_score, n_rows, "init_score")
    if n_feature_names != 0 and n_feature_names != n_features:
        raise Error(
            "feature_name must have one name per feature: got ",
            n_feature_names,
            " for ",
            n_features,
            " features",
        )
    if n_group != 0 and n_group > n_rows:
        raise Error(
            "group has ",
            n_group,
            " queries but the data has only ",
            n_rows,
            " rows, and every query needs at least one row",
        )


def check_training_inputs(
    n_rows: Int,
    n_features: Int,
    label: List[Float64],
    weight: List[Float64],
    n_init_score: Int,
) raises -> Float64:
    """The contract every single-output trainer opens with, and the total
    weight it produces.

    Shape, one finite label per row, the sample-weight rules, and an init
    score that is absent or one per row. The objective's own rules about
    those labels (nonnegative for Poisson, positive for gamma, and the rest)
    stay in `boosting._check_objective`, which runs after this and knows
    which objective it is.
    """
    check_shape(n_rows, n_features)
    check_labels_finite(label, n_rows)
    check_column_length(n_init_score, n_rows, "init_score")
    return check_weights(weight, n_rows)


def check_multiclass_inputs(
    n_rows: Int,
    n_features: Int,
    label: List[Float64],
    weight: List[Float64],
    n_classes: Int,
) raises -> List[Int]:
    """The multiclass counterpart, returning the class codes.

    `check_classes_present` runs here rather than being left to the trainer
    because the codes have just been built and the sweep is free at that
    point; deferring it would mean a second pass over the labels.
    """
    check_shape(n_rows, n_features)
    check_required_length(len(label), n_rows, "label")
    _ = check_weights(weight, n_rows)
    var codes = check_class_codes(label, n_classes)
    check_classes_present(codes, n_classes)
    return codes^


def check_ranking_inputs(
    n_rows: Int,
    n_features: Int,
    label: List[Float64],
    weight: List[Float64],
    group: List[Int],
) raises -> List[Int]:
    """The ranking counterpart, returning the graded relevances.

    A ranking dataset without `group` is refused here rather than defaulted
    to one query per row: LambdaRank computes its gradients within a query,
    so treating every row as its own query produces zero lambdas everywhere
    and a model that fits nothing.
    """
    check_shape(n_rows, n_features)
    check_required_length(len(label), n_rows, "label")
    _ = check_weights(weight, n_rows)
    if len(group) == 0:
        raise Error(
            "a ranking dataset needs `group`: the number of rows in each"
            " query, in row order"
        )
    _ = check_group_counts(group, n_rows)
    return check_relevance_labels(label)


def check_valid_set(
    train_n_features: Int,
    valid_n_features: Int,
    valid_n_rows: Int,
    n_valid_label: Int,
) raises:
    """A validation set is shaped like the training set it is scored against.

    The feature count is the one that matters: a validation matrix with a
    different width is binned by the training mapper into bins that mean
    different features, and the resulting metric is a number with no meaning
    rather than a bad score.
    """
    if valid_n_features != train_n_features:
        raise Error(
            "valid_data must have the same features as the training data:"
            " got ",
            valid_n_features,
            " against ",
            train_n_features,
        )
    if valid_n_rows < 1:
        raise Error(
            "a validation set needs at least one row, got ", valid_n_rows
        )
    check_required_length(n_valid_label, valid_n_rows, "valid_label")


def check_loaded_tree(
    feature: List[Int],
    left: List[Int],
    right: List[Int],
    n_nodes: Int,
    n_leaves: Int,
    n_features: Int,
    running_total: Int,
) raises -> Int:
    """One tree as it comes off disk, plus the ensemble's running node total.

    Returns the updated total, so a loader threads one accumulator through
    its tree loop and both ceilings are enforced by the same call.
    """
    check_tree_topology(feature, left, right, n_nodes, n_leaves, n_features)
    var total = checked_add(running_total, n_nodes, "model node count")
    check_model_nodes(total)
    return total
