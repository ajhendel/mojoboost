"""Monotonic constraints for numerical features.

A constraint vector holds one entry per feature: `1` for nondecreasing, `-1`
for nonincreasing, `0` for unconstrained. A fitted model then satisfies, for
every constrained feature `f` and every pair of examples that differ only in
`f`,

    x_f <= x'_f  implies  pred(x) <= pred(x')     (constraint  1)
    x_f <= x'_f  implies  pred(x) >= pred(x')     (constraint -1)

globally, for any raw feature values, not just the ones in the training data.

Method
------
This is LightGBM's `monotone_constraints_method="basic"`, the cheap and
always-correct one. Two mechanisms working together:

1. **Output bounds.** Every node carries a closed interval its output must
   lie in. The root's is unbounded. Splitting a node on a constrained
   feature `f` computes `mid`, the midpoint of the two child outputs, and
   hands the child on the low side of the split the interval capped at `mid`
   while the child on the high side gets the interval floored at `mid`
   (swapped for a `-1` constraint). Bounds only ever tighten with depth, and
   every leaf value is clamped into its own interval.

2. **Candidate rejection.** While scanning split candidates for a
   constrained feature, a candidate whose left output would exceed its right
   output (or fall below it, for `-1`) is discarded outright, whatever its
   gain.

Why that is enough: take two examples differing only in `f`, with
`x_f < x'_f`. Their root-to-leaf paths agree until some node that splits on
`f` and sends them apart, `x` to the low child and `x'` to the high child
(bin ids are nondecreasing in the raw value, so this is the only way they can
diverge). Every leaf below the low child has an interval capped at that
node's `mid`, and every leaf below the high child has one floored at it, so
`pred(x) <= mid <= pred(x')`. Each tree is therefore monotone in `f`, and a
sum of monotone trees scaled by a positive learning rate, plus a constant
base score, is monotone. The response-scale links (sigmoid, exp) are
increasing, so the guarantee survives them.

Bounds are not stored on the tree. A grown tree's internal nodes keep the
leaf value they held when they were created, which is exactly what the
midpoints were computed from, so `node_bounds` in tree.mojo recovers the
whole interval chain from the tree and the constraint vector alone.

Cost when unused
----------------
An empty constraint vector, or one whose every entry is `0`, is inactive:
split search takes its original code path and the fit is bit-identical to
one with no constraint vector at all. Only an active vector pays for the
clamped-output gain formula.

LightGBM differences
--------------------
- LightGBM decides that constraints are in play by checking whether the
  vector is empty. mojotrees also treats an all-zero vector as inactive,
  which is what makes the bit-exact equivalence above testable.
- Only the `basic` method is implemented. LightGBM's `intermediate` and
  `advanced` methods recover some of the accuracy that `basic` gives up (by
  tracking bounds across whole subtrees rather than parent to child), and are
  not attempted here.
- Quantile and L1 leaf renewal replaces every leaf's Newton value with a
  residual percentile after the tree is grown, which knows nothing about
  monotonicity. mojotrees clamps the renewed value back into the leaf's
  interval so the guarantee holds for those objectives too. We have not
  verified what LightGBM does at this step, so treat the clamp as
  mojotrees-defined rather than matched: it biases the renewed quantile
  toward the interval, and the alternative is a model that silently violates
  the constraint it was asked for.

Multiclass policy
-----------------
Constraints apply to every per-class tree, so each class's **raw** score is
monotone in the constrained features. Softmax probabilities are **not**
guaranteed monotone: a class's probability also depends on the other
classes' raw scores, which move at their own rate. Constrain a multiclass
model when the raw score is what carries the meaning, and do not read the
probabilities as monotone.
"""

comptime MONOTONE_FREE = 0
comptime MONOTONE_INCREASING = 1
comptime MONOTONE_DECREASING = -1

# Stand-in for "no bound", the largest finite double rather than an infinity,
# so that midpoints of unbounded intervals stay finite. LightGBM's
# BasicConstraint uses the same sentinel.
comptime NO_BOUND = Float64.MAX_FINITE


struct MonotoneConstraints(Copyable, Movable):
    """One constraint sign per feature, in feature order.

    Empty means unconstrained. A non-empty vector must have exactly one entry
    per feature, each `-1`, `0`, or `1`; a vector of all zeros is accepted and
    is inactive.
    """

    var signs: List[Int]

    def __init__(out self):
        """No constraints."""
        self.signs = List[Int]()

    @staticmethod
    def from_signs(
        signs: List[Int], n_features: Int
    ) raises -> MonotoneConstraints:
        """Build from one sign per feature, validating length and values. An
        empty `signs` means unconstrained."""
        if len(signs) == 0:
            return MonotoneConstraints()
        if len(signs) != n_features:
            raise Error(
                "monotone_constraints needs one entry per feature: got ",
                len(signs),
                " for ",
                n_features,
                " features",
            )
        for f in range(len(signs)):
            var s = signs[f]
            if (
                s != MONOTONE_FREE
                and s != MONOTONE_INCREASING
                and s != MONOTONE_DECREASING
            ):
                raise Error(
                    "monotone constraint for feature ",
                    f,
                    " must be -1, 0, or 1, got ",
                    s,
                )
        var out = MonotoneConstraints()
        out.signs = signs.copy()
        return out^

    def is_empty(self) -> Bool:
        """True when no constraint vector was given at all."""
        return len(self.signs) == 0

    def is_active(self) -> Bool:
        """True when at least one feature is constrained. An all-zero vector
        is inactive and costs nothing during growth."""
        for f in range(len(self.signs)):
            if self.signs[f] != MONOTONE_FREE:
                return True
        return False

    def active_signs(self) -> List[Int]:
        """The signs to grow with: the vector itself when it constrains
        something, an empty list otherwise. Split search reads an empty list
        as "no constraints" and keeps its unconstrained code path."""
        if not self.is_active():
            return List[Int]()
        return self.signs.copy()

    def sign(self, feature: Int) -> Int:
        """This feature's constraint, `0` for features past the end of the
        vector (which includes the unconstrained case)."""
        return monotone_sign(self.signs, feature)

    def check_features(self, n_features: Int) raises:
        """Raise unless this vector fits a dataset with `n_features`
        columns."""
        if self.is_empty():
            return
        if len(self.signs) != n_features:
            raise Error(
                "monotone_constraints has ",
                len(self.signs),
                " entries but the data has ",
                n_features,
                " features",
            )


@always_inline
def monotone_sign(signs: List[Int], feature: Int) -> Int:
    """Constraint sign for `feature` from a raw sign vector, `0` when the
    vector is empty or does not reach that feature."""
    if feature < 0 or feature >= len(signs):
        return MONOTONE_FREE
    return signs[feature]


@fieldwise_init
struct OutputBounds(Copyable, Movable):
    """The closed interval `[lo, hi]` a node's output must lie in."""

    var lo: Float64
    var hi: Float64

    @staticmethod
    def unbounded() -> OutputBounds:
        return OutputBounds(-NO_BOUND, NO_BOUND)

    def is_active(self) -> Bool:
        return self.lo > -NO_BOUND or self.hi < NO_BOUND

    def clamp(self, value: Float64) -> Float64:
        if value < self.lo:
            return self.lo
        if value > self.hi:
            return self.hi
        return value


@fieldwise_init
struct ChildBounds(Copyable, Movable):
    """Output intervals for the two children of a node that was split."""

    var left: OutputBounds
    var right: OutputBounds


@always_inline
def midpoint(a: Float64, b: Float64) -> Float64:
    return (a + b) / 2.0


def child_bounds(
    parent: OutputBounds, sign: Int, left_value: Float64, right_value: Float64
) -> ChildBounds:
    """Split a node's interval between its children.

    An unconstrained split feature passes the parent's interval down
    unchanged. A constrained one splits it at the midpoint of the two child
    values, giving the low-value side the lower half. Both child values must
    already be clamped into `parent` and ordered per `sign`, so the midpoint
    lands inside `parent` and neither child gets an empty interval.
    """
    if sign == MONOTONE_FREE:
        return ChildBounds(parent.copy(), parent.copy())
    var mid = midpoint(left_value, right_value)
    if sign == MONOTONE_INCREASING:
        return ChildBounds(
            OutputBounds(parent.lo, mid), OutputBounds(mid, parent.hi)
        )
    return ChildBounds(
        OutputBounds(mid, parent.hi), OutputBounds(parent.lo, mid)
    )


@always_inline
def violates(sign: Int, left_output: Float64, right_output: Float64) -> Bool:
    """Whether a candidate split's child outputs run against `sign`."""
    if sign == MONOTONE_INCREASING:
        return left_output > right_output
    if sign == MONOTONE_DECREASING:
        return left_output < right_output
    return False


@always_inline
def output_score(
    grad_sum: Float64,
    hess_sum: Float64,
    lambda_reg: Float64,
    output: Float64,
) -> Float64:
    """LightGBM's GetLeafSplitGainGivenOutput: the second-order objective
    improvement of a leaf forced to emit `output`,

        -(2 * G * output + (H + lambda_l2) * output^2)

    `grad_sum` is the leaf's gradient sum after L1 soft-thresholding. At the
    unclamped Newton value `output = -G / (H + lambda_l2)` this reduces to the
    usual `G^2 / (H + lambda_l2)`, so a constrained scan that never clamps
    scores its candidates the same way an unconstrained one does (up to
    floating-point association).
    """
    return -(
        2.0 * grad_sum * output + (hess_sum + lambda_reg) * output * output
    )
