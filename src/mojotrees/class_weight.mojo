"""Class weighting.

Turns a class-level weighting policy into the per-row `sample_weight` the
trainers already take, which is the whole mechanism: nothing downstream of
this module knows a class weight from any other weight, so a class-weighted
run is bit-identical to the same run with those row weights passed by hand.

Three policies, each LightGBM's or scikit-learn's:

- explicit per-class weights, scikit-learn's `class_weight={0: 1.0, 1: 4.0}`
  and LightGBM's `class_weight`. `class_weight_rows` expands them.
- `balanced`, scikit-learn's `class_weight="balanced"`:
  `n_rows / (n_classes * count_k)` for class k, so every class contributes
  the same total weight and the weights average to 1.
- `scale_pos_weight` and `is_unbalance`, LightGBM's two binary-only knobs.
  `scale_pos_weight` multiplies the positive rows by a number you choose;
  `is_unbalance` chooses it for you as `negatives / positives`. They are
  mutually exclusive in LightGBM, and `check_class_balance_params` rejects
  the combination here rather than silently letting one win.

  **`is_unbalance` is not `balanced` up to a constant factor.** A uniform
  rescale of every row weight is not gain-invariant here: `lambda_l2`
  defaults to 1.0 and `min_sum_hessian_in_leaf` to 1e-3, so scaling `G`
  and `H` while `lambda_l2` stays fixed moves both
  `G**2 / (H + lambda_l2)` and `-G / (H + lambda_l2)`, and the two
  policies pick different splits and write different leaf values rather
  than the same tree at a different scale. On a majority-positive label
  they do not even agree on which class moves: `unbalance_scale` always
  leaves the negatives at 1.0, so its multiplier is below 1.0 there and it
  shrinks the positives, while LightGBM lifts whichever class is the
  minority (`src/objective/binary_objective.hpp:93-101` at 4.7.0). The
  ratio between the classes is the same in both, and that is the only part
  the old claim had right.

  A second difference from LightGBM in the same two knobs.
  `unbalance_scale` derives its ratio from weighted class counts
  (`class_counts` takes `sample_weight`) where LightGBM's `Init` counts
  rows. On an unweighted fit the two coincide.

Interaction with `sample_weight`: a class weight multiplies whatever row
weight the caller already has, LightGBM's rule and scikit-learn's. So a row
with `sample_weight` 2.0 in a class weighted 3.0 trains at 6.0.

What this does not do is change any objective. A class-weighted binary model
is still fitting the logistic loss; it is fitting it to a reweighted
sample, which moves the decision threshold that a given probability implies.
Weighting is not calibration: predicted probabilities from a class-weighted
model are probabilities under the *weighted* distribution, so a model
trained with `balanced` on 1% positives predicts far above the base rate by
design. That is the tradeoff scikit-learn's option makes too, and it is
worth knowing before reading the output as a probability.
"""

from std.math import isfinite


def check_class_weights(class_weights: List[Float64], n_classes: Int) raises:
    """One finite, nonnegative weight per class, with a positive sum.

    A zero is allowed and drops that class from training, exactly as a zero
    `sample_weight` drops a row. All zeros is not: there would be nothing
    left to fit."""
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if len(class_weights) != n_classes:
        raise Error("class_weight must have one entry per class")
    var total = 0.0
    for k in range(n_classes):
        if not isfinite(class_weights[k]):
            raise Error("class_weight entries must be finite")
        if class_weights[k] < 0.0:
            raise Error("class_weight entries must be nonnegative")
        total += class_weights[k]
    if total <= 0.0:
        raise Error("class_weight entries must have a positive sum")


def class_counts(
    labels: List[Int], n_classes: Int, sample_weight: List[Float64] = []
) raises -> List[Float64]:
    """Total weight per class, the row count per class when
    `sample_weight` is empty. Labels outside 0..n_classes-1 raise."""
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if len(labels) == 0:
        raise Error("class weighting needs at least one row")
    if len(sample_weight) > 0 and len(sample_weight) != len(labels):
        raise Error("sample_weight length must equal the number of rows")
    var counts = List[Float64](capacity=n_classes)
    for _ in range(n_classes):
        counts.append(0.0)
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
        if w < 0.0:
            raise Error("sample_weight entries must be nonnegative")
        counts[labels[r]] += w
    return counts^


def balanced_class_weights(
    labels: List[Int], n_classes: Int, sample_weight: List[Float64] = []
) raises -> List[Float64]:
    """scikit-learn's `class_weight="balanced"`:
    `total / (n_classes * count_k)` for class k.

    With `sample_weight` the counts are weighted counts, so balancing runs
    on the sample the model actually sees rather than on the row count. A
    class with no rows would divide by zero; that raises, because a class
    the training data never shows cannot be balanced against.
    """
    var counts = class_counts(labels, n_classes, sample_weight)
    var total = 0.0
    for k in range(n_classes):
        total += counts[k]
    if total <= 0.0:
        raise Error("class weighting needs a positive total weight")
    var out = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        if counts[k] <= 0.0:
            raise Error(
                String(
                    "class ",
                    k,
                    " has no weight in the training data, so 'balanced'"
                    " has nothing to balance",
                )
            )
        out.append(total / (Float64(n_classes) * counts[k]))
    return out^


def class_weight_rows(
    labels: List[Int],
    n_classes: Int,
    class_weights: List[Float64],
    sample_weight: List[Float64] = [],
) raises -> List[Float64]:
    """Per-row weights from per-class weights: `class_weights[label[r]]`,
    multiplied by `sample_weight[r]` when there is one.

    The result is what the trainers take, so this is the only place the
    class policy exists; `train`, `train_multiclass`, and the GPU trainers
    see ordinary row weights.
    """
    check_class_weights(class_weights, n_classes)
    if len(labels) == 0:
        raise Error("class weighting needs at least one row")
    if len(sample_weight) > 0 and len(sample_weight) != len(labels):
        raise Error("sample_weight length must equal the number of rows")
    var out = List[Float64](capacity=len(labels))
    var total = 0.0
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
        if w < 0.0:
            raise Error("sample_weight entries must be nonnegative")
        var combined = w * class_weights[labels[r]]
        total += combined
        out.append(combined)
    if total <= 0.0:
        raise Error(
            "class weighting produced a zero total weight; every row would"
            " be ignored"
        )
    return out^


def balanced_sample_weight(
    labels: List[Int], n_classes: Int, sample_weight: List[Float64] = []
) raises -> List[Float64]:
    """`class_weight_rows` under the `balanced` policy, the one call a
    caller needs for scikit-learn's `class_weight="balanced"`."""
    var weights = balanced_class_weights(labels, n_classes, sample_weight)
    return class_weight_rows(labels, n_classes, weights, sample_weight)


def binary_labels_to_codes(labels: List[Float64]) raises -> List[Int]:
    """{0, 1} Float64 labels as class codes, for reusing the multiclass
    class-weight helpers on a binary problem. Anything that is not 0 or 1
    raises: a 0.5 here would be a cross-entropy soft label, which has no
    class to weight."""
    var out = List[Int](capacity=len(labels))
    for r in range(len(labels)):
        if labels[r] == 0.0:
            out.append(0)
        elif labels[r] == 1.0:
            out.append(1)
        else:
            raise Error("binary class weighting needs labels in {0, 1}")
    return out^


def check_class_balance_params(
    is_unbalance: Bool, scale_pos_weight: Float64
) raises:
    """LightGBM's mutual exclusion: `is_unbalance` computes the positive
    class's weight, `scale_pos_weight` states it, and asking for both is
    asking for two different numbers. LightGBM warns and ignores one; this
    raises."""
    if not isfinite(scale_pos_weight) or scale_pos_weight <= 0.0:
        raise Error("scale_pos_weight must be positive")
    if is_unbalance and scale_pos_weight != 1.0:
        raise Error(
            "is_unbalance and scale_pos_weight cannot both be set; they are"
            " two ways to weight the positive class"
        )


def unbalance_scale(
    labels: List[Float64], sample_weight: List[Float64] = []
) raises -> Float64:
    """LightGBM's `is_unbalance`: the positive class's multiplier,
    `negative weight / positive weight`. Both classes must be present, since
    the ratio is otherwise zero or undefined."""
    var codes = binary_labels_to_codes(labels)
    var counts = class_counts(codes, 2, sample_weight)
    if counts[0] <= 0.0 or counts[1] <= 0.0:
        raise Error(
            "is_unbalance needs both classes present in the training data"
        )
    return counts[0] / counts[1]


def scale_pos_weight_rows(
    labels: List[Float64],
    scale_pos_weight: Float64,
    sample_weight: List[Float64] = [],
) raises -> List[Float64]:
    """Per-row weights with the positive rows multiplied by
    `scale_pos_weight`, LightGBM's parameter of that name. Negative rows
    keep their weight, which is what makes this different from `balanced`:
    the total weight grows rather than being redistributed."""
    var codes = binary_labels_to_codes(labels)
    var class_weights = List[Float64](capacity=2)
    class_weights.append(1.0)
    class_weights.append(scale_pos_weight)
    return class_weight_rows(codes, 2, class_weights, sample_weight)


def unbalanced_sample_weight(
    labels: List[Float64], sample_weight: List[Float64] = []
) raises -> List[Float64]:
    """`scale_pos_weight_rows` at the ratio `is_unbalance` computes."""
    return scale_pos_weight_rows(
        labels, unbalance_scale(labels, sample_weight), sample_weight
    )
