"""DART (Dropouts meet Multiple Additive Regression Trees) algorithm core.

This module is the algorithm, not a trainer and not a public API. Nothing
here grows a tree, reads a label, or owns a boosting loop: `boosting.mojo`
keeps that job, and this module supplies the three things DART adds to a
round that plain GBDT does not have.

    1. Which already-grown iterations are dropped this round, drawn from a
       counter-based stream so the draw depends only on (seed, round) and
       never on how the run was split into calls.
    2. What the dropped iterations and the new tree weigh afterwards, which
       is the normalization step DART exists for.
    3. How the cached raw scores move as those two happen, so the trainer
       never rebuilds a prediction it already had.

Everything is expressed over `(trees, weights, raw)`, the state a boosting
loop already carries, plus one weight per tree that the current model
representation does not yet have. That missing weight is the whole reason
DART is not reachable from any public entry point; see "Model state" below
and handoffs/remaining_01_dart.md.

Why a weight per tree
---------------------

`Booster` and `MulticlassBooster` shrink every tree by one scalar,
`learning_rate`, and `Booster.predict_raw_row` sums
`learning_rate * trees[i].predict_row(...)`. That is exact for GBDT, where
every tree really does carry the same factor, and it is the reason those
structs serialize one number instead of a vector.

DART breaks that invariant on purpose. When a round drops k iterations, the
new tree enters at a reduced weight and each dropped iteration is scaled
down so that the dropped group plus the newcomer weigh about what the
dropped group weighed alone. After one such round the ensemble holds trees
with at least two distinct factors, and no single scalar reproduces it. So a
DART model is not representable by today's `Booster`, cannot be written by
today's `serialize.mojo`, and must not be reachable from `fit` until both
carry a weight vector. This module therefore takes and returns the weight
list explicitly rather than pretending a `Booster` could hold one.

Determinism
-----------

The draw follows the rule the rest of the library follows (bagging.mojo,
goss.mojo, sampling.mojo): splitmix64 over a counter, seeded by
`(seed, round)`, so a draw depends on its seed and its absolute round index
and never on history. Continued training that resumes at round 40 draws what
an uninterrupted 100-round run would have drawn at round 40, which is the
property `train_more` already promises for bagging and GOSS.

Within a round the counter stream is laid out so nothing collides: offset 0
is the skip decision and offset `1 + i` is iteration i's draw.

What is deliberately refused
----------------------------

`uniform_drop=False`, LightGBM's default, selects the drop set with a
non-uniform rule this module does not implement. It is refused by name
rather than silently approximated by the uniform draw, the same way
`params.mojo` refuses `multiclassova` instead of quietly training something
else. Ranking, GPU training, and GOSS are refused for the reasons in
`check_dart_supported`.

Model state
-----------

A DART model needs, beyond what a GBDT model needs, one Float64 weight per
tree. Prediction, iteration slicing, continued training, and serialization
all have to read it. `dart_weights_are_uniform` exists so a consumer can
tell a DART model that happens to be uniform (every round skipped) from one
that is not, and so a serializer can keep writing the compact scalar form
when the vector carries no information.
"""

from .binning import BinnedMatrix
from .device import CPU_DEVICE
from .tree import Tree

# LightGBM's DART defaults. `drop_seed` is LightGBM's own default of 4, which
# keeps the seed distinct from `bagging_seed` (3) and the feature-fraction
# seed, so two samplers in one run never share a stream.
comptime DEFAULT_DROP_RATE = 0.1
comptime DEFAULT_MAX_DROP = 50
comptime DEFAULT_SKIP_DROP = 0.5
comptime DEFAULT_DROP_SEED = 4

comptime _GOLDEN = UInt64(0x9E3779B97F4A7C15)
comptime _TWO_POW_NEG_53 = 1.0 / 9007199254740992.0


@fieldwise_init
struct DartParams(Copyable, Movable):
    """DART configuration. Disabled by default: ordinary GBDT drops nothing.

    `drop_rate` is the per-iteration probability of being dropped in a round
    that drops at all, `skip_drop` the probability that a round drops
    nothing, and `max_drop` a cap on the size of one round's drop set with
    values <= 0 meaning uncapped. `uniform_drop` selects the drop set by an
    independent draw per iteration; see the module docstring for why the
    non-uniform variant is refused. `xgboost_dart_mode` selects XGBoost's
    normalization constant instead of the DART paper's; see
    `dart_normalization`.
    """

    var enabled: Bool
    var drop_rate: Float64
    var max_drop: Int
    var skip_drop: Float64
    var uniform_drop: Bool
    var xgboost_dart_mode: Bool
    var seed: Int

    @staticmethod
    def disabled() -> DartParams:
        """Ordinary GBDT (the library default). The other fields hold
        LightGBM's defaults so that flipping `enabled` alone gives LightGBM's
        configuration, except `uniform_drop`, which is True here because the
        False branch is unimplemented and refused."""
        return DartParams(
            False,
            DEFAULT_DROP_RATE,
            DEFAULT_MAX_DROP,
            DEFAULT_SKIP_DROP,
            True,
            False,
            DEFAULT_DROP_SEED,
        )

    @staticmethod
    def enable(
        drop_rate: Float64 = DEFAULT_DROP_RATE,
        max_drop: Int = DEFAULT_MAX_DROP,
        skip_drop: Float64 = DEFAULT_SKIP_DROP,
        uniform_drop: Bool = True,
        xgboost_dart_mode: Bool = False,
        seed: Int = DEFAULT_DROP_SEED,
    ) -> DartParams:
        """DART with LightGBM's defaults, save `uniform_drop`; see
        `disabled`."""
        return DartParams(
            True,
            drop_rate,
            max_drop,
            skip_drop,
            uniform_drop,
            xgboost_dart_mode,
            seed,
        )

    def validate(self) raises:
        """Reject out-of-range settings. Comparisons are written so that a
        NaN rate is rejected too, as in `GossParams.validate`."""
        if not self.enabled:
            return
        if not (self.drop_rate >= 0.0 and self.drop_rate <= 1.0):
            raise Error("dart drop_rate must be in [0, 1]")
        if not (self.skip_drop >= 0.0 and self.skip_drop <= 1.0):
            raise Error("dart skip_drop must be in [0, 1]")
        if self.seed < 0:
            raise Error("dart drop_seed must be nonnegative")
        if not self.uniform_drop:
            raise Error(
                "dart uniform_drop=false is not implemented; LightGBM selects"
                " the drop set by a non-uniform rule that mojoboost has not"
                " reproduced, and approximating it with the uniform draw"
                " would train a different model than the parameter names"
                " promise. Set uniform_drop=true"
            )
        # `drop_rate=0` with `skip_drop=1` is a DART run that never drops,
        # which is plain GBDT wearing DART's parameter names. That is a
        # configuration mistake worth naming: the caller asked for dropout
        # and would silently get none.
        if self.drop_rate == 0.0 and self.skip_drop >= 1.0:
            raise Error(
                "dart with drop_rate=0 and skip_drop=1 never drops a tree,"
                " which is ordinary gbdt; select boosting='gbdt' instead"
            )


def _splitmix64(state: UInt64) -> UInt64:
    """splitmix64's mixing function, as in bagging.mojo: a bijection with
    full avalanche, so consecutive counter values look independent."""
    var z = state + _GOLDEN
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    """Uniform in [0, 1) with 53 significant bits, from a counter value."""
    return Float64(_splitmix64(counter) >> 11) * _TWO_POW_NEG_53


def _stream(seed: Int, round: Int) -> UInt64:
    """Start of the counter stream for one round. The sign bit is masked off
    so negative seeds are accepted without relying on signed conversion, the
    same rule bagging.mojo uses."""
    return _splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ (UInt64(round) * _GOLDEN)
    )


struct DartDrop(Copyable, Movable):
    """One round's dropout decision.

    `iterations` holds the dropped boosting iteration indices, ascending and
    duplicate free. These are *iteration* indices, not tree indices: a
    multiclass round grows one tree per class and they are dropped together,
    because dropping one class's tree alone would tilt the softmax toward the
    classes whose trees survived.

    `skipped` is True when the round's skip draw fired, in which case
    `iterations` is empty and the round is an ordinary GBDT round. An empty
    `iterations` with `skipped` False means the ensemble had nothing to drop
    yet, which is what every round before the first tree looks like.
    """

    var iterations: List[Int]
    var skipped: Bool

    def __init__(out self, var iterations: List[Int], skipped: Bool):
        self.iterations = iterations^
        self.skipped = skipped

    @staticmethod
    def none(skipped: Bool) -> DartDrop:
        """A round that drops nothing, either because the skip draw fired or
        because the ensemble is still empty."""
        return DartDrop(List[Int](), skipped)

    @always_inline
    def count(self) -> Int:
        return len(self.iterations)

    @always_inline
    def is_empty(self) -> Bool:
        return len(self.iterations) == 0


def _insert_by_key(
    mut keys: List[Float64], mut values: List[Int], key: Float64, value: Int,
    cap: Int,
):
    """Insert `(key, value)` into a list kept ascending by key and truncated
    to `cap` entries, with `cap <= 0` meaning uncapped. Used to keep the
    `max_drop` most eligible candidates without sorting the whole ensemble.

    Written with append and index assignment only, which is the list surface
    the rest of the library uses.
    """
    var pos = len(keys)
    for i in range(len(keys)):
        if key < keys[i]:
            pos = i
            break
    if cap > 0 and pos >= cap:
        return
    # Grow by one, then shift the tail right to open a slot at `pos`.
    keys.append(key)
    values.append(value)
    var slot = len(keys) - 1
    while slot > pos:
        keys[slot] = keys[slot - 1]
        values[slot] = values[slot - 1]
        slot -= 1
    keys[pos] = key
    values[pos] = value
    if cap > 0 and len(keys) > cap:
        _ = keys.pop()
        _ = values.pop()


def select_drop(
    params: DartParams, n_iterations: Int, round: Int
) raises -> DartDrop:
    """Draw the set of iterations dropped in `round`.

    Deterministic in `(params.seed, round, n_iterations)` alone. The draw
    reads the absolute round index, so a continued run resumes the schedule
    an uninterrupted run would have followed, matching `refresh_bag` and
    `goss_round`.

    The rule, in order:

    1. Offset 0 of the round's stream is the skip draw. Below `skip_drop`
       the round drops nothing and returns `skipped=True`.
    2. Offset `1 + i` is iteration i's draw. Iteration i is a candidate when
       its draw is below `drop_rate`.
    3. If no iteration was drawn and the ensemble is not empty, the single
       iteration with the smallest draw is dropped anyway. A dropout round
       that drops nothing is indistinguishable from a skipped one, which
       would make `skip_drop` mean two different things at once.
    4. If more iterations were drawn than `max_drop` allows, the `max_drop`
       with the smallest draws are kept. LightGBM truncates a shuffled list
       instead; this rule is deterministic in the same draws that selected
       the set, so it needs no second random stream. It is a documented
       mojoboost difference, not a reproduction of LightGBM's order.

    An empty ensemble yields an empty, unskipped drop: there is nothing to
    drop before the first tree exists, and reporting that as a skip would
    misattribute it to the skip draw.
    """
    params.validate()
    if not params.enabled or n_iterations <= 0:
        return DartDrop.none(False)

    var base = _stream(params.seed, round)
    if _uniform(base) < params.skip_drop:
        return DartDrop.none(True)

    var cap = params.max_drop if params.max_drop > 0 else 0
    var keys = List[Float64]()
    var picked = List[Int]()
    var best_key = 0.0
    var best_iter = -1
    for i in range(n_iterations):
        var u = _uniform(base + UInt64(i + 1))
        if best_iter < 0 or u < best_key:
            best_key = u
            best_iter = i
        if u < params.drop_rate:
            _insert_by_key(keys, picked, u, i, cap)

    if len(picked) == 0:
        # Rule 3: guarantee a non-empty drop set for an unskipped round.
        picked.append(best_iter)
        return DartDrop(picked^, False)

    # `picked` is ordered by draw, not by iteration. The raw-score arithmetic
    # accumulates in this order, so sort it ascending by iteration to keep the
    # sum reproducible independent of which draws selected it. The set is at
    # most `max_drop` long, so an insertion sort is the right shape here.
    var ordered = List[Int](capacity=len(picked))
    var taken = List[Bool](capacity=len(picked))
    for _ in range(len(picked)):
        taken.append(False)
    for _ in range(len(picked)):
        var best = -1
        var best_pos = -1
        for j in range(len(picked)):
            if not taken[j] and (best_pos < 0 or picked[j] < best):
                best = picked[j]
                best_pos = j
        taken[best_pos] = True
        ordered.append(best)
    return DartDrop(ordered^, False)


@fieldwise_init
struct DartNormalization(Copyable, Movable):
    """The two factors a dropout round produces.

    `new_weight` is the weight the round's new tree enters with, and
    `dropped_scale` multiplies the weight of every dropped iteration. Both
    are positive, which is what lets monotone constraints survive DART: a
    positive combination of trees each monotone in a feature is monotone in
    that feature, so `Booster.monotone` keeps meaning what it claims.
    """

    var new_weight: Float64
    var dropped_scale: Float64


def dart_normalization(
    k: Int, learning_rate: Float64, xgboost_dart_mode: Bool
) -> DartNormalization:
    """The normalization for a round that dropped `k` iterations.

    With k dropped iterations weighing W in total, the point of the step is
    that the k dropped iterations plus the new tree weigh about W afterwards,
    so the ensemble does not grow in scale every time a round drops.

    Two conventions, both of which shrink by `learning_rate` first and
    normalize on top of it:

    - The DART paper's, and LightGBM's default: the new tree enters at
      `learning_rate / (k + 1)` and each dropped iteration keeps
      `k / (k + 1)` of its weight.
    - XGBoost's, which LightGBM exposes as `xgboost_dart_mode`: the new tree
      enters at `learning_rate / (k + learning_rate)` and each dropped
      iteration keeps `k / (k + learning_rate)`.

    `k == 0` is the no-drop round: the new tree enters at `learning_rate`
    and nothing is rescaled, so a DART run whose every round skips produces
    exactly the GBDT ensemble, weight for weight. That identity is what makes
    `skip_drop=1` degenerate rather than merely unusual, and it is the
    cheapest correctness check there is.

    NOTE: whether LightGBM's factor is `learning_rate / (k + 1)` or
    `1 / (k + 1)` (that is, whether normalization multiplies the shrinkage or
    replaces it) has NOT been verified against LightGBM's `dart.hpp` in this
    lane, which ran no code and fetched nothing. The convention above is the
    one the DART paper implies once shrinkage is applied, and it is isolated
    to this function so that settling the question is a one-line change. Do
    not claim LightGBM parity for DART on the strength of this function.
    """
    if k <= 0:
        return DartNormalization(learning_rate, 1.0)
    var kf = Float64(k)
    var denom = kf + learning_rate if xgboost_dart_mode else kf + 1.0
    return DartNormalization(learning_rate / denom, kf / denom)


def check_dart_supported(
    params: DartParams,
    device: Int,
    is_ranking: Bool,
    goss_enabled: Bool,
    n_classes: Int,
) raises:
    """Refuse the combinations whose DART semantics are not settled.

    Each is refused by name, with what it would take, rather than being
    downgraded to GBDT or trained as something the parameters do not
    describe. That is the rule `params.mojo` and
    `tree_parameters_extra.check_extra_option_supported` already follow.

    What is allowed, and why, so the list is not read as "DART is refused":

    - Monotone constraints. Every weight this module produces is positive
      (`dart_normalization`), and a positive combination of trees monotone in
      a feature is monotone in it, so the claim `Booster.monotone` records
      still holds after a dropout round.
    - Interaction constraints, categorical splits, feature subsampling, and
      row bagging. None of them read a tree's weight; they shape how a tree
      is grown, and DART only changes what a grown tree weighs.
    - Leaf-renewing objectives (mae, mape, quantile, huber). Renewal fits
      leaf values to residuals of the current raw scores, and under DART the
      current raw scores are the dropped-out ones, which is the residual the
      new tree is actually responsible for. The trainer has to renew against
      the raw scores this module hands back from `dart_begin_round`, not
      against the full ensemble's; that is a requirement on the caller, not a
      reason to refuse.
    """
    if not params.enabled:
        return
    params.validate()
    if device != CPU_DEVICE:
        raise Error(
            "dart is CPU only; train_gpu advances device-resident raw scores"
            " by one shrinkage factor per round and has no path for undoing"
            " a dropped tree's contribution on the device"
        )
    if is_ranking:
        raise Error(
            "dart is not supported for lambdarank; the pairwise lambdas are"
            " computed from each query's current score ordering, and a"
            " dropout round reorders queries by trees the round then puts"
            " back, so the gradients would not describe the model being"
            " built"
        )
    if goss_enabled:
        raise Error(
            "dart cannot be combined with goss; goss ranks rows by gradient"
            " magnitude, which under dart is the gradient of the dropped-out"
            " ensemble rather than of the model, so the sampled rows would"
            " track the dropout draw instead of the residual"
        )
    if n_classes < 1:
        raise Error("dart n_classes must be at least 1")


@always_inline
def _slot(iteration: Int, class_index: Int, n_classes: Int) -> Int:
    """The tree index of `(iteration, class_index)` in the round-major layout
    `MulticlassBooster` uses: `trees[i * n_classes + k]`. With `n_classes` 1
    this is the single-output layout, where a slot is its iteration."""
    return iteration * n_classes + class_index


def dart_begin_round(
    data: BinnedMatrix,
    trees: List[Tree],
    weights: List[Float64],
    drop: DartDrop,
    n_classes: Int,
    mut raw: List[Float64],
    mut contribution: List[Float64],
) raises:
    """Remove the dropped iterations from the cached raw scores.

    On entry `raw` is the full ensemble's raw score per row (row-major
    `raw[r * n_classes + k]`, which is one entry per row when `n_classes` is
    1, matching `_boost_rounds` and `_boost_rounds_multiclass`). On exit
    `raw` is the dropped-out ensemble's score, and `contribution` holds
    exactly what was removed, so the round can put a rescaled share of it
    back without a second pass over the dropped trees.

    Keeping `contribution` is the reason a dropout round costs one pass over
    the dropped trees rather than two. It is sized and overwritten here, so
    a caller reuses one buffer across rounds.

    The new tree is grown against the `raw` this leaves behind. That is the
    whole mechanism: DART's new tree fits the residual of an ensemble that is
    missing k of its trees, which is why it cannot simply refine whatever the
    previous tree got wrong.
    """
    var n = data.n_rows
    var total = n * n_classes
    if len(raw) != total:
        raise Error("raw length must equal n_rows * n_classes")
    if len(weights) != len(trees):
        raise Error("dart needs one weight per tree")

    contribution.clear()
    for _ in range(total):
        contribution.append(0.0)
    if drop.is_empty():
        return

    for d in range(drop.count()):
        var it = drop.iterations[d]
        for k in range(n_classes):
            var s = _slot(it, k, n_classes)
            if s < 0 or s >= len(trees):
                raise Error("dart dropped an iteration the ensemble lacks")
            var w = weights[s]
            for r in range(n):
                contribution[r * n_classes + k] += (
                    w * trees[s].predict_row(data, r)
                )

    for i in range(total):
        raw[i] -= contribution[i]


def dart_commit_round(
    data: BinnedMatrix,
    drop: DartDrop,
    norm: DartNormalization,
    n_classes: Int,
    contribution: List[Float64],
    var new_trees: List[Tree],
    mut trees: List[Tree],
    mut weights: List[Float64],
    mut raw: List[Float64],
) raises:
    """Apply the normalization and fold the round's new trees in.

    `new_trees` is one tree per class for this round, in class order, which
    for a single-output model is a list of one. `contribution` is what
    `dart_begin_round` removed, and `raw` is the dropped-out score it left.

    Three things happen, in the order that keeps `raw` exact:

    1. Each dropped iteration's weight is scaled by `norm.dropped_scale`, and
       the same fraction of `contribution` goes back into `raw`. Scaling the
       stored weight and the cached score by one factor is what keeps the
       cache equal to a fresh sum over the trees, rather than drifting from
       it a round at a time.
    2. The new trees are appended at `norm.new_weight`.
    3. `raw` takes the new trees' contribution at that weight.

    After this returns, `raw[r * n_classes + k]` equals the base score plus
    `sum_j weights[j] * trees[j].predict_row(data, r)` over class k's trees,
    to floating-point association. `dart_recompute_raw` is the check.
    """
    var n = data.n_rows
    var total = n * n_classes
    if len(new_trees) != n_classes:
        raise Error("dart needs one new tree per class per round")
    if len(contribution) != total:
        raise Error("dart contribution length must equal n_rows * n_classes")
    if len(raw) != total:
        raise Error("raw length must equal n_rows * n_classes")
    if len(weights) != len(trees):
        raise Error("dart needs one weight per tree")

    for d in range(drop.count()):
        var it = drop.iterations[d]
        for k in range(n_classes):
            weights[_slot(it, k, n_classes)] *= norm.dropped_scale
    if not drop.is_empty():
        for i in range(total):
            raw[i] += norm.dropped_scale * contribution[i]

    for k in range(n_classes):
        for r in range(n):
            raw[r * n_classes + k] += (
                norm.new_weight * new_trees[k].predict_row(data, r)
            )
    for k in range(n_classes):
        trees.append(new_trees[k].copy())
        weights.append(norm.new_weight)


def dart_recompute_raw(
    data: BinnedMatrix,
    trees: List[Tree],
    weights: List[Float64],
    base_scores: List[Float64],
    n_classes: Int,
) raises -> List[Float64]:
    """Raw scores summed from the model, the slow way.

    This is what the incrementally maintained cache is supposed to equal. It
    exists for the continued-training path, which has to rebuild the cache
    from a fitted model rather than inherit it, and as the reference a later
    test compares the cache against. Accumulation runs base score first, then
    trees in index order, which is the order `train_more` already uses so
    that a resumed run lands on the same bits.
    """
    if len(weights) != len(trees):
        raise Error("dart needs one weight per tree")
    if len(base_scores) != n_classes:
        raise Error("dart needs one base score per class")
    if n_classes > 0 and len(trees) % n_classes != 0:
        raise Error("dart tree count must be a multiple of the class count")

    var n = data.n_rows
    var raw = List[Float64](capacity=n * n_classes)
    var n_iterations = len(trees) // n_classes if n_classes > 0 else 0
    for r in range(n):
        for k in range(n_classes):
            var s = base_scores[k]
            for i in range(n_iterations):
                var slot = _slot(i, k, n_classes)
                s += weights[slot] * trees[slot].predict_row(data, r)
            raw.append(s)
    return raw^


def dart_weights_are_uniform(weights: List[Float64]) -> Bool:
    """Whether every tree carries the same weight, so the ensemble is
    representable by today's single-scalar `Booster`.

    A DART run whose every round skipped lands here, and so does a
    zero-round or one-round ensemble. A serializer uses this to keep writing
    the compact scalar form when the vector carries nothing, and a loader of
    an older file uses the converse: a v4 model is a uniform-weight model
    whose scalar is its `learning_rate`.
    """
    if len(weights) <= 1:
        return True
    var first = weights[0]
    for i in range(1, len(weights)):
        if weights[i] != first:
            return False
    return True


def dart_uniform_weights(
    n_trees: Int, learning_rate: Float64
) -> List[Float64]:
    """The weight vector of a GBDT ensemble: `learning_rate` everywhere.

    This is how an existing `Booster` is lifted into the weighted
    representation, which continued training needs in order to add DART
    rounds to an ensemble that was not trained with them.
    """
    var out = List[Float64](capacity=n_trees)
    for _ in range(n_trees):
        out.append(learning_rate)
    return out^


struct DartBestState(Copyable, Movable):
    """The ensemble as it stood at the best validation round.

    Early stopping under DART cannot be done by truncation, and this struct
    is why. A GBDT ensemble truncated to its best round IS the model that
    scored best, because the trees kept are untouched by the rounds that were
    dropped. Under DART they are not: a round after the best one may have
    dropped and rescaled trees that the best round contained, so the weights
    left in place are not the weights that produced the best score. Popping
    trees off the end recovers the right tree set and the wrong model.

    So the weights are snapshotted whenever the validation loss improves.
    That costs one copy of a Float64 per tree per improvement, which is the
    smaller price: the alternative is keeping every round's vector, which is
    quadratic in the round count.

    `n_trees` is the tree count at the snapshot, so restoring is "truncate to
    `n_trees`, then overwrite the weights with `weights`".
    """

    var weights: List[Float64]
    var n_trees: Int
    var loss: Float64

    def __init__(
        out self, var weights: List[Float64], n_trees: Int, loss: Float64
    ):
        self.weights = weights^
        self.n_trees = n_trees
        self.loss = loss

    @staticmethod
    def initial(loss: Float64) -> DartBestState:
        """The state before any tree is grown: the base-score-only model."""
        return DartBestState(List[Float64](), 0, loss)


def dart_record_best(
    mut best: DartBestState, weights: List[Float64], loss: Float64,
    min_delta: Float64,
) -> Bool:
    """Snapshot the weights when `loss` improves on the best by more than
    `min_delta`, and report whether it did.

    The improvement test is `loss < best.loss - min_delta`, the same test
    `train_with_valid` already applies, so DART early stopping stops on the
    same rounds a GBDT run would for the same loss sequence.
    """
    if not (loss < best.loss - min_delta):
        return False
    best.loss = loss
    best.n_trees = len(weights)
    best.weights = weights.copy()
    return True


def dart_restore_best(
    best: DartBestState, mut trees: List[Tree], mut weights: List[Float64]
) raises:
    """Cut the ensemble back to the best round and restore its weights.

    Truncating alone would leave the surviving trees carrying weights later
    rounds rescaled; see `DartBestState`.
    """
    if best.n_trees > len(trees):
        raise Error("dart best-round snapshot is larger than the ensemble")
    if len(best.weights) != best.n_trees:
        raise Error("dart best-round snapshot is inconsistent")
    while len(trees) > best.n_trees:
        _ = trees.pop()
    weights.clear()
    for i in range(best.n_trees):
        weights.append(best.weights[i])
