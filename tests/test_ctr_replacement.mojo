"""CTR columns REPLACE their source column in CatBoost mode.

The ruling this file pins. In CatBoost mode there are no raw categorical
splits, exactly as in CatBoost: a categorical column above `one_hot_max_size`
is replaced by its CTR columns and dropped from `BinnedMatrix.usable`, so no
split search is ever offered it (`greedy_tensor_search.cpp:182` returns from
the one-hot candidate generator for such a column, and `:469` sends it to
`AddSimpleCtrs` instead). A column at or below `one_hot_max_size` stays
searchable and is one-hot, again as in CatBoost.

Two refusals were dissolved by that rather than worked around.

- `tree._check_oblivious_supported` refused any DECLARED categorical column.
  It now refuses a SEARCHABLE one (`BinnedMatrix.any_usable_categorical`), so a
  replaced column does not reach it.
- `split.find_best_split`'s `random_strength` refusal asked the same wrong
  question. It now asks whether the scan it is about to run is offered a
  categorical feature. It is not on the oblivious path at all -- a level's
  search is `find_best_split_shared`, which takes no `CategoricalSpec` -- so
  the fix there is for the leaf-wise path with a CatBoost-mode bundle.

**The deliverable is `test_symmetric_plus_categorical_plus_noise_trains`.**
That combination raised before this lane and must train, and it is the whole of
what the ruling claims.

`test_ctr_carries_the_signal` is the check that the shape has substance: the
label here is a function of the category and of nothing else, so a fit that
merely removed the column would score like a fit on noise. If replacement ever
scores like dropping, the CTR columns are occupying slots without carrying
anything.

`test_lossguide_ctr_still_accompanies` is the non-regression. Under our own
opt-in rule (`CTR_SOURCE_BIN_OVERFLOW`, what `SimpleCtrConfig.auto()` builds)
the raw column stays in the pool beside its CTR columns, because that is the
configuration that was measured -- average precision from 12.02 percent worse
than LightGBM to 3.63 percent better -- and removing the column would remove
what it was measured with.

No tolerance appears in any float assertion that pins a value: comparisons are
on `to_bits()`, per `bench/results/LANE_RULES.md`. The accuracy assertions are
inequalities between two losses, which is a comparison and not a tolerance.
"""

from std.math import log
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees import (
    BINARY_LOGISTIC,
    BinnedMatrix,
    BoosterParams,
    RawData,
    TreeParams,
)
from mojotrees.binning import ctr_extend_usable
from mojotrees.boosting import Booster, train
from mojotrees.categorical import (
    CategoricalSpec,
    any_searchable_categorical,
)
from mojotrees.ctr_columns import SimpleCtrConfig
from mojotrees.growth_policy import GROW_LEAFWISE, GROW_OBLIVIOUS
from mojotrees.trainset import Dataset
from mojotrees.tree_parameters_extra import ExtraTreeParams

# ------------------------------------------------------------------ fixtures

comptime _N_ROWS = 600
comptime _N_CATEGORIES = 40
"""Well past `one_hot_max_size = 2`, so CatBoost mode gives this column CTRs,
and well inside `_MAX_BIN`'s category table, so our own `auto()` rule does NOT
-- which is what keeps the two policies separable in one fixture."""

comptime _MAX_BIN = 64


def _values() -> List[Float64]:
    """Column-major, three columns.

    0: pseudo-random noise. 1: the categorical column, `_N_CATEGORIES` levels.
    2: more noise. The two noise columns exist so the split search has
    something to reject and so a fit with the categorical column removed still
    has features to grow on.
    """
    var out = List[Float64](capacity=_N_ROWS * 3)
    var state = UInt64(20260816)
    for _ in range(_N_ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        out.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    for r in range(_N_ROWS):
        out.append(Float64(_category_of(r)))
    for _ in range(_N_ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        out.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    return out^


def _offered(data: BinnedMatrix, feature: Int) -> Bool:
    """Whether `feature` is in the pool a split search draws from. The
    question the whole ruling turns on, asked of the matrix directly."""
    var usable = data.usable_features()
    for i in range(len(usable)):
        if usable[i] == feature:
            return True
    return False


def _category_of(r: Int) -> Int:
    # A stride that is coprime with the level count, so consecutive rows do
    # not share a category and the ordered statistic is not reading a block.
    return (r * 7) % _N_CATEGORIES


def _label_of(r: Int) -> Float64:
    """A 0/1 target that is a function of the CATEGORY and of nothing else.

    Not monotone in the code, so a fit that treated the column as an ordinary
    numeric feature could not recover it either; only a statistic keyed on the
    category can. That is what makes the accuracy comparison against dropping
    the column evidence rather than decoration.
    """
    var c = _category_of(r)
    return 1.0 if ((c * c) % 7) >= 4 else 0.0


def _labels() -> List[Float64]:
    var out = List[Float64](capacity=_N_ROWS)
    for r in range(_N_ROWS):
        out.append(_label_of(r))
    return out^


def _two_column_values() -> List[Float64]:
    """The same matrix with the categorical column DROPPED ENTIRELY: columns 0
    and 2 only, bit-identical to the ones above."""
    var full = _values()
    var out = List[Float64](capacity=_N_ROWS * 2)
    for r in range(_N_ROWS):
        out.append(full[r])
    for r in range(_N_ROWS):
        out.append(full[2 * _N_ROWS + r])
    return out^


def _catboost_dataset() raises -> Dataset:
    var cats: List[Int] = [1]
    return Dataset.from_raw(
        RawData.dense(_values(), _N_ROWS, 3),
        label=_labels(),
        categorical_features=cats^,
        max_bin=_MAX_BIN,
        ctr=SimpleCtrConfig.catboost_defaults(),
    )


def _dropped_dataset() raises -> Dataset:
    return Dataset.from_raw(
        RawData.dense(_two_column_values(), _N_ROWS, 2),
        label=_labels(),
        max_bin=_MAX_BIN,
        ctr=SimpleCtrConfig.disabled(),
    )


def _oblivious_noisy_params() -> BoosterParams:
    var ex = ExtraTreeParams()
    # The parameter the second refusal was about. `random_score_scale` is
    # ensemble state and `boosting._round_random_score_scale` writes it per
    # round, so setting the strength alone is the whole configuration.
    ex.random_strength = 1.0
    return BoosterParams(
        24,
        0.2,
        TreeParams(
            31,
            1,
            1.0,
            1e-3,
            max_depth=4,
            extra=ex^,
            grow_policy=GROW_OBLIVIOUS,
        ),
    )


def _log_loss(booster: Booster, data: BinnedMatrix, label: List[Float64]) -> (
    Float64
):
    """Mean binary log loss on the rows the model was fit on.

    Fit quality is the question here and a held-out split is not: the label is
    a deterministic function of the category, so the only way to score well is
    to have recovered that function, and a model that cannot see the category
    at all cannot memorize it either.
    """
    var total = 0.0
    for r in range(data.n_rows):
        var p = booster.predict_row(data, r)
        if p < 1e-12:
            p = 1e-12
        if p > 1.0 - 1e-12:
            p = 1.0 - 1e-12
        if label[r] > 0.5:
            total += -log(p)
        else:
            total += -log(1.0 - p)
    return total / Float64(data.n_rows)


def _digest(booster: Booster, data: BinnedMatrix) -> UInt64:
    """A bitwise digest of every prediction, mixed as `UInt64` words.

    `to_bits()` and never a tolerance: the point of the digest is to catch a
    draw that moved by one ulp between worker counts, and a tolerance is
    exactly the instrument that would not.
    """
    var h = UInt64(0xCBF29CE484222325)
    for r in range(data.n_rows):
        var bits = booster.predict_raw_row(data, r).to_bits().cast[
            DType.uint64
        ]()
        h = (h ^ bits) * UInt64(0x100000001B3)
        h ^= h >> 29
    return h


# ------------------------------------------------------------------- the pool


def test_catboost_mode_removes_the_source_column_from_usable() raises:
    var ds = _catboost_dataset()
    # It earned CTR columns at all.
    assert_true(ds.data.n_features > 3)
    assert_true(ds.mapper.ctr.is_active())

    # The raw categorical column is still THERE -- binned, predicted through,
    # holding its own id -- and is no longer OFFERED.
    assert_true(ds.data.cats.is_cat(1))
    assert_false(_offered(ds.data, 1))

    # Every CTR column is offered, and so are both numeric columns.
    assert_true(_offered(ds.data, 0))
    assert_true(_offered(ds.data, 2))
    for c in range(3, ds.data.n_features):
        assert_true(_offered(ds.data, c))

    # And the searchability predicate the guards now ask agrees.
    assert_true(ds.data.cats.any_categorical())
    assert_false(ds.data.any_usable_categorical())

    # The mapper carries the same removal, so a matrix transformed later is
    # offered the pool the trees were grown on.
    assert_false(ds.mapper.is_usable(1))


def test_symmetric_plus_categorical_plus_noise_trains() raises:
    """THE DELIVERABLE. Symmetric trees, a high-cardinality categorical
    column, `random_strength=1`. This raised at `tree.mojo`'s oblivious guard
    before the replacement landed, and both refusals it could have hit are
    correct as written; what changed is that neither is reached."""
    var ds = _catboost_dataset()
    var booster = train(
        ds.data, ds.label, BINARY_LOGISTIC, _oblivious_noisy_params()
    )
    assert_equal(len(booster.trees), 24)
    # It actually grew: an ensemble of stumps would satisfy the line above.
    var internal = 0
    for n in range(len(booster.trees[0].feature)):
        if booster.trees[0].feature[n] >= 0:
            internal += 1
    assert_true(internal > 1)


def test_the_guard_still_refuses_a_searchable_categorical() raises:
    """The refusal is dissolved, not deleted. With no CTR bundle the column is
    in the pool, the level has no one partition to share, and the fit is
    refused with the reason it always had."""
    var cats: List[Int] = [1]
    var ds = Dataset.from_raw(
        RawData.dense(_values(), _N_ROWS, 3),
        label=_labels(),
        categorical_features=cats^,
        max_bin=_MAX_BIN,
        ctr=SimpleCtrConfig.disabled(),
    )
    assert_true(ds.data.any_usable_categorical())
    with assert_raises(contains="grow_policy=oblivious"):
        _ = train(
            ds.data, ds.label, BINARY_LOGISTIC, _oblivious_noisy_params()
        )


def test_ctr_carries_the_signal() raises:
    """Replacement must score better than dropping the column entirely.

    If these two came out level, the CTR columns would be occupying feature
    slots without carrying the statistic they exist for, which is the failure
    mode a shape-only implementation has.
    """
    var kept = _catboost_dataset()
    var dropped = _dropped_dataset()
    var params = _oblivious_noisy_params()

    var b_kept = train(kept.data, kept.label, BINARY_LOGISTIC, params)
    var b_drop = train(dropped.data, dropped.label, BINARY_LOGISTIC, params)

    var loss_kept = _log_loss(b_kept, kept.data, kept.label)
    var loss_drop = _log_loss(b_drop, dropped.data, dropped.label)

    print("CTR_REPLACEMENT_LOGLOSS kept", loss_kept, "dropped", loss_drop)
    # The label is a function of the category alone, so the dropped fit has
    # nothing to learn from and sits near the entropy of the label; the kept
    # fit has the statistic. A margin rather than a tolerance: the assertion
    # is that one is materially lower, not that either is a value.
    assert_true(loss_kept < loss_drop * 0.9)


def test_lossguide_ctr_still_accompanies() raises:
    """Our own opt-in rule is untouched: the raw column stays in the pool.

    Built with `auto()` at a bin budget the category table cannot hold, which
    is the only shape `CTR_SOURCE_BIN_OVERFLOW` fires on. The assertion is that
    the source column is STILL usable beside its CTR columns, because that is
    the configuration the measured lossguide result came from.
    """
    var cats: List[Int] = [1]
    var ds = Dataset.from_raw(
        RawData.dense(_values(), _N_ROWS, 3),
        label=_labels(),
        categorical_features=cats^,
        max_bin=16,
        ctr=SimpleCtrConfig.auto(),
    )
    assert_true(ds.mapper.ctr.is_active())
    assert_true(ds.data.n_features > 3)
    assert_true(_offered(ds.data, 1))
    assert_true(ds.data.any_usable_categorical())
    assert_true(ds.mapper.is_usable(1))

    # And it still grows leaf-wise, which is the path that result was on.
    var params = BoosterParams(
        12, 0.2, TreeParams(15, 1, 1.0, 1e-3, grow_policy=GROW_LEAFWISE)
    )
    var booster = train(ds.data, ds.label, BINARY_LOGISTIC, params)
    assert_equal(len(booster.trees), 12)


def test_ctr_extend_usable_replaces_and_accompanies() raises:
    var usable: List[Int] = [0, 1, 2]
    var none: List[Int] = []
    var accompanied = ctr_extend_usable(usable, 3, 2, none)
    assert_equal(len(accompanied), 5)
    assert_equal(accompanied[1], 1)
    assert_equal(accompanied[3], 3)
    assert_equal(accompanied[4], 4)

    var drop: List[Int] = [1]
    var replaced = ctr_extend_usable(usable, 3, 2, drop)
    assert_equal(len(replaced), 4)
    assert_equal(replaced[0], 0)
    assert_equal(replaced[1], 2)
    assert_equal(replaced[2], 3)
    assert_equal(replaced[3], 4)

    # An id already outside the pool is satisfied, not refused.
    var short: List[Int] = [0, 2]
    var missing: List[Int] = [1]
    var still = ctr_extend_usable(short, 3, 1, missing)
    assert_equal(len(still), 3)
    assert_equal(still[0], 0)
    assert_equal(still[1], 2)
    assert_equal(still[2], 3)


def test_any_searchable_categorical_follows_the_scan() raises:
    var flags: List[Bool] = [False, True, False]
    var codes: List[Int] = [0, 1, 2, 3]
    var offsets: List[Int] = [0, 0, 4, 4]
    var spec = CategoricalSpec(flags^, codes^, offsets^)
    var all_features: List[Int] = []
    var no_mask: List[Bool] = []

    # Every feature scanned: the categorical one is reached.
    assert_true(
        any_searchable_categorical(spec, 3, all_features, no_mask)
    )
    # A feature list that omits it: not reached.
    var numeric_only: List[Int] = [0, 2]
    assert_false(
        any_searchable_categorical(spec, 3, numeric_only, no_mask)
    )
    # A mask that forbids it: not reached.
    var mask: List[Bool] = [True, False, True]
    assert_true(spec.any_categorical())
    assert_false(any_searchable_categorical(spec, 3, all_features, mask))


def test_determinism_digest() raises:
    """Printed rather than asserted against a constant.

    The comparison is between two RUNS at different `MOJOTREES_NUM_WORKERS`,
    which one process cannot make: the worker count is read when the fit
    starts. So this prints the digest and the runner compares the two lines,
    which is the same instrument as an assertion with the environment as the
    variable.
    """
    var ds = _catboost_dataset()
    var booster = train(
        ds.data, ds.label, BINARY_LOGISTIC, _oblivious_noisy_params()
    )
    print("CTR_REPLACEMENT_DIGEST", _digest(booster, ds.data))
    # A digest of zero would pass a cross-run comparison for the wrong reason.
    assert_true(_digest(booster, ds.data) != UInt64(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
