"""The sampling naming layer: one canonical name per sampling parameter.

`canonical_sampling_param`, `sampling_param_names` and
`canonical_bootstrap_type` are the one place a row or feature sampling
parameter's spellings are written down, so the Python layer, the CLI and
the C API resolve through them instead of each keeping a list. This file
holds them to the table in docs/PARAMETER_NAMING.md.

The load-bearing assertion is the fixed point at the end: every name
`canonical_sampling_param` *answers with* must be a name it also *accepts*.
A table that answered with a spelling it would then reject is the failure
mode a two-sided alias map has, and it cannot be caught by testing either
direction alone.

Run with `bash tools/run_tests.sh cpu test_canonical_sampling_names`.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.sampling import (
    canonical_bootstrap_type,
    canonical_sampling_param,
    is_sampling_param,
    sampling_param_names,
)


def test_the_row_share_is_one_key_under_every_spelling() raises:
    """LightGBM's `bagging_fraction` and CatBoost's `subsample` are the same
    number -- the share of rows a round keeps -- so they are one key.

    CatBoost has exactly one `subsample` option and Bernoulli, MVS and
    Poisson all read it; which sampler consumes it is decided by
    `bootstrap_type`, not by the spelling. A second key for MVS would have
    made `subsample=0.8` mean one thing under one bootstrap and nothing
    under another.
    """
    for spelling in [
        String("subsample"),
        String("bagging_fraction"),
        String("sub_row"),
        String("bagging"),
    ]:
        assert_equal(canonical_sampling_param(spelling), "subsample", spelling)
    for spelling in [String("subsample_freq"), String("bagging_freq")]:
        assert_equal(
            canonical_sampling_param(spelling), "subsample_freq", spelling
        )


def test_the_feature_shares_take_every_vendor_spelling() raises:
    """`colsample_bytree` and `colsample_bynode` are canonical;
    LightGBM's, CatBoost's and scikit-learn's words resolve onto them."""
    for spelling in [
        String("colsample_bytree"),
        String("feature_fraction"),
        String("sub_feature"),
        # CatBoost's, which is opaque enough to be the alias rather than the
        # canonical name.
        String("rsm"),
    ]:
        assert_equal(
            canonical_sampling_param(spelling), "colsample_bytree", spelling
        )
    for spelling in [
        String("colsample_bynode"),
        String("feature_fraction_bynode"),
        String("sub_feature_bynode"),
        # scikit-learn's HistGradientBoosting*.
        String("max_features"),
    ]:
        assert_equal(
            canonical_sampling_param(spelling), "colsample_bynode", spelling
        )
    for spelling in [
        String("colsample_bylevel"),
        String("feature_fraction_bylevel"),
    ]:
        assert_equal(
            canonical_sampling_param(spelling), "colsample_bylevel", spelling
        )


def test_the_names_only_one_vendor_has_keep_that_vendor_s_spelling() raises:
    """A parameter no other library has a word for is not renamed: there is
    nothing to reconcile it with, and renaming it would only cost a reader
    the ability to look it up in the vendor's own documentation."""
    for name in [
        # LightGBM's, and nobody else's.
        String("pos_bagging_fraction"),
        String("neg_bagging_fraction"),
        String("top_rate"),
        String("other_rate"),
        String("data_sample_strategy"),
        String("bagging_seed"),
        String("feature_fraction_seed"),
        # CatBoost's, and nobody else's.
        String("bootstrap_type"),
        String("bagging_temperature"),
        String("bootstrap_seed"),
        String("mvs_reg"),
    ]:
        assert_equal(canonical_sampling_param(name), name, name)


def test_every_answer_is_also_an_accepted_spelling() raises:
    """The fixed point.

    `canonical_sampling_param` maps into `sampling_param_names` and nowhere
    else, so every name it can return must resolve to itself. Without this
    a rename on one side of the table would leave the other side answering
    with a spelling it rejects, and every one-directional test would still
    pass.
    """
    var names = sampling_param_names()
    assert_true(len(names) > 0)
    for i in range(len(names)):
        assert_equal(canonical_sampling_param(names[i]), names[i], names[i])
        assert_true(is_sampling_param(names[i]), names[i])


def test_an_unknown_sampling_name_is_still_an_error() raises:
    """The table widens what is accepted; it does not make the resolver
    tolerant. A misspelling must not pass through as itself."""
    for unknown in [
        String("colsample_bytre"),
        String("subsamp"),
        String("bagging_frac"),
    ]:
        assert_true(not is_sampling_param(unknown), unknown)
        with assert_raises(contains="unknown sampling parameter"):
            _ = canonical_sampling_param(unknown)


def test_bootstrap_type_resolves_what_is_implemented() raises:
    """CatBoost's own parameter, keeping CatBoost's name and vocabulary.

    `No`, `Bayesian` and `MVS` resolve. `Bernoulli` is refused by name
    rather than as an unknown value, because mojotrees has it under another
    name -- it is row bagging, `subsample` with `subsample_freq`, which is
    the same draw CatBoost's Bernoulli `subsample` makes -- and telling a
    user that a real bootstrap type is unknown would be misleading.
    """
    assert_equal(canonical_bootstrap_type(String("no")), "no")
    assert_equal(canonical_bootstrap_type(String("none")), "no")
    assert_equal(canonical_bootstrap_type(String("bayesian")), "bayesian")
    assert_equal(canonical_bootstrap_type(String("mvs")), "mvs")
    with assert_raises(contains="subsample with subsample_freq"):
        _ = canonical_bootstrap_type(String("bernoulli"))
    # CatBoost itself refuses Poisson on the CPU.
    with assert_raises(contains="not implemented"):
        _ = canonical_bootstrap_type(String("poisson"))
    with assert_raises(contains="unknown bootstrap_type"):
        _ = canonical_bootstrap_type(String("bagged"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
