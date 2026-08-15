"""The remaining tree-parameter primitives (task 12).

Every expectation here is analytical: each case states the decision rule in
numbers small enough to check by hand, so a change in behavior shows up as a
changed rule rather than as a changed golden value.

    min_gain_to_split   gain > floor, strict
    max_delta_step      |output| <= step, sign kept
    path_smooth         out = raw * w/(w+1) + parent/(w+1),  w = n / smooth
    feature_contri      gain *= multiplier
    CEGB                see cegb.mojo; this file holds the parameters only
    extra_trees         one uniform threshold index per (tree, node, feature)
    monotone_penalty    the three-case depth factor
    forced splits       a validated four-key document, errors on anything else
"""

from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.cegb import cegb_split_cost
from mojotrees.tree_parameters_extra import (
    DEFAULT_EXTRA_SEED,
    MONOTONE_ADVANCED,
    MONOTONE_BASIC,
    MONOTONE_INTERMEDIATE,
    MONOTONE_PENALTY_EPSILON,
    ExtraTreeParams,
    FeaturePenalties,
    ForcedSplits,
    apply_monotone_penalty,
    cap_leaf_output,
    cat_effective_l2,
    cat_enters_search,
    cat_partition_gain,
    cat_side_cap,
    cat_sort_key,
    check_extra_option_supported,
    extra_candidate_index,
    extra_split_stream,
    extra_threshold_index,
    finish_leaf_output,
    monotone_method_name,
    monotone_penalty_factor,
    parse_forced_splits,
    parse_monotone_method,
    passes_min_gain,
    raw_leaf_output,
    smooth_leaf_output,
    split_gain_from_outputs,
)


def close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-12


# ---------------------------------------------------------------------------
# min_gain_to_split
# ---------------------------------------------------------------------------


def test_min_gain_floor_is_strict() raises:
    # A gain exactly at the floor is rejected, which is LightGBM's
    # `current_gain <= min_gain_shift`.
    assert_false(passes_min_gain(2.0, 2.0))
    assert_true(passes_min_gain(2.0000001, 2.0))
    assert_false(passes_min_gain(1.9999999, 2.0))


def test_min_gain_default_is_todays_rule() raises:
    # The default floor of 0.0 is what find_best_split already enforces: a
    # candidate must beat a best gain of 0.0.
    assert_true(passes_min_gain(1e-12, 0.0))
    assert_false(passes_min_gain(0.0, 0.0))
    assert_false(passes_min_gain(-1.0, 0.0))


# ---------------------------------------------------------------------------
# max_delta_step
# ---------------------------------------------------------------------------


def test_raw_leaf_output_is_the_newton_step() raises:
    # G = -4, H = 4, lambda_l2 = 1: -(-4) / 5 = 0.8.
    assert_true(close(raw_leaf_output(-4.0, 4.0, 0.0, 1.0), 0.8))
    # L1 soft-thresholds the gradient sum first: T(-4) = -3, so 3 / 5 = 0.6.
    assert_true(close(raw_leaf_output(-4.0, 4.0, 1.0, 1.0), 0.6))
    # A gradient sum inside the threshold collapses the leaf to zero.
    assert_true(close(raw_leaf_output(-4.0, 4.0, 5.0, 1.0), 0.0))


def test_max_delta_step_caps_magnitude_and_keeps_sign() raises:
    assert_true(close(cap_leaf_output(5.0, 2.0), 2.0))
    assert_true(close(cap_leaf_output(-5.0, 2.0), -2.0))
    # Exactly at the cap, and inside it, are untouched.
    assert_true(close(cap_leaf_output(2.0, 2.0), 2.0))
    assert_true(close(cap_leaf_output(-1.5, 2.0), -1.5))


def test_max_delta_step_zero_is_off() raises:
    assert_true(close(cap_leaf_output(1e6, 0.0), 1e6))
    assert_true(close(cap_leaf_output(-1e6, -1.0), -1e6))


# ---------------------------------------------------------------------------
# path_smooth
# ---------------------------------------------------------------------------


def test_path_smooth_zero_is_the_identity() raises:
    # The default must leave every leaf value exactly as it was, whatever the
    # parent holds.
    assert_true(close(smooth_leaf_output(1.0, 10, 0.0, 99.0), 1.0))
    assert_true(close(smooth_leaf_output(-3.0, 1, -1.0, 99.0), -3.0))


def test_path_smooth_half_weight_at_the_smoothing_size() raises:
    # n == path_smooth gives w = 1, so the leaf and its parent weigh the
    # same: (1.0 + 3.0) / 2 = 2.0.
    assert_true(close(smooth_leaf_output(1.0, 10, 10.0, 3.0), 2.0))


def test_path_smooth_fades_as_a_leaf_grows() raises:
    # w = 3: three parts leaf, one part parent.
    assert_true(close(smooth_leaf_output(1.0, 30, 10.0, 3.0), 1.5))
    # A leaf far larger than the smoothing size keeps almost all of itself.
    assert_true(abs(smooth_leaf_output(1.0, 1000000, 10.0, 3.0) - 1.0) < 1e-4)
    # An empty leaf is entirely its parent.
    assert_true(close(smooth_leaf_output(1.0, 0, 10.0, 3.0), 3.0))


def test_path_smooth_shrinks_the_root_toward_zero() raises:
    # The root has no parent, so its parent output is 0.0 and smoothing
    # shrinks it rather than pulling it anywhere: w = 1 halves it.
    assert_true(close(smooth_leaf_output(4.0, 10, 10.0, 0.0), 2.0))


def test_finish_caps_before_it_smooths() raises:
    # 5.0 capped to 2.0, then averaged with a parent output of 0.0.
    assert_true(close(finish_leaf_output(5.0, 2.0, 10.0, 10, 0.0), 1.0))
    # Smoothing first would give (5 + 6) / 2 = 5.5 and then cap to 2.0.
    # Capping first gives (2 + 6) / 2 = 4.0: the leaf never exceeds the cap
    # by more than its parent already does.
    assert_true(close(finish_leaf_output(5.0, 2.0, 10.0, 10, 6.0), 4.0))
    # Both off is the identity.
    assert_true(close(finish_leaf_output(5.0, 0.0, 0.0, 10, 6.0), 5.0))


def test_split_gain_from_outputs_matches_the_plain_formula() raises:
    # GL = -4, HL = 4, GR = 4, HR = 4, G = 0, lambda_l2 = 1. At the free
    # Newton outputs (+0.8, -0.8) this must reproduce 16/5 + 16/5 = 6.4, the
    # same number the ordinal scan computes for this node.
    var gain = split_gain_from_outputs(
        -4.0, 4.0, 0.8, 4.0, 4.0, -0.8, 1.0, 0.0
    )
    assert_true(close(gain, 6.4))


def test_split_gain_from_outputs_drops_when_outputs_are_forced() raises:
    # Forcing both children to 0.5 costs gain: those are no longer the
    # objective minimizers, so the split is worth less than 6.4.
    var free = split_gain_from_outputs(
        -4.0, 4.0, 0.8, 4.0, 4.0, -0.8, 1.0, 0.0
    )
    var capped = split_gain_from_outputs(
        -4.0, 4.0, 0.5, 4.0, 4.0, -0.5, 1.0, 0.0
    )
    assert_true(capped < free)
    # -(2 * -4 * 0.5 + 5 * 0.25) = 2.75 per side.
    assert_true(close(capped, 5.5))


# ---------------------------------------------------------------------------
# Per-feature penalties
# ---------------------------------------------------------------------------


def test_penalties_default_to_the_identity() raises:
    var neutral = FeaturePenalties()
    assert_false(neutral.is_active())
    assert_false(neutral.contri_active())
    assert_true(close(neutral.penalized_gain(7.0, 3), 7.0))
    assert_true(close(neutral.contri_of(3), 1.0))
    # The CEGB half lives on `cegb` (cegb.mojo) and is inactive too.
    assert_false(neutral.cegb.is_active())
    assert_true(close(neutral.cegb.coupled_of(3), 0.0))


def test_feature_contri_scales_one_feature_only() raises:
    var contri: List[Float64] = [1.0, 0.5, 0.0]
    var p = FeaturePenalties.from_contri(contri)
    assert_true(p.is_active())
    assert_true(close(p.penalized_gain(8.0, 0), 8.0))
    assert_true(close(p.penalized_gain(8.0, 1), 4.0))
    # A zero multiplier leaves the feature unable to clear any floor, which
    # switches it off by weight rather than by mask.
    assert_true(close(p.penalized_gain(8.0, 2), 0.0))
    assert_false(passes_min_gain(p.penalized_gain(8.0, 2), 0.0))
    # A feature past the end of the vector is unweighted.
    assert_true(close(p.penalized_gain(8.0, 9), 8.0))


def test_cegb_split_penalty_scales_with_leaf_rows() raises:
    # The costs live on `penalties.cegb` and are charged by cegb.mojo, so
    # these assert on the config this struct carries. The arithmetic itself
    # is tests/test_cegb.mojo's.
    var p = FeaturePenalties.from_cegb(2.0, 0.5)
    assert_true(p.is_active())
    assert_false(p.contri_active())
    assert_true(p.cegb.split_cost_active())
    assert_true(close(cegb_split_cost(p.cegb, 4), 4.0))
    assert_true(close(cegb_split_cost(p.cegb, 8), 8.0))
    # A tradeoff of zero switches the whole CEGB side off.
    var off = FeaturePenalties.from_cegb(0.0, 0.5)
    assert_false(off.is_active())
    assert_true(close(cegb_split_cost(off.cegb, 8), 0.0))
    # And the multiplier half is untouched by either.
    assert_true(close(p.penalized_gain(8.0, 0), 8.0))


def test_cegb_coupled_penalty_is_carried_not_charged_here() raises:
    var coupled: List[Float64] = [0.0, 3.0]
    var p = FeaturePenalties.from_cegb(2.0, 0.0, coupled)
    assert_true(p.is_active())
    assert_true(p.cegb.coupled_active())
    assert_true(p.cegb.needs_feature_ledger())
    assert_true(close(p.cegb.coupled_of(1), 3.0))
    assert_true(close(p.cegb.coupled_of(0), 0.0))
    # `penalized_gain` is the multiplier and nothing else, whatever the CEGB
    # costs say: charging them here as well is how they would be applied
    # twice.
    assert_true(close(p.penalized_gain(8.0, 1), 8.0))


def test_penalties_keep_the_multiplier_and_the_costs_apart() raises:
    var weights: List[Float64] = [0.5, 1.0]
    var p = FeaturePenalties.from_contri(weights)
    p.cegb.tradeoff = 1.0
    p.cegb.penalty_split = 1.0
    # The multiplier scales the gain: 10 * 0.5 = 5. The cost is subtracted
    # from that by cegb.mojo, at `split._feature_gain`, one call later:
    # 5 - 1 * 1 * 2 = 3.
    assert_true(close(p.penalized_gain(10.0, 0), 5.0))
    assert_true(
        close(p.penalized_gain(10.0, 0) - cegb_split_cost(p.cegb, 2), 3.0)
    )


def test_penalties_reject_unusable_vectors() raises:
    var has_negative: List[Float64] = [1.0, -1.0]
    var negative = FeaturePenalties.from_contri(has_negative)
    with assert_raises():
        negative.check_features(2)

    var two_ones: List[Float64] = [1.0, 1.0]
    var wrong_length = FeaturePenalties.from_contri(two_ones)
    with assert_raises():
        wrong_length.check_features(3)

    var bad_costs: List[Float64] = [1.0, -2.0]
    var bad_coupled = FeaturePenalties.from_cegb(1.0, 0.0, bad_costs)
    with assert_raises():
        bad_coupled.check_features(2)

    var bad_tradeoff = FeaturePenalties.from_cegb(-1.0, 0.0)
    with assert_raises():
        bad_tradeoff.check_features(2)

    var bad_lazy: List[Float64] = [1.0, -2.0]
    var lazy = FeaturePenalties.from_cegb(1.0, 0.0, [], bad_lazy)
    with assert_raises():
        lazy.check_features(2)

    # An empty bundle fits any width.
    FeaturePenalties().check_features(7)


# ---------------------------------------------------------------------------
# extra_trees
# ---------------------------------------------------------------------------


def test_extra_candidate_index_stays_in_range() raises:
    for n in range(1, 12):
        for node in range(6):
            for f in range(4):
                var i = extra_threshold_index(n, 7, 1, node, f)
                assert_true(i >= 0)
                assert_true(i < n)
    # No candidates means no split, not index 0.
    assert_equal(extra_threshold_index(0, 7, 1, 0, 0), -1)
    assert_equal(extra_threshold_index(-3, 7, 1, 0, 0), -1)
    # A single candidate is forced.
    assert_equal(extra_threshold_index(1, 7, 1, 0, 0), 0)


def test_extra_draw_is_reproducible_from_its_coordinates() raises:
    # The point of the counter-based stream: the same (seed, tree, node,
    # feature) draws the same threshold no matter what else ran.
    var a = extra_threshold_index(64, 11, 3, 5, 2)
    var b = extra_threshold_index(64, 11, 3, 5, 2)
    assert_equal(a, b)
    assert_equal(
        extra_split_stream(11, 3, 5, 2), extra_split_stream(11, 3, 5, 2)
    )


def test_extra_draw_separates_seed_tree_node_and_feature() raises:
    var base = extra_split_stream(11, 3, 5, 2)
    assert_true(extra_split_stream(12, 3, 5, 2) != base)
    assert_true(extra_split_stream(11, 4, 5, 2) != base)
    assert_true(extra_split_stream(11, 3, 6, 2) != base)
    assert_true(extra_split_stream(11, 3, 5, 3) != base)
    # Negative seeds are accepted, as they are for feature subsampling.
    var i = extra_threshold_index(16, -5, 0, 0, 0)
    assert_true(i >= 0)
    assert_true(i < 16)


def test_extra_draw_spreads_over_the_candidates() raises:
    # Not a distribution test: proof that one feature's draws move with the
    # node, so a tree does not force the same threshold everywhere.
    var seen_low = False
    var seen_high = False
    for node in range(64):
        var i = extra_threshold_index(8, DEFAULT_EXTRA_SEED, 0, node, 0)
        if i < 4:
            seen_low = True
        else:
            seen_high = True
    assert_true(seen_low)
    assert_true(seen_high)


def test_extra_candidate_index_uses_the_whole_stream() raises:
    # extra_candidate_index is the seedless half of the rule, so a caller
    # holding its own stream gets the same index the keyed helper does.
    var stream = extra_split_stream(2, 0, 0, 0)
    assert_equal(
        extra_candidate_index(32, stream),
        extra_threshold_index(32, 2, 0, 0, 0),
    )


# ---------------------------------------------------------------------------
# Categorical arithmetic
# ---------------------------------------------------------------------------


def test_cat_effective_l2_adds_cat_l2() raises:
    assert_true(close(cat_effective_l2(1.0, 10.0), 11.0))
    assert_true(close(cat_effective_l2(1.0, 0.0), 1.0))


def test_cat_sort_key_is_smoothed_gradient_over_hessian() raises:
    assert_true(close(cat_sort_key(2.0, 3.0, 10.0), 2.0 / 13.0))
    # cat_smooth = 0 leaves the raw ratio.
    assert_true(close(cat_sort_key(2.0, 4.0, 0.0), 0.5))


def test_cat_entry_filter_uses_exact_counts() raises:
    assert_false(cat_enters_search(9, 10.0))
    assert_true(cat_enters_search(10, 10.0))
    assert_true(cat_enters_search(0, 0.0))


def test_cat_side_cap_never_claims_more_than_half() raises:
    # 5 usable categories: a side may hold 3 at most, whatever the parameter.
    assert_equal(cat_side_cap(5, 32), 3)
    assert_equal(cat_side_cap(100, 32), 32)
    assert_equal(cat_side_cap(2, 32), 1)
    assert_equal(cat_side_cap(0, 32), 0)
    assert_equal(cat_side_cap(100, -1), 0)


def test_cat_partition_gain_scores_children_with_cat_l2() raises:
    # GL = -4, HL = 4, GR = 4, HR = 4, parent G = 0 so parent_score = 0.
    # cat_l2 = 0: 16/5 + 16/5 = 6.4, the same as the ordinal scan.
    var plain = cat_partition_gain(
        -4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0
    )
    assert_true(close(plain, 6.4))
    # cat_l2 = 1 pushes the children's L2 to 2: 16/6 + 16/6.
    var regularized = cat_partition_gain(
        -4.0, 4.0, 4.0, 4.0, 0.0, 1.0, 1.0, 0.0
    )
    assert_true(close(regularized, 32.0 / 6.0))
    assert_true(regularized < plain)


# ---------------------------------------------------------------------------
# Monotone penalty and method
# ---------------------------------------------------------------------------


def test_monotone_penalty_zero_is_the_identity() raises:
    assert_true(close(monotone_penalty_factor(0, 0.0), 1.0))
    assert_true(close(monotone_penalty_factor(5, 0.0), 1.0))


def test_monotone_penalty_small_penalization_halves_by_depth() raises:
    # p <= 1: factor = 1 - p / 2^depth, so the discount fades with depth.
    assert_true(close(monotone_penalty_factor(0, 0.5), 0.5))
    assert_true(close(monotone_penalty_factor(1, 0.5), 0.75))
    assert_true(close(monotone_penalty_factor(2, 0.5), 0.875))


def test_monotone_penalty_large_penalization_forbids_shallow_splits() raises:
    # p >= depth + 1 forbids the split outright: a tiny positive factor
    # rather than an exact zero, as in LightGBM.
    var eps = MONOTONE_PENALTY_EPSILON
    assert_true(close(monotone_penalty_factor(0, 1.0), eps))
    assert_true(close(monotone_penalty_factor(1, 2.0), eps))
    assert_true(close(monotone_penalty_factor(2, 3.0), eps))
    # Below that boundary: 1 - 2^(p - 1 - depth). With p = 2 the first depth
    # that allows a constrained split at all is 2, at half its gain, and the
    # discount halves again with each further level.
    assert_true(close(monotone_penalty_factor(2, 2.0), 0.5))
    assert_true(close(monotone_penalty_factor(3, 2.0), 0.75))
    assert_true(close(monotone_penalty_factor(4, 2.0), 0.875))


def test_monotone_penalty_is_monotone_in_depth() raises:
    # A constrained split gets cheaper the deeper it sits, which is the
    # intent: forbid near the root, allow further down.
    var previous = monotone_penalty_factor(1, 2.0)
    for depth in range(2, 8):
        var factor = monotone_penalty_factor(depth, 2.0)
        assert_true(factor > previous)
        previous = factor


def test_monotone_penalty_only_touches_constrained_features() raises:
    # sign 0 is never penalized, whatever the depth or the penalization, so
    # an unconstrained model is unaffected by the parameter.
    assert_true(close(apply_monotone_penalty(9.0, 0, 0, 4.0), 9.0))
    # A constrained feature at depth 0 with p = 0.5 keeps half its gain.
    assert_true(close(apply_monotone_penalty(9.0, 1, 0, 0.5), 4.5))
    assert_true(close(apply_monotone_penalty(9.0, -1, 0, 0.5), 4.5))
    # No penalization is the identity for constrained features too.
    assert_true(close(apply_monotone_penalty(9.0, 1, 0, 0.0), 9.0))


def test_monotone_method_names_round_trip() raises:
    assert_equal(parse_monotone_method("basic"), MONOTONE_BASIC)
    assert_equal(monotone_method_name(MONOTONE_BASIC), String("basic"))
    assert_equal(
        monotone_method_name(MONOTONE_INTERMEDIATE), String("intermediate")
    )
    assert_equal(
        monotone_method_name(MONOTONE_ADVANCED), String("advanced")
    )


def test_monotone_method_rejects_the_unimplemented_ones_by_name() raises:
    # These are real LightGBM methods, so they are refused rather than
    # silently downgraded to basic.
    with assert_raises():
        _ = parse_monotone_method("intermediate")
    with assert_raises():
        _ = parse_monotone_method("advanced")
    with assert_raises():
        _ = parse_monotone_method("Basic")
    with assert_raises():
        _ = parse_monotone_method("")


# ---------------------------------------------------------------------------
# Forced splits
# ---------------------------------------------------------------------------


def one_forced_node() -> String:
    return String("{\"feature\": 3, \"threshold\": 0.5}")


def test_forced_splits_empty_document_is_no_constraint() raises:
    assert_true(parse_forced_splits("").is_empty())
    assert_true(parse_forced_splits("   \n\t ").is_empty())
    assert_true(ForcedSplits.none().is_empty())
    assert_equal(ForcedSplits.none().depth(), 0)


def test_forced_splits_single_node() raises:
    var forced = parse_forced_splits(one_forced_node())
    assert_equal(forced.n_nodes(), 1)
    assert_equal(forced.depth(), 0)
    assert_equal(forced.nodes[0].feature, 3)
    assert_true(close(forced.nodes[0].threshold, 0.5))
    # Both sides are open, so leaf-wise growth resumes on each.
    assert_equal(forced.nodes[0].left, -1)
    assert_equal(forced.nodes[0].right, -1)


def test_forced_splits_nest_and_keep_parents_first() raises:
    var spec = String(
        "{\"feature\": 0, \"threshold\": 1.5,",
        " \"left\": {\"feature\": 1, \"threshold\": -2.25,",
        " \"right\": {\"feature\": 2, \"threshold\": 3e2}},",
        " \"right\": {\"feature\": 4, \"threshold\": 0.0}}",
    )
    var forced = parse_forced_splits(spec)
    assert_equal(forced.n_nodes(), 4)
    # A parent is always appended before its children, so the structure is
    # acyclic by construction.
    assert_equal(forced.nodes[0].feature, 0)
    var left = forced.nodes[0].left
    var right = forced.nodes[0].right
    assert_true(left > 0)
    assert_true(right > 0)
    assert_equal(forced.nodes[left].feature, 1)
    assert_true(close(forced.nodes[left].threshold, -2.25))
    var grandchild = forced.nodes[left].right
    assert_true(grandchild > left)
    assert_equal(forced.nodes[grandchild].feature, 2)
    assert_true(close(forced.nodes[grandchild].threshold, 300.0))
    assert_equal(forced.nodes[right].feature, 4)
    # Two edges from the root to the deepest forced node.
    assert_equal(forced.depth(), 2)


def test_forced_splits_check_features() raises:
    var forced = parse_forced_splits(one_forced_node())
    forced.check_features(4)
    with assert_raises():
        forced.check_features(3)


def test_forced_splits_check_budget() raises:
    # Two forced nodes need three leaves and reach depth 1, so a tree grown
    # under them needs two edges of depth.
    var spec = String(
        "{\"feature\": 0, \"threshold\": 1.0,",
        " \"left\": {\"feature\": 1, \"threshold\": 2.0}}",
    )
    var forced = parse_forced_splits(spec)
    assert_equal(forced.n_nodes(), 2)
    assert_equal(forced.depth(), 1)
    forced.check_budget(3, -1)
    forced.check_budget(31, 2)
    with assert_raises():
        forced.check_budget(2, -1)
    with assert_raises():
        forced.check_budget(31, 1)
    # No forced splits never constrains anything.
    ForcedSplits.none().check_budget(2, 1)


def test_forced_splits_reject_unknown_keys() raises:
    # An unknown key is an error, matching parse_params: a typo must not
    # silently drop a level of the forced tree.
    with assert_raises():
        _ = parse_forced_splits("{\"feature\": 0, \"thresh\": 0.5}")
    # A categorical forced split is named rather than reported as unknown.
    with assert_raises():
        _ = parse_forced_splits(
            "{\"feature\": 0, \"threshold\": 0.5, \"cat_threshold\": 1}"
        )


def test_forced_splits_require_feature_and_threshold() raises:
    with assert_raises():
        _ = parse_forced_splits("{\"feature\": 0}")
    with assert_raises():
        _ = parse_forced_splits("{\"threshold\": 0.5}")
    with assert_raises():
        _ = parse_forced_splits("{}")
    # A feature must be a nonnegative whole number.
    with assert_raises():
        _ = parse_forced_splits("{\"feature\": 0.5, \"threshold\": 0.5}")
    with assert_raises():
        _ = parse_forced_splits("{\"feature\": -1, \"threshold\": 0.5}")


def test_forced_splits_reject_malformed_documents() raises:
    with assert_raises():
        _ = parse_forced_splits("{\"feature\": 0, \"threshold\": 0.5")
    with assert_raises():
        _ = parse_forced_splits("{\"feature\": 0, \"threshold\": 0.5}}")
    with assert_raises():
        _ = parse_forced_splits("{\"feature\": 0, \"threshold\": }")
    with assert_raises():
        _ = parse_forced_splits("{feature: 0, \"threshold\": 0.5}")
    with assert_raises():
        _ = parse_forced_splits(
            "{\"feature\": 0, \"threshold\": 0.5, \"left\": 2}"
        )


# ---------------------------------------------------------------------------
# Rejections and the bundle
# ---------------------------------------------------------------------------


def test_deferred_options_are_rejected_by_name() raises:
    # `feature_pre_filter` is a Dataset construction step mojotrees does not
    # perform, so the name is refused rather than ignored.
    with assert_raises():
        check_extra_option_supported("feature_pre_filter")
    # Forced splits are implemented and reachable from the Mojo API, but a
    # parameter string cannot carry a document, so every LightGBM spelling of
    # the key is refused with that path named.
    with assert_raises():
        check_extra_option_supported("forcedsplits_filename")
    with assert_raises():
        check_extra_option_supported("forced_splits_filename")
    with assert_raises():
        check_extra_option_supported("forced_splits")
    with assert_raises():
        check_extra_option_supported("fs")
    # An implemented name passes through; this checker speaks only to the
    # options it refuses. All four cegb_* names are implemented now
    # (cegb.mojo); the two that need the per-ensemble ledger are refused per
    # grower, by `cegb.check_cegb_grower_support`, not by name here.
    # `linear_tree` and `linear_lambda` are implemented as well, as
    # `BoosterParams.linear` (linear_tree.mojo) rather than as tree controls,
    # so they pass through here and `params.parse_params` accepts them.
    check_extra_option_supported("path_smooth")
    check_extra_option_supported("cegb_penalty_split")
    check_extra_option_supported("cegb_penalty_feature_coupled")
    check_extra_option_supported("cegb_penalty_feature_lazy")
    check_extra_option_supported("linear_tree")
    check_extra_option_supported("linear_lambda")


def test_defaults_are_lightgbms_and_are_inactive() raises:
    var params = ExtraTreeParams.default()
    assert_true(close(params.min_gain_to_split, 0.0))
    assert_true(close(params.max_delta_step, 0.0))
    assert_true(close(params.path_smooth, 0.0))
    assert_false(params.extra_trees)
    assert_equal(params.extra_seed, DEFAULT_EXTRA_SEED)
    assert_equal(params.extra_seed, 6)
    assert_true(close(params.monotone_penalty, 0.0))
    assert_equal(params.monotone_method, MONOTONE_BASIC)
    assert_true(params.forced.is_empty())
    # The contract the growers rely on: untouched means unchanged.
    assert_false(params.is_active())
    params.check(4, 31, -1, 20)


def test_each_control_makes_the_bundle_active() raises:
    var gain_floor = ExtraTreeParams()
    gain_floor.min_gain_to_split = 0.1
    assert_true(gain_floor.is_active())

    var delta = ExtraTreeParams()
    delta.max_delta_step = 1.0
    assert_true(delta.is_active())

    var smooth = ExtraTreeParams()
    smooth.path_smooth = 1.0
    assert_true(smooth.is_active())

    var extra = ExtraTreeParams()
    extra.extra_trees = True
    assert_true(extra.is_active())

    var penalty = ExtraTreeParams()
    penalty.monotone_penalty = 1.0
    assert_true(penalty.is_active())

    var half: List[Float64] = [1.0, 0.5]
    var contri = ExtraTreeParams()
    contri.penalties = FeaturePenalties.from_contri(half)
    assert_true(contri.is_active())

    var forced = ExtraTreeParams()
    forced.forced = parse_forced_splits(one_forced_node())
    assert_true(forced.is_active())


def test_bundle_validation_rejects_rather_than_clamps() raises:
    var negative = ExtraTreeParams()
    negative.min_gain_to_split = -1.0
    with assert_raises():
        negative.check(4, 31, -1, 20)

    var delta = ExtraTreeParams()
    delta.max_delta_step = -1.0
    with assert_raises():
        delta.check(4, 31, -1, 20)

    # LightGBM raises min_data_in_leaf to 2 with a warning here; mojotrees
    # says so instead of changing a number the caller set.
    var smooth = ExtraTreeParams()
    smooth.path_smooth = 5.0
    with assert_raises():
        smooth.check(4, 31, -1, 1)
    smooth.check(4, 31, -1, 2)

    var method = ExtraTreeParams()
    method.monotone_method = MONOTONE_INTERMEDIATE
    with assert_raises():
        method.check(4, 31, -1, 20)

    var pair: List[Float64] = [1.0, 1.0]
    var wide = ExtraTreeParams()
    wide.penalties = FeaturePenalties.from_contri(pair)
    with assert_raises():
        wide.check(3, 31, -1, 20)

    var out_of_range = ExtraTreeParams()
    out_of_range.forced = parse_forced_splits(
        "{\"feature\": 9, \"threshold\": 1.0}"
    )
    with assert_raises():
        out_of_range.check(4, 31, -1, 20)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
