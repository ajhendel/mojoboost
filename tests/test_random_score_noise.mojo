"""`random_strength` on the device split search: the half that needs no GPU.

CatBoost's `random_strength` adds a seeded normal to every candidate's gain.
The feature is not the noise; the feature is that **the CPU and the GPU pick
the same split under the same seed**, and everything asserted here exists to
hold that property up.

The draw is two stages, and they are tested differently on purpose:

- **Stage A, the key.** (seed, tree, node, feature, bin) folded through
  splitmix64. Pure 64-bit integer arithmetic: no rounding, no libm, no
  multiply-add for a compiler to contract. This is the stage that has to be
  identical on the host, on Metal, and on CUDA, so it is pinned to literal
  words below. Literals are the right test for it precisely because there is
  no floating point in it: nothing about these numbers can move under a
  toolchain change. `tests/test_gpu_random_score_noise.mojo` runs the same
  function on an accelerator and compares the words.

- **Stage B, the normal.** Marsaglia's polar method in Float64. This one is
  **not** pinned to literals, and must not be. `u * u + v * v` is a
  multiply-add and a compiler may contract it; the standing rule in this
  repository is deterministic on a given toolchain, not identical to the
  past, so pinning a draw value would be pinning a contraction decision. What
  is asserted instead is what actually has to hold: that the draw is a
  function of the key alone, that it reproduces within a run, and that a zero
  strength touches nothing.

Nothing here opens a device. The replica (`reference_search`) is the same
Float32 arithmetic the kernels run, so the placement of the noise inside the
scan -- one draw per threshold, shared by both routing directions, added
after the admission tests and before any comparison -- is testable without
one.

WHEN THE HOST LANE MERGES. `tree_parameters_extra.random_score_stream` is the
same construction as `gpu_random_score_stream` and
`tree_parameters_extra.standard_normal` the same draw as
`host_standard_normal`. On a branch that has both, this file grows two lines
in `test_key_words_are_pinned`:

    from mojotrees.tree_parameters_extra import random_score_stream
    assert_equal(random_score_stream(s, t, n, f, b), gpu_random_score_stream(s, t, n, f, b))

and that is the whole cross-backend contract, because the device side is
already pinned to these words by the GPU file.
"""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.gpu_split_search import (
    GAIN_FORM_CROSS,
    GAIN_FORM_SUBTRACTIVE,
    GpuSplitParams,
    GpuSplitRecord,
    RANDOM_SCORE_DOMAIN,
    gpu_random_score_stream,
    host_random_score_noise,
    host_standard_normal,
    random_score_plane,
    reference_search,
)


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _histogram_words(
    n_features: Int, n_bins: Int, g: List[Int], h: List[Int], c: List[Int]
) raises -> List[Int32]:
    var size = n_features * n_bins
    if len(g) != size or len(h) != size or len(c) != size:
        raise Error("plane length must equal n_features * n_bins")
    var words = _zeroed(3 * size)
    for i in range(size):
        words[i] = Int32(g[i])
        words[size + i] = Int32(h[i])
        words[2 * size + i] = Int32(c[i])
    return words^


def _params(
    lambda_l2: Float64 = 1.0,
    lambda_l1: Float64 = 0.0,
    min_child_hess: Float64 = 0.0,
    min_data_in_leaf: Int = 0,
    cat: CategoricalParams = CategoricalParams.default(),
) -> GpuSplitParams:
    return GpuSplitParams(
        lambda_l2, lambda_l1, min_child_hess, min_data_in_leaf, cat.copy()
    )


def _one_categorical(n_features: Int, n_categories: Int) -> CategoricalSpec:
    var flags = List[Bool](capacity=n_features)
    var offsets = List[Int](capacity=n_features + 1)
    var codes = List[Int](capacity=n_categories)
    for i in range(n_categories):
        codes.append(i)
    offsets.append(0)
    for f in range(n_features):
        flags.append(f == 0)
        offsets.append(n_categories if f == 0 else offsets[f])
    return CategoricalSpec(flags^, codes^, offsets^)


def _two_feature_words(n_bins: Int) -> List[Int32]:
    """Two ordinal features whose best splits sit at different bins, so that
    a noise plane has something to reorder."""
    var size = 2 * n_bins
    var g = List[Int](capacity=size)
    var h = List[Int](capacity=size)
    var c = List[Int](capacity=size)
    for f in range(2):
        for b in range(n_bins):
            # Feature 0 changes sign a third of the way in, feature 1 two
            # thirds, so the two winners are different bins and neither is at
            # an end.
            var turn = n_bins // 3 if f == 0 else (2 * n_bins) // 3
            g.append(6 - 2 * b if b < turn else b - 3)
            h.append(2 + (b % 3))
            c.append(5 + (b % 4))
    try:
        return _histogram_words(2, n_bins, g, h, c)
    except:
        return _zeroed(3 * size)


# --- Stage A: the key ----------------------------------------------------
#
# The pinned vectors, as two flat lists rather than one list of pairs so that
# `tests/test_gpu_random_score_noise.mojo` can hand `KEY_QUERIES` straight to
# `random_score_key_probe`, which takes exactly this five-ints-per-query
# flattening. One table, two callers, no chance of the host and the device
# being pinned to different tuples.


def KEY_QUERIES() -> List[Int]:
    """(seed, tree, node, feature, bin), five ints per query."""
    return [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        -1, 0, 0, 0, 0,
        12345, 7, 3, 11, 255,
        2147483647, 1000, 4095, 999, 128,
    ]


def KEY_WORDS() -> List[UInt64]:
    """The 64-bit word `gpu_random_score_stream` returns for each query."""
    return [
        UInt64(0x3213C4E076A52544),
        UInt64(0x917B2B43C3F89F8B),
        UInt64(0xD33F729C02C59F7E),
        UInt64(0x7609B6B5EBEB5CB4),
        UInt64(0x7D5354DD6F600097),
        UInt64(0xD99866BAB1CAD2E4),
        UInt64(0xC48583B6C5CFB83C),
        UInt64(0x25D8B11246805639),
        UInt64(0xF3BDDAD7F0D492FA),
    ]


def test_key_words_are_pinned() raises:
    """The exact 64-bit words `gpu_random_score_stream` returns.

    This is the cross-backend contract. The device runs this same function
    and `tests/test_gpu_random_score_noise.mojo` compares it to these words,
    so if either side of the crossing ever stopped agreeing, one of the two
    files fails and names which.

    It is also the contract against the host scan: the construction is
    `tree_parameters_extra.random_score_stream`'s, byte for byte, and on a
    branch carrying both these words are what proves it.
    """
    var q = KEY_QUERIES()
    var want = KEY_WORDS()
    assert_equal(len(q), 5 * len(want))
    for i in range(len(want)):
        var b = 5 * i
        assert_equal(
            gpu_random_score_stream(
                q[b], q[b + 1], q[b + 2], q[b + 3], q[b + 4]
            ),
            want[i],
        )


def test_key_separates_every_component() raises:
    """All five components reach the key, and none of them is absorbed.

    The bin is mixed in as `bin + 1` and the node and feature likewise, so
    that index 0 is not the identity element of the xor; a component that had
    been dropped would show up here as two tuples sharing a word."""
    var base = gpu_random_score_stream(3, 4, 5, 6, 7)
    assert_not_equal(base, gpu_random_score_stream(4, 4, 5, 6, 7))
    assert_not_equal(base, gpu_random_score_stream(3, 5, 5, 6, 7))
    assert_not_equal(base, gpu_random_score_stream(3, 4, 6, 6, 7))
    assert_not_equal(base, gpu_random_score_stream(3, 4, 5, 7, 7))
    assert_not_equal(base, gpu_random_score_stream(3, 4, 5, 6, 8))
    # Bin 0 of feature 1 and bin 1 of feature 0 are different candidates and
    # must not collide, which is what the two `+ 1`s buy.
    assert_not_equal(
        gpu_random_score_stream(0, 0, 0, 1, 0),
        gpu_random_score_stream(0, 0, 0, 0, 1),
    )


def test_key_is_domain_separated() raises:
    """The domain separator is the ASCII bytes it claims to be, so a reader
    checking this stream against `extra_split_stream`'s can see at a glance
    that the two cannot coincide."""
    assert_equal(RANDOM_SCORE_DOMAIN, UInt64(0x52414E4453434F52))


# --- Stage B: the draw ---------------------------------------------------


def test_draw_is_a_function_of_the_key_alone() raises:
    """No hidden state, no counter that advances with call order.

    This is the property that makes the noise independent of
    `MOJOTREES_NUM_WORKERS`, of which feature ran on which task, and of how
    many candidates were scored before this one. Interleaving two streams and
    then replaying one of them alone has to give the same numbers."""
    var direct = List[Float64]()
    for b in range(8):
        direct.append(
            host_standard_normal(gpu_random_score_stream(11, 2, 3, 4, b))
        )
    var interleaved = List[Float64]()
    for b in range(8):
        _ = host_standard_normal(gpu_random_score_stream(99, 9, 9, 9, b))
        interleaved.append(
            host_standard_normal(gpu_random_score_stream(11, 2, 3, 4, b))
        )
        _ = host_standard_normal(gpu_random_score_stream(77, 7, 7, 7, b))
    for b in range(8):
        assert_equal(direct[b], interleaved[b])


def test_draw_is_not_constant_and_is_centered_about_zero() raises:
    """A normal, not a stuck value. Both signs appear, the magnitudes are in
    the range a unit normal lives in, and the mean of a few thousand draws is
    small. Loose bounds on purpose: this asserts the transform is a normal
    draw, not that it passes a distribution test, which is not what a unit
    test can honestly claim."""
    var n = 4096
    var total = 0.0
    var positives = 0
    var largest = 0.0
    for b in range(n):
        var x = host_standard_normal(gpu_random_score_stream(5, 0, 0, 0, b))
        total += x
        if x > 0.0:
            positives += 1
        if x > largest:
            largest = x
        elif -x > largest:
            largest = -x
    assert_true(positives > n // 4 and positives < (3 * n) // 4)
    assert_true(total / Float64(n) < 0.2 and total / Float64(n) > -0.2)
    assert_true(largest > 2.0 and largest < 8.0)


def test_noise_is_exactly_zero_at_the_default_strength() raises:
    """The default is a strict no-op: no stream is touched and the value is
    the literal zero, not a small number that rounds to it."""
    for b in range(16):
        assert_equal(
            host_random_score_noise(0.0, 1, 2, 3, 4, b), Float32(0.0)
        )
        assert_equal(
            host_random_score_noise(-1.0, 1, 2, 3, 4, b), Float32(0.0)
        )


def test_noise_scales_linearly_in_the_strength() raises:
    """`stdev` multiplies the draw and nothing else, so doubling it doubles
    every candidate's shift. This is what makes `random_strength` a single
    knob rather than a reseeding."""
    for b in range(8):
        var one = Float64(host_random_score_noise(1.0, 3, 1, 2, 5, b))
        var two = Float64(host_random_score_noise(2.0, 3, 1, 2, 5, b))
        assert_almost_equal(two, 2.0 * one, atol=1e-6, rtol=1e-6)


# --- The plane -----------------------------------------------------------


def test_plane_is_reproducible_bit_for_bit() raises:
    """Two calls, same bits. Determinism within a run is this project's
    standing rule and the plane is the whole of what the device sees."""
    var f = List[Int]([0, 1, 2, 3])
    var a = random_score_plane(0.5, 7, 2, 9, f, 16)
    var b = random_score_plane(0.5, 7, 2, 9, f, 16)
    assert_equal(len(a), 4 * 16)
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def test_plane_is_keyed_by_the_global_feature_id_not_the_slot() raises:
    """Reordering a node's feature set permutes the plane and changes no
    value in it, and narrowing it (which is what `feature_fraction_bynode`
    does) leaves the surviving features' rows alone.

    This is the property that keeps per-node feature sampling and
    `random_strength` from interfering: a feature's noise belongs to the
    feature, not to where it happened to land in the scan."""
    var n_bins = 8
    var straight = random_score_plane(
        0.75, 4, 1, 6, List[Int]([0, 1, 2]), n_bins
    )
    var permuted = random_score_plane(
        0.75, 4, 1, 6, List[Int]([2, 0, 1]), n_bins
    )
    for b in range(n_bins):
        assert_equal(permuted[0 * n_bins + b], straight[2 * n_bins + b])
        assert_equal(permuted[1 * n_bins + b], straight[0 * n_bins + b])
        assert_equal(permuted[2 * n_bins + b], straight[1 * n_bins + b])
    var narrowed = random_score_plane(0.75, 4, 1, 6, List[Int]([1]), n_bins)
    for b in range(n_bins):
        assert_equal(narrowed[b], straight[1 * n_bins + b])


def test_plane_separates_nodes_and_trees() raises:
    """Two nodes of one tree, and one node of two trees, draw different
    planes. A grower that failed to pass its node id would otherwise noise
    every node of a tree identically, which is the failure
    `ExtraTreeParams.needs_node_identity` exists to refuse."""
    var f = List[Int]([0, 1])
    var node0 = random_score_plane(1.0, 0, 0, 0, f, 8)
    var node1 = random_score_plane(1.0, 0, 0, 1, f, 8)
    var tree1 = random_score_plane(1.0, 0, 1, 0, f, 8)
    var same = 0
    for i in range(len(node0)):
        if node0[i] == node1[i]:
            same += 1
        if node0[i] == tree1[i]:
            same += 1
    assert_equal(same, 0)


def test_plane_refuses_a_missing_node_id() raises:
    """-1 is "not supplied" and is refused rather than treated as node 0."""
    with assert_raises(contains="node id"):
        _ = random_score_plane(1.0, 0, 0, -1, List[Int]([0]), 4)


# --- Placement inside the scan -------------------------------------------


def _assert_same_record(
    got: GpuSplitRecord, want: GpuSplitRecord
) raises:
    assert_equal(got.found, want.found)
    assert_equal(got.feature, want.feature)
    assert_equal(got.bin, want.bin)
    assert_equal(got.ordinal, want.ordinal)
    assert_equal(got.default_left, want.default_left)
    assert_equal(got.gain, want.gain)
    assert_equal(got.left.count, want.left.count)
    assert_equal(got.right.count, want.right.count)


def test_an_empty_plane_changes_no_bit() raises:
    """The default path. `reference_search` with no plane and
    `reference_search` with an all-zero plane return the identical record,
    and the all-zero plane is the strongest available statement that the
    arithmetic added is exactly `+ 0.0f` and not a reassociation."""
    var words = _two_feature_words(12)
    var plain = reference_search(words, 2, 12, 1.0, 1.0, _params())
    var zeros = List[Float32]()
    for _ in range(2 * 12):
        zeros.append(Float32(0.0))
    var zeroed = reference_search(
        words, 2, 12, 1.0, 1.0, _params(), noise=zeros
    )
    _assert_same_record(zeroed, plain)
    assert_true(plain.found)


def test_the_plane_shifts_the_winning_gain_by_exactly_its_own_value(
) raises:
    """The noise is additive on the gain and nothing else moves: the child
    statistics, the counts, and the routing of the same candidate are what
    they were, and the gain differs by the plane entry for the winner's
    (slot, bin)."""
    var n_bins = 12
    var words = _two_feature_words(n_bins)
    var plain = reference_search(words, 2, n_bins, 1.0, 1.0, _params())
    assert_true(plain.found)
    # A plane that is zero everywhere except the winner's own bin, so the
    # winner cannot change and the shift is readable directly.
    var plane = List[Float32]()
    for _ in range(2 * n_bins):
        plane.append(Float32(0.0))
    var shift = Float32(0.125)
    plane[plain.feature * n_bins + plain.bin] = shift
    var noised = reference_search(
        words, 2, n_bins, 1.0, 1.0, _params(), noise=plane
    )
    assert_equal(noised.feature, plain.feature)
    assert_equal(noised.bin, plain.bin)
    assert_equal(noised.ordinal, plain.ordinal)
    assert_equal(noised.left.count, plain.left.count)
    assert_almost_equal(
        noised.gain, plain.gain + Float64(shift), atol=1e-6, rtol=1e-6
    )


def test_the_plane_can_move_the_split_to_another_feature() raises:
    """The point of the regularizer: a large enough shift on a loser makes it
    the winner, and the record that comes back is that candidate's record,
    not the old winner's with a new gain."""
    var n_bins = 12
    var words = _two_feature_words(n_bins)
    var plain = reference_search(words, 2, n_bins, 1.0, 1.0, _params())
    assert_true(plain.found)
    var other = 1 - plain.feature
    var plane = List[Float32]()
    for _ in range(2 * n_bins):
        plane.append(Float32(0.0))
    # Enough to clear any gain this histogram can produce.
    for b in range(n_bins):
        plane[other * n_bins + b] = Float32(1000.0)
    var noised = reference_search(
        words, 2, n_bins, 1.0, 1.0, _params(), noise=plane
    )
    assert_true(noised.found)
    assert_equal(noised.feature, other)
    assert_true(noised.gain > plain.gain)


def test_the_shift_is_the_same_addend_under_both_gain_forms() raises:
    """The noise is defined against the *gain*, not against a gain form.

    `GAIN_FORM_CROSS` and `GAIN_FORM_SUBTRACTIVE` evaluate the same gain
    through different algebra and do not agree in their last bits, and the
    noise term is indifferent to that: both arms receive the identical
    Float32 addend, so a noised gain differs from an un-noised one by the
    same amount under either. Nothing about `random_strength` had to be
    recalibrated for the form that became the default this week."""
    var n_bins = 12
    var words = _two_feature_words(n_bins)
    var shift = Float32(0.0625)
    for arm in range(2):
        var form = GAIN_FORM_CROSS if arm == 0 else GAIN_FORM_SUBTRACTIVE
        var plain = reference_search(
            words, 2, n_bins, 1.0, 1.0, _params(), gain_form=form
        )
        assert_true(plain.found)
        var plane = List[Float32]()
        for _ in range(2 * n_bins):
            plane.append(Float32(0.0))
        plane[plain.feature * n_bins + plain.bin] = shift
        var noised = reference_search(
            words,
            2,
            n_bins,
            1.0,
            1.0,
            _params(),
            gain_form=form,
            noise=plane,
        )
        assert_equal(noised.bin, plain.bin)
        assert_almost_equal(
            noised.gain - plain.gain, Float64(shift), atol=1e-6, rtol=1e-3
        )


def test_l1_takes_the_subtractive_arm_and_the_noise_is_unaffected() raises:
    """`gpu_resolve_gain_form` sends `lambda_l1 != 0` back to the subtractive
    gain, because the cross form's identity needs `GL + GR = G` and soft
    thresholding is not additive. That refusal changes which gain the noise
    lands on and changes nothing about the noise itself, which is the whole
    content of "form-independent"."""
    var n_bins = 12
    var words = _two_feature_words(n_bins)
    var p = _params(lambda_l1=0.5)
    var plain = reference_search(words, 2, n_bins, 1.0, 1.0, p)
    assert_true(plain.found)
    var plane = List[Float32]()
    for _ in range(2 * n_bins):
        plane.append(Float32(0.0))
    var shift = Float32(0.25)
    plane[plain.feature * n_bins + plain.bin] = shift
    var noised = reference_search(words, 2, n_bins, 1.0, 1.0, p, noise=plane)
    assert_equal(noised.bin, plain.bin)
    assert_almost_equal(
        noised.gain - plain.gain, Float64(shift), atol=1e-6, rtol=1e-3
    )


def test_both_missing_directions_share_one_draw() raises:
    """The noise belongs to the threshold, not to the routing direction.

    A bin whose two directions tie keeps `default_left`, which is LightGBM's
    rule, and it keeps it *after* the noise because the same number was added
    to both. A plane whose entry for one bin is large moves both of that
    bin's candidates together, so the winner among them is decided by the
    un-noised comparison exactly as before."""
    var n_bins = 6
    var size = n_bins
    var g = List[Int](capacity=size)
    var h = List[Int](capacity=size)
    var c = List[Int](capacity=size)
    for b in range(n_bins):
        g.append(4 - b)
        h.append(2)
        c.append(3)
    var missing = List[Int]([n_bins - 1])
    var words = _histogram_words(1, n_bins, g, h, c)
    var plain = reference_search(
        words, 1, n_bins, 1.0, 1.0, _params(), missing_bins=missing
    )
    assert_true(plain.found)
    var plane = List[Float32]()
    for _ in range(n_bins):
        plane.append(Float32(0.0))
    plane[plain.bin] = Float32(2.0)
    var noised = reference_search(
        words,
        1,
        n_bins,
        1.0,
        1.0,
        _params(),
        missing_bins=missing,
        noise=plane,
    )
    # Same bin and the same routing direction: shifting both candidates of a
    # threshold by one number cannot reorder them against each other.
    assert_equal(noised.bin, plain.bin)
    assert_equal(noised.ordinal, plain.ordinal)
    assert_equal(noised.default_left, plain.default_left)


def test_a_categorical_feature_is_refused_rather_than_half_noised() raises:
    """A categorical candidate is a category *set* chosen inside the
    partition search, so only that search's winner would reach a
    per-candidate draw. Noising it would noise one candidate per categorical
    feature while every numerical feature had every candidate noised, which
    is a different regularizer wearing the same name.
    `split.find_best_split` refuses the combination and so does this."""
    var n_bins = 8
    var words = _two_feature_words(n_bins)
    var plane = List[Float32]()
    for _ in range(2 * n_bins):
        plane.append(Float32(0.0))
    with assert_raises(contains="numerical thresholds only"):
        _ = reference_search(
            words,
            2,
            n_bins,
            1.0,
            1.0,
            _params(),
            cats=_one_categorical(2, 4),
            noise=plane,
        )


def test_a_wrong_length_plane_is_refused() raises:
    """One Float32 per (active feature, bin), checked, because a plane that
    is silently short would noise some candidates and not others."""
    var words = _two_feature_words(8)
    var short = List[Float32]()
    for _ in range(8):
        short.append(Float32(0.0))
    with assert_raises(contains="one Float32 per"):
        _ = reference_search(words, 2, 8, 1.0, 1.0, _params(), noise=short)


def test_a_plane_from_random_score_plane_runs_end_to_end() raises:
    """The two halves fit together: a plane drawn by `random_score_plane` for
    a node is the right length and the right layout for the replica, and the
    search it produces is reproducible."""
    var n_bins = 12
    var words = _two_feature_words(n_bins)
    var plane = random_score_plane(
        0.5, 20260816, 3, 5, List[Int]([0, 1]), n_bins
    )
    var a = reference_search(
        words, 2, n_bins, 1.0, 1.0, _params(), noise=plane
    )
    var b = reference_search(
        words, 2, n_bins, 1.0, 1.0, _params(), noise=plane
    )
    _assert_same_record(a, b)
    assert_true(a.found)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
