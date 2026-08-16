"""`random_strength` under `grow_policy = oblivious`, on the device.

The parameter was refused on this path with one reason: **the noise is keyed
by node and a level has no node.** It has a DEPTH, and the depth is exactly
what CatBoost's key is standing in for -- `greedy_tensor_search.cpp:1199`
calls `CalcScores` inside the `curDepth` loop and `:884` draws a fresh seed
there, so the same candidate draws different noise at each depth, while the
standard deviation at `:1186` is drawn once per tree immediately before the
loop. So the level's draw is keyed by (seed, tree, **depth**, feature, bin)
in its own domain, and this file is what says the two backends key it the
same way.

WHY THE ANTI-VACUITY ASSERTIONS ARE THE POINT
---------------------------------------------
A cross-backend equality test proves two implementations are identical, which
is free when both are identically wrong. **If both backends dropped the depth
term, every equality assertion here would still pass** -- they would agree
perfectly on the same wrong draw. So three of the assertions below exist to
make that impossible:

  - `test_the_level_draw_moves_with_depth`: the draw at depth 0 differs from
    the draw at depth 5, everything else fixed. This is the one that fails if
    the depth term is dropped on either side.
  - `test_the_level_domain_is_disjoint_from_the_node_domain`: an oblivious
    level at depth `d` and a leaf-wise node `d` draw DIFFERENT values from the
    same (seed, tree, feature, bin). Without the second domain they would be
    the same number, which is harmless while a fit is one growth policy or the
    other and stops being harmless the moment anything compares the two.
  - `test_the_cosine_level_noise_lands_on_the_ratio`: the noised level record
    differs from the unnoised one by exactly the plane's Float32 word, which
    is only true if the addend landed on the level's aggregate score. Noise
    folded into `cos_num`/`cos_den` would scale with the level's width and
    fail this by a wide margin.

WHAT IS COMPARED AS VALUES AND WHAT AS WORDS
---------------------------------------------
Stage A -- key to counter -- is pure 64-bit integer arithmetic and is
compared as words, on the device against the host, exactly as
`tests/test_gpu_random_score_noise.mojo` does for the node domain. Stage B --
counter to normal, Marsaglia's polar method in Float64 -- runs on the host and
the plane is uploaded, because Apple GPUs have no Float64 at all; so the
CPU/GPU value comparison is between `tree_parameters_extra`'s own draw and the
Float32 the device consumes, which is one rounding of exactly that number.

The determinism half needs no accelerator and is asserted for both, since a
running generator reintroduced "as a fidelity improvement" is the failure mode
the counter-based key exists to make impossible.

The device tests skip (passing) with no accelerator.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.categorical import CategoricalParams, CategoricalSpec
from mojotrees.split import SCORE_COSINE, SCORE_L2
from mojotrees.tree_parameters_extra import (
    oblivious_score_noise,
    oblivious_score_stream,
    random_score_noise,
    random_score_stream,
)
from mojotrees.gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    OBLIVIOUS_SCORE_DOMAIN,
    RANDOM_SCORE_DOMAIN,
    gpu_oblivious_score_stream,
    gpu_random_score_stream,
    oblivious_score_key_probe,
    oblivious_score_plane,
)


# --- Fixtures -------------------------------------------------------------

comptime SEED = 20260816
comptime TREE = 7
comptime STDEV = 0.5


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _pair(a: List[Int], b: List[Int]) -> List[List[Int]]:
    """A two-leaf level's plane, leaf 0 first. The order is the summation
    order; see `gpu_resident_round.OBLIVIOUS_LEAF_INDEX_RULE`."""
    var out = List[List[Int]]()
    out.append(a.copy())
    out.append(b.copy())
    return out^


def _level_words(
    n_features: Int,
    n_bins: Int,
    g: List[List[Int]],
    h: List[List[Int]],
    c: List[List[Int]],
) raises -> List[Int32]:
    """`n_slots` consecutive `[grad | hess | count]` histograms, leaf 0 first.
    The same shape `tests/test_gpu_oblivious_cosine.mojo` builds."""
    var size = n_features * n_bins
    var n_slots = len(g)
    if len(h) != n_slots or len(c) != n_slots:
        raise Error("every leaf needs all three planes")
    var words = _zeroed(3 * size * n_slots)
    for s in range(n_slots):
        var base = s * 3 * size
        for i in range(size):
            words[base + i] = Int32(g[s][i])
            words[base + size + i] = Int32(h[s][i])
            words[base + 2 * size + i] = Int32(c[s][i])
    return words^


def _fixture() raises -> List[Int32]:
    """A two-leaf level over two features and four bins, both fixed-point
    scales 1.0, so every word dequantizes to itself and nothing below is
    about quantization."""
    return _level_words(
        2,
        4,
        _pair([-6, -3, 3, 6, -2, -1, 4, 5], [-5, -2, 2, 7, -3, -1, 3, 4]),
        _pair([2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2]),
        _pair([9, 9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9]),
    )


def _params() -> GpuSplitParams:
    """CatBoost's `l2_leaf_reg` default, for the reason the Cosine file
    gives: Cosine's whole difference from L2 is a function of it."""
    return GpuSplitParams(
        3.0, 0.0, 0.0, 0, CategoricalParams.default()
    )


def _search_level(
    words: List[Int32],
    score_function: Int,
    stdev: Float64,
    depth: Int,
) raises -> GpuSplitRecord:
    """One level search, with the noise on or off. `stdev <= 0` takes the
    default path, which stages no plane and moves no byte."""
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var searcher = GpuSplitSearcher(
            2, 4, List[Int](), CategoricalSpec.none(), 3
        )
        searcher.set_monotone(List[Int]())
        searcher.upload_level_histogram(words, 2)
        var slots = List[Int]()
        slots.append(0)
        slots.append(1)
        if stdev > 0.0:
            searcher.set_random_score(stdev, SEED, TREE)
            searcher.stage_random_score_level(0, depth)
        return searcher.search_oblivious_level(
            _params(),
            1.0,
            1.0,
            slots,
            level_record=0,
            leaf_base=1,
            score_function=score_function,
        )


# --- Half 1: the two backends draw the same number ------------------------


def test_the_device_level_key_is_the_host_level_key() raises:
    """Stage A, as 64-bit words, over a grid that moves every term.

    The assertion `random_score_key_probe` makes for the node domain, made
    for the level domain. If it holds and the plane is uploaded, the device
    and the host noise the same candidate by the same amount, because
    everything downstream of the key is host-computed."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var queries = List[Int]()
        var want = List[UInt64]()
        for depth in range(6):
            for f in range(3):
                for b in range(3):
                    queries.append(SEED)
                    queries.append(TREE)
                    queries.append(depth)
                    queries.append(f)
                    queries.append(b)
                    want.append(
                        oblivious_score_stream(SEED, TREE, depth, f, b)
                    )
        var got = oblivious_score_key_probe(queries)
        assert_equal(len(got), len(want))
        for i in range(len(want)):
            assert_equal(got[i], want[i])


def test_the_two_stage_a_copies_agree() raises:
    """`gpu_split_search`'s copy of stage A against
    `tree_parameters_extra`'s, which is the pair that has to stay byte for
    byte identical while both exist. Needs no accelerator: this is the host
    replica against the host original."""
    for depth in range(8):
        for f in range(4):
            for b in range(4):
                assert_equal(
                    gpu_oblivious_score_stream(SEED, TREE, depth, f, b),
                    oblivious_score_stream(SEED, TREE, depth, f, b),
                )
                assert_equal(
                    gpu_random_score_stream(SEED, TREE, depth, f, b),
                    random_score_stream(SEED, TREE, depth, f, b),
                )


def test_the_cpu_draw_is_the_value_the_gpu_plane_carries() raises:
    """The cross-backend comparison as VALUES, no tolerance.

    `oblivious_score_noise` is what the CPU grower adds, in Float64;
    `oblivious_score_plane` is what the device adds, in Float32. The plane's
    word must be exactly `Float32` of the CPU's number -- one rounding of the
    same product, and no second evaluation of stage B."""
    var features = List[Int]()
    features.append(0)
    features.append(1)
    for depth in range(4):
        var plane = oblivious_score_plane(
            STDEV, SEED, TREE, depth, features, 4
        )
        assert_equal(len(plane), 8)
        for slot in range(2):
            for b in range(4):
                var cpu = oblivious_score_noise(
                    STDEV, SEED, TREE, depth, features[slot], b
                )
                assert_equal(plane[slot * 4 + b], Float32(cpu))


# --- Half 2: the anti-vacuity assertions ----------------------------------


def test_the_level_draw_moves_with_depth() raises:
    """**The assertion that fails if either backend drops the depth term.**

    Half 1 passes trivially if both backends draw the same wrong thing, so
    this pins the property Half 1 cannot see: the same (seed, tree, feature,
    bin) draws a DIFFERENT number at depth 0 and at depth 5. That is
    CatBoost's behavior -- a fresh seed per `curDepth` -- reproduced
    counter-based, and it is the difference between `random_strength` being a
    per-level regularizer and being a single per-tree perturbation repeated
    six times."""
    var moved = 0
    for f in range(3):
        for b in range(4):
            var at0 = oblivious_score_noise(STDEV, SEED, TREE, 0, f, b)
            var at5 = oblivious_score_noise(STDEV, SEED, TREE, 5, f, b)
            assert_true(at0 != at5)
            assert_true(
                gpu_oblivious_score_stream(SEED, TREE, 0, f, b)
                != gpu_oblivious_score_stream(SEED, TREE, 5, f, b)
            )
            if at0 != at5:
                moved += 1
    # Every one of them, not merely one: a key that folded the depth in
    # weakly enough to collide on most candidates would still be broken.
    assert_equal(moved, 12)


def test_the_level_domain_is_disjoint_from_the_node_domain() raises:
    """An oblivious level at depth `d` and a leaf-wise node `d` must draw
    different numbers from the same (seed, tree, feature, bin).

    With one domain they would be the identical value, since a depth and a
    node id are both small nonnegative integers occupying the same key
    position. That is harmless while a fit is one growth policy or the other,
    and it stops being harmless the moment anything compares the two policies
    at a fixed seed. The second constant makes them disjoint as a property
    rather than as a convention."""
    assert_true(OBLIVIOUS_SCORE_DOMAIN != RANDOM_SCORE_DOMAIN)
    for site in range(6):
        for f in range(3):
            assert_true(
                oblivious_score_stream(SEED, TREE, site, f, 1)
                != random_score_stream(SEED, TREE, site, f, 1)
            )
            assert_true(
                oblivious_score_noise(STDEV, SEED, TREE, site, f, 1)
                != random_score_noise(STDEV, SEED, TREE, site, f, 1)
            )


# --- Half 3: the counter-based property itself ----------------------------


def test_the_level_draw_is_stable_across_calls() raises:
    """The same tuple, twice, is the same number.

    Trivial to assert and the whole point of a counter-based key: it reads no
    state that advances with evaluation order, so it cannot depend on how many
    draws preceded it, on which worker took it, or on `MOJOTREES_NUM_WORKERS`.
    A running generator reintroduced here as a fidelity improvement -- which
    is what CatBoost itself uses -- would fail this the moment two draws were
    taken in a different order, and would pass it if they happened to be taken
    in the same one, which is why the interleaved order below is not the
    ascending one."""
    var first = List[Float64]()
    for depth in range(4):
        for f in range(3):
            first.append(oblivious_score_noise(STDEV, SEED, TREE, depth, f, 2))
    # The same twelve draws, taken back to front and with unrelated draws
    # interleaved between them. A stream that advanced would be somewhere else
    # entirely by now.
    var i = 11
    while i >= 0:
        var depth = i // 3
        var f = i % 3
        _ = oblivious_score_noise(STDEV, SEED + 1, TREE + 1, 9, 9, 9)
        assert_equal(
            oblivious_score_noise(STDEV, SEED, TREE, depth, f, 2), first[i]
        )
        i -= 1


def test_a_restaged_level_plane_is_the_same_plane() raises:
    """The searcher's staging is a function of its key too: staging depth 3,
    then depth 1, then depth 3 again gives the third plane the first's
    values."""
    var features = List[Int]()
    features.append(0)
    features.append(1)
    var a = oblivious_score_plane(STDEV, SEED, TREE, 3, features, 4)
    _ = oblivious_score_plane(STDEV, SEED, TREE, 1, features, 4)
    var c = oblivious_score_plane(STDEV, SEED, TREE, 3, features, 4)
    assert_equal(len(a), len(c))
    for i in range(len(a)):
        assert_equal(a[i], c[i])


# --- The kernel: where the addend lands -----------------------------------


def test_the_l2_level_noise_shifts_the_level_gain() raises:
    """A small standard deviation cannot move the winner, so the noised
    record's gain must be the unnoised gain plus exactly the plane's word at
    the winning (slot, bin) -- one Float32 add, compared with no tolerance."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var words = _fixture()
        var plain = _search_level(words, SCORE_L2, 0.0, 2)
        assert_true(plain.found)
        # Small enough that no candidate can overtake the winner, and the
        # assertion below checks that rather than assuming it.
        var stdev = 1e-4
        var noisy = _search_level(words, SCORE_L2, stdev, 2)
        assert_true(noisy.found)
        assert_equal(noisy.feature, plain.feature)
        assert_equal(noisy.bin, plain.bin)
        var features = List[Int]()
        features.append(0)
        features.append(1)
        var plane = oblivious_score_plane(
            stdev, SEED, TREE, 2, features, 4
        )
        # `GpuSplitRecord.gain` is the kernel's Float32 widened to Float64, so
        # the comparison is narrowed back: the kernel performed exactly one
        # Float32 add and this reproduces exactly that add.
        assert_equal(
            Float32(noisy.gain),
            Float32(plain.gain) + plane[plain.feature * 4 + plain.bin],
        )


def test_the_cosine_level_noise_lands_on_the_ratio() raises:
    """The same statement under `score_function = Cosine`, which is the one
    that distinguishes the two candidate implementations.

    A level's Cosine score is `sum(num) / sqrt(sum(den))` minus the level's
    unsplit score: ONE ratio, taken after the leaf loop closes. The noise goes
    on that number. Folding it into `cos_num` or `cos_den` instead would be a
    different regularizer -- it would pass through the square root and scale
    with the level's width -- and it would look entirely correct in a diff.
    The difference between the noised and unnoised records being EXACTLY the
    plane's word is what says the addend landed after the ratio."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var words = _fixture()
        var plain = _search_level(words, SCORE_COSINE, 0.0, 4)
        assert_true(plain.found)
        var stdev = 1e-5
        var noisy = _search_level(words, SCORE_COSINE, stdev, 4)
        assert_true(noisy.found)
        assert_equal(noisy.feature, plain.feature)
        assert_equal(noisy.bin, plain.bin)
        var features = List[Int]()
        features.append(0)
        features.append(1)
        var plane = oblivious_score_plane(
            stdev, SEED, TREE, 4, features, 4
        )
        assert_equal(
            Float32(noisy.gain),
            Float32(plain.gain) + plane[plain.feature * 4 + plain.bin],
        )


def test_a_large_strength_can_move_the_level_winner() raises:
    """The noise reaches the argmax, not merely the reported gain.

    `SetBestScore` runs its argmax over the NOISED instances, so a large
    enough standard deviation must be able to elect a different candidate. If
    the addend were applied after the winner was chosen, every assertion above
    would still pass and this one would not."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var words = _fixture()
        var plain = _search_level(words, SCORE_L2, 0.0, 1)
        assert_true(plain.found)
        var moved = False
        for depth in range(6):
            var got = _search_level(words, SCORE_L2, 50.0, depth)
            if got.found and (
                got.feature != plain.feature or got.bin != plain.bin
            ):
                moved = True
        assert_true(moved)


def test_a_level_without_a_staged_plane_is_refused() raises:
    """`random_strength` on and no plane drawn for the level record is a
    refusal, not a search against zeros. The alternative announces itself as
    nothing at all."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var searcher = GpuSplitSearcher(
            2, 4, List[Int](), CategoricalSpec.none(), 3
        )
        searcher.set_monotone(List[Int]())
        searcher.upload_level_histogram(_fixture(), 2)
        searcher.set_random_score(STDEV, SEED, TREE)
        var slots = List[Int]()
        slots.append(0)
        slots.append(1)
        var raised = False
        try:
            _ = searcher.search_oblivious_level(
                _params(), 1.0, 1.0, slots, level_record=0, leaf_base=1
            )
        except:
            raised = True
        assert_true(raised)


def test_a_negative_depth_is_refused() raises:
    """A level's depth is nonnegative. A default standing in for one is the
    failure `ExtraTreeParams.needs_node_identity` exists to prevent, in the
    level's spelling."""
    var raised = False
    try:
        var features = List[Int]()
        features.append(0)
        _ = oblivious_score_plane(STDEV, SEED, TREE, -1, features, 4)
    except:
        raised = True
    assert_true(raised)


def test_the_default_level_search_is_untouched() raises:
    """`random_strength` off -- the default, and every LightGBM-mode fit --
    stages nothing, copies nothing, and the kernel takes the branch it took
    before this lane existed. Asserted as a value: the L2 and Cosine levels
    still elect what they elected."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var words = _fixture()
        var l2 = _search_level(words, SCORE_L2, 0.0, 0)
        var cos = _search_level(words, SCORE_COSINE, 0.0, 0)
        assert_true(l2.found)
        assert_true(cos.found)
        # Twice, to say the un-noised path is not reading a stale plane.
        var l2_again = _search_level(words, SCORE_L2, 0.0, 0)
        assert_equal(l2_again.feature, l2.feature)
        assert_equal(l2_again.bin, l2.bin)
        assert_equal(l2_again.gain, l2.gain)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
