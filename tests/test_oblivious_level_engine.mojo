"""The CPU level engine for `grow_policy = oblivious`.

`MOJOTREES_OBLIVIOUS_LEVEL_ENGINE` selects between two producers of one
level's leaf statistics, and nothing else in the grower differs between them:

- **on** (the default) is `oblivious_level.accumulate_level_stats`, CatBoost's
  arrangement -- fan out over candidate features, one contiguous pass over the
  level's kept documents per feature, every leaf of the level folded into a
  private `[slot][feature][bin]` stripe, and the level's other side derived by
  subtraction.
- **off** is the leaf-by-leaf builder that shipped, which walks the whole bin
  matrix once per leaf of the level.

So the tests here are about **agreement**, not about either arm's arithmetic:
the split chosen at every level, the shape of the tree, the leaf counts and
the node ids must be the same object either way, and the leaf values must
agree to the last few bits.

**A tolerance appears in this file and `bench/results/LANE_RULES.md` forbids
that by default, so here is the exemption and its reason.** The two arms are
declared NOT bit-identical in `docs/design/OBLIVIOUS.md` D4, for two reasons
that are both associativity: the leaf-wise builder folds per-row-block partial
sums while the level engine adds strictly in ascending document order, and the
two choose a different side of the level to build and therefore a different
side to subtract. Structure is asserted exactly; only the values carry a
tolerance, and it is 1e-9 relative against values of order 1.

Determinism within one arm is still asserted on `to_bits`, because that is a
property neither divergence touches: every cell of the level buffer is written
by exactly one task, in an order fixed by the arguments rather than by the
worker count.
"""

from std.os import setenv
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees import Tree, TreeParams, bin_equal_width, grow_tree
from mojotrees.growth_policy import GROW_OBLIVIOUS
from mojotrees.oblivious_level import level_engine_enabled


comptime _ENGINE = "MOJOTREES_OBLIVIOUS_LEVEL_ENGINE"
comptime _WORKERS = "MOJOTREES_NUM_WORKERS"


# ------------------------------------------------------------------ fixtures


def _dense(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major pseudo-random features, the shape `bin_equal_width` takes.
    """
    var out = List[Float64](capacity=n_rows * n_features)
    var state = UInt64(20260816)
    for _ in range(n_rows * n_features):
        state = state * 6364136223846793005 + 1442695040888963407
        out.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    return out^


def _grads(n_rows: Int, features: List[Float64]) -> List[Float64]:
    """A gradient with real structure in the first four features, so a level
    has something to disagree about and the winner is decided by the data
    rather than by a tie. Copied from `test_oblivious.mojo` on purpose: the
    two files must be arguing about the same trees."""
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(
            -(
                3.0 * features[r]
                - 2.0 * features[n_rows + r]
                + 1.5 * features[2 * n_rows + r] * features[3 * n_rows + r]
            )
        )
    return out^


def _ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


def _params(max_depth: Int) -> TreeParams:
    return TreeParams(
        3,
        1,
        1.0,
        1e-3,
        max_depth=max_depth,
        grow_policy=GROW_OBLIVIOUS,
    )


def _grow(n_rows: Int, n_features: Int, max_depth: Int) raises -> Tree:
    var features = _dense(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var grad = _grads(n_rows, features)
    return grow_tree(data, grad, _ones(n_rows), _params(max_depth))


def _grow_with(
    engine: String, n_rows: Int, n_features: Int, d: Int
) raises -> Tree:
    _ = setenv(_ENGINE, engine)
    var tree = _grow(n_rows, n_features, d)
    _ = setenv(_ENGINE, "")
    return tree^


def _close(a: Float64, b: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    var scale = a if a > 0.0 else -a
    return d <= 1e-9 * (1.0 + scale)


# --------------------------------------------------------------- the switch


def test_engine_default_is_on():
    """Unset means ON. This is the assertion that keeps the engine REACHED:
    if the default ever flips to off, every timing below still passes and
    nothing ships."""
    _ = setenv(_ENGINE, "")
    assert_true(level_engine_enabled())


def test_engine_words():
    _ = setenv(_ENGINE, "0")
    assert_true(not level_engine_enabled())
    _ = setenv(_ENGINE, "off")
    assert_true(not level_engine_enabled())
    _ = setenv(_ENGINE, "1")
    assert_true(level_engine_enabled())
    _ = setenv(_ENGINE, "on")
    assert_true(level_engine_enabled())
    _ = setenv(_ENGINE, "")


def test_engine_refuses_an_unknown_word():
    """A typo raises rather than silently selecting the path the caller was
    trying to leave."""
    _ = setenv(_ENGINE, "oon")
    with assert_raises():
        _ = level_engine_enabled()
    _ = setenv(_ENGINE, "")


# -------------------------------------------------------------- agreement


def test_engine_agrees_with_the_leafwise_builder():
    """Same splits, same shape, same node ids, same row counts, and leaf
    values that agree to 1e-9. Depth 4 over 400 rows is six levels' worth of
    the interesting case: several leaves per level, both sides of the level
    non-empty, and a level whose own smaller children are not all on the same
    side."""
    var fast = _grow_with(String("1"), 400, 6, 4)
    var slow = _grow_with(String("0"), 400, 6, 4)

    assert_equal(len(fast.feature), len(slow.feature))
    assert_equal(fast.n_leaves, slow.n_leaves)
    for n in range(len(fast.feature)):
        assert_equal(fast.feature[n], slow.feature[n])
        assert_equal(fast.threshold_bin[n], slow.threshold_bin[n])
        assert_equal(fast.left[n], slow.left[n])
        assert_equal(fast.right[n], slow.right[n])
        assert_true(fast.default_left[n] == slow.default_left[n])
        assert_true(_close(fast.count[n], slow.count[n]))
        assert_true(_close(fast.value[n], slow.value[n]))


def test_engine_agrees_at_depth_one():
    """The degenerate level. One leaf, one kept side, and the subtraction is
    the root minus that side, which is the case where the two arms have the
    fewest ways to differ and so the one that fails first if the slot
    arithmetic is wrong."""
    var fast = _grow_with(String("1"), 200, 4, 1)
    var slow = _grow_with(String("0"), 200, 4, 1)
    assert_equal(len(fast.feature), len(slow.feature))
    for n in range(len(fast.feature)):
        assert_equal(fast.feature[n], slow.feature[n])
        assert_equal(fast.threshold_bin[n], slow.threshold_bin[n])
        assert_true(_close(fast.value[n], slow.value[n]))


def test_engine_keeps_the_tree_symmetric():
    """The marker of an oblivious tree, re-asserted under the new producer.
    Without this the engine could be growing something else entirely and the
    agreement test above would only say the two arms broke identically."""
    var tree = _grow_with(String("1"), 400, 6, 4)
    var depths = List[Int](capacity=len(tree.feature))
    depths.resize(len(tree.feature), 0)
    for n in range(len(tree.feature)):
        if tree.feature[n] >= 0:
            depths[tree.left[n]] = depths[n] + 1
            depths[tree.right[n]] = depths[n] + 1
    var d_max = 0
    for n in range(len(depths)):
        if depths[n] > d_max:
            d_max = depths[n]
    for d in range(d_max + 1):
        var feature = -2
        var threshold = 0
        for n in range(len(tree.feature)):
            if depths[n] != d or tree.feature[n] < 0:
                continue
            if feature == -2:
                feature = tree.feature[n]
                threshold = tree.threshold_bin[n]
                continue
            assert_equal(tree.feature[n], feature)
            assert_equal(tree.threshold_bin[n], threshold)


# ------------------------------------------------------------ determinism


def test_engine_is_deterministic_across_worker_counts():
    """Bit-identical at 1, 3 and 8 workers. No tolerance here and none is
    owed: the divergence D4 declares is against the OTHER arm, not against
    another worker count of this one."""
    _ = setenv(_ENGINE, "1")
    _ = setenv(_WORKERS, "1")
    var one = _grow(400, 6, 4)
    _ = setenv(_WORKERS, "3")
    var three = _grow(400, 6, 4)
    _ = setenv(_WORKERS, "8")
    var eight = _grow(400, 6, 4)
    _ = setenv(_WORKERS, "")
    _ = setenv(_ENGINE, "")

    assert_equal(len(one.feature), len(three.feature))
    assert_equal(len(one.feature), len(eight.feature))
    for n in range(len(one.feature)):
        assert_equal(one.feature[n], three.feature[n])
        assert_equal(one.feature[n], eight.feature[n])
        assert_equal(one.threshold_bin[n], three.threshold_bin[n])
        assert_equal(one.threshold_bin[n], eight.threshold_bin[n])
        assert_equal(one.value[n].to_bits(), three.value[n].to_bits())
        assert_equal(one.value[n].to_bits(), eight.value[n].to_bits())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
