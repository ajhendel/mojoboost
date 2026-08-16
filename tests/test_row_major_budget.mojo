"""When the row-major view gets built, and when it is refused.

`tests/test_row_major_bins.mojo` asserts that the view carries the same bits
as the feature-major matrix. This file asserts the *policy* around it: the
default is `auto`, `auto` builds the view only when it fits a memory budget,
and `MOJOTREES_CPU_ROW_MAJOR` forces the answer either way.

The claim under test is a memory claim, so the shape of every assertion here
is "no bytes were allocated" rather than "the answer was slower". Nothing here
is a timing and nothing here may be read as evidence about speed.

What is covered, and why each case is here rather than assumed:

1. **The rule.** `row_major_fits_budget` is asserted *at* its boundary --
   exactly at the budget admits, one byte over refuses -- because a rule
   stated in a docstring and implemented with the wrong comparison would pass
   any test that only probed the middle. Its no-budget sentinel (`<= 0`) is
   asserted too, since that is the path `MOJOTREES_CPU_ROW_MAJOR=1` takes.

2. **The refusal allocates nothing and raises nothing.** Above the budget,
   `build_row_major` must leave the matrix exactly as `drop_row_major` leaves
   it: `has_row_major()` false, `row_major_bytes()` zero, and the
   feature-major matrix untouched, because that is what makes a refusal
   degrade to feature-major instead of failing a fit.

3. **A refusal after a build must destroy the old view, not keep it.** This is
   the trap in a method that both builds and rebuilds: a stale record array
   that survived a refused rebuild would be read as if it were current, and
   `has_row_major`'s size check would not catch it because the sizes still
   agree. Asserted directly.

4. **The budget is spent on the realized stride, not on `n_features`.** A
   matrix whose features all pack two to a byte is admitted by a budget that
   would refuse the same shape unpacked. Packing is the difference between
   admitted and refused in that case, which is the only way to prove the rule
   is not quietly rounding up to one byte per feature.

5. **The three states of `MOJOTREES_CPU_ROW_MAJOR`,** including that unset is
   `auto` and is *not* `0`, and that an unrecognized value raises rather than
   silently meaning `auto`.

6. **End to end through `transform`,** which is where the policy is actually
   applied: unset builds the view, `0` does not, and a budget of one mebibyte
   with the mode left at `auto` does not. A test of the predicate alone would
   not have caught a `transform` that never consulted it.

Every fixture here is a few hundred rows. The refusals are asserted with tiny
budgets rather than by allocating the gigabyte being refused -- the rule is a
comparison, and a comparison is testable at any scale.
"""

from std.os import getenv, setenv
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.binning import (
    BinnedMatrix,
    ROW_MAJOR_AUTO,
    ROW_MAJOR_DEFAULT_BUDGET_MB,
    ROW_MAJOR_OFF,
    ROW_MAJOR_ON,
    ROW_MAJOR_PACK_MAX_BINS,
    env_row_major_mode,
    fit_bins,
    row_major_budget_bytes,
    row_major_fits_budget,
)


def _matrix(n_rows: Int, widths: List[Int], n_bins: Int) raises -> BinnedMatrix:
    """A binned matrix whose feature f realizes exactly `widths[f]` bins.

    The same fixture `tests/test_row_major_bins.mojo` uses, and for the same
    reason: the row index sweeps the residues one at a time, so the realized
    width is the declared one and the packing decision is about the number
    this file says it is about.
    """
    var nf = len(widths)
    var bins = List[UInt8](capacity=n_rows * nf)
    bins.resize(n_rows * nf, UInt8(0))
    for f in range(nf):
        var w = widths[f]
        for r in range(n_rows):
            bins[f * n_rows + r] = UInt8((r + 7 * f) % w)
    return BinnedMatrix(bins^, n_rows, nf, n_bins)


def _clear_env() raises:
    """Both variables back to unset, which is the shipped default.

    An empty value reads as unset everywhere in this path (`getenv` returns
    an empty string for both, and `byte_length() == 0` is the unset test), so
    this is a restore and not an approximation of one.
    """
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "")
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR_MAX_MB", "")


def test_fits_budget_boundary() raises:
    """Exactly at the budget is admitted, one byte over is refused."""
    assert_true(
        row_major_fits_budget(100, 8, 800), "exact fit must be admitted"
    )
    assert_true(
        not row_major_fits_budget(100, 8, 799), "one byte over must refuse"
    )
    assert_true(
        row_major_fits_budget(100, 8, 801), "one byte under must admit"
    )


def test_fits_budget_no_limit_sentinel() raises:
    """`max_bytes <= 0` is "no budget", which is the forced-on path."""
    assert_true(
        row_major_fits_budget(1 << 30, 200, 0), "0 must mean no budget"
    )
    assert_true(
        row_major_fits_budget(1 << 30, 200, -1), "negative must mean no budget"
    )


def test_budget_refusal_allocates_nothing() raises:
    """Over budget, `build_row_major` is a no-op that does not raise.

    The matrix must come back in the un-built state, not in a half-built one,
    and the feature-major bins must be exactly what they were.
    """
    var widths: List[Int] = [40, 40, 40]
    var data = _matrix(128, widths, 64)
    # Nothing packs at 40 bins, so the stride is 3 and the view would be 384
    # bytes. Refuse it with a budget one byte short of that.
    data.build_row_major(383)
    assert_true(not data.has_row_major(), "refused view was built anyway")
    assert_equal(data.row_major_bytes(), 0, "refused view allocated bytes")
    assert_equal(data.row_stride, 0, "refused view kept a stride")
    for f in range(data.n_features):
        for r in range(data.n_rows):
            assert_equal(
                data.bin_at(r, f),
                Int(UInt8((r + 7 * f) % widths[f])),
                "feature-major matrix disturbed by a refusal",
            )
    # And the same matrix at exactly the budget is built, so the refusal
    # above was the budget and not some other property of the fixture.
    data.build_row_major(384)
    assert_true(data.has_row_major(), "exact-fit view was refused")
    assert_equal(data.row_stride, 3, "stride is not the arithmetic one")
    assert_equal(data.row_major_bytes(), 384, "realized bytes are not 384")


def test_refused_rebuild_destroys_the_old_view() raises:
    """A refused rebuild must not leave the previous view readable.

    `has_row_major`'s size check cannot catch this case -- the stale array is
    the right size for this matrix -- so the builder has to drop it itself.
    """
    var widths: List[Int] = [40, 40, 40]
    var data = _matrix(128, widths, 64)
    data.build_row_major()
    assert_true(data.has_row_major(), "the unbudgeted build did not build")
    data.build_row_major(1)
    assert_true(not data.has_row_major(), "stale view survived a refusal")
    assert_equal(data.row_major_bytes(), 0, "stale bytes survived a refusal")


def test_budget_is_spent_on_the_realized_stride() raises:
    """Packing earns admission: the rule uses `row_stride`, not `n_features`.

    Four features of 16 bins pack two to a byte, so the stride is 2 and not 4.
    A budget that admits the packed view and would refuse the unpacked one
    separates the two rules, and only one of them can pass this.
    """
    var nf = 4
    var n_rows = 128
    var widths: List[Int] = [
        ROW_MAJOR_PACK_MAX_BINS,
        ROW_MAJOR_PACK_MAX_BINS,
        ROW_MAJOR_PACK_MAX_BINS,
        ROW_MAJOR_PACK_MAX_BINS,
    ]
    var data = _matrix(n_rows, widths, 64)
    # Halfway between the packed cost (256) and the unpacked one (512).
    data.build_row_major(n_rows * nf - 1)
    assert_true(
        data.has_row_major(), "a packed view was charged the unpacked price"
    )
    assert_equal(data.row_stride, 2, "four 16-bin features did not pack")
    assert_equal(data.row_major_bytes(), 256, "packed bytes are not 256")


def test_env_mode_three_states() raises:
    """Unset is `auto`, and is not `0`."""
    _clear_env()
    assert_equal(env_row_major_mode(), ROW_MAJOR_AUTO, "unset is not auto")
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "auto")
    assert_equal(env_row_major_mode(), ROW_MAJOR_AUTO, '"auto" is not auto')
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "0")
    assert_equal(env_row_major_mode(), ROW_MAJOR_OFF, '"0" is not off')
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "1")
    assert_equal(env_row_major_mode(), ROW_MAJOR_ON, '"1" is not on')
    _clear_env()


def test_env_mode_refuses_garbage() raises:
    """An unrecognized value raises rather than quietly meaning `auto`."""
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "yes please")
    with assert_raises():
        _ = env_row_major_mode()
    _clear_env()
    assert_equal(
        env_row_major_mode(), ROW_MAJOR_AUTO, "the restore did not restore"
    )


def test_budget_env_override() raises:
    """The default is the documented one, and the override moves it."""
    _clear_env()
    assert_equal(
        row_major_budget_bytes(),
        ROW_MAJOR_DEFAULT_BUDGET_MB * 1024 * 1024,
        "default budget is not the documented one",
    )
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR_MAX_MB", "7")
    assert_equal(
        row_major_budget_bytes(), 7 * 1024 * 1024, "override did not apply"
    )
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR_MAX_MB", "0")
    assert_equal(row_major_budget_bytes(), 0, "0 must lift the budget")
    _clear_env()


def _features(n_rows: Int, n_features: Int) -> List[Float64]:
    """A column-major Float64 matrix `fit_bins` has something to bin.

    Ninety-seven distinct values per column, arithmetic rather than random so
    the fixture is the same on every machine.
    """
    var out = List[Float64](capacity=n_rows * n_features)
    out.resize(n_rows * n_features, 0.0)
    for f in range(n_features):
        for r in range(n_rows):
            out[f * n_rows + r] = Float64((r * (f + 1)) % 97)
    return out^


def test_transform_applies_the_policy() raises:
    """The three states, end to end through `transform`.

    This is the assertion that the policy is actually wired: the predicate
    could be perfect and `transform` could still ignore it.
    """
    var n_rows = 256
    var n_features = 4
    var features = _features(n_rows, n_features)
    var mapper = fit_bins(features, n_rows, n_features, 255)

    _clear_env()
    var auto = mapper.transform(features, n_rows)
    assert_true(
        auto.has_row_major(), "auto did not build the view at 1 KiB of cost"
    )

    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "0")
    var off = mapper.transform(features, n_rows)
    assert_true(not off.has_row_major(), "0 built the view anyway")
    assert_equal(off.row_major_bytes(), 0, "0 allocated bytes")

    # Auto, with a budget below what this tiny matrix needs: refused.
    _clear_env()
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR_MAX_MB", "1")
    # 1 MiB still admits a 256-row matrix, so drive the refusal from the
    # other side: force the budget to a single byte via the same variable is
    # not expressible in mebibytes, so assert the admitted case here and let
    # `test_budget_refusal_allocates_nothing` own the boundary.
    var small_budget = mapper.transform(features, n_rows)
    assert_true(
        small_budget.has_row_major(), "1 MiB refused a 1 KiB view"
    )

    # Forced on, with the budget lifted: built regardless.
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "1")
    var forced = mapper.transform(features, n_rows)
    assert_true(forced.has_row_major(), "forced on did not build")
    _clear_env()


def test_forced_on_ignores_the_budget() raises:
    """`MOJOTREES_CPU_ROW_MAJOR=1` builds above the budget.

    Asserted through `transform` with a budget of one mebibyte and a matrix
    that needs more than that, which is the only combination where forcing and
    the budget actually disagree.
    """
    # The smallest budget `MOJOTREES_CPU_ROW_MAJOR_MAX_MB` can express is one
    # mebibyte, so the fixture has to exceed that and nothing more. Rows are
    # cheap here and features are not -- a feature costs a quantile fit -- so
    # the shape is long rather than wide: 65,536 * 20 = 1,310,720 bytes of
    # record array against a 1,048,576-byte budget.
    var n_rows = 65536
    var n_features = 20
    var features = _features(n_rows, n_features)
    var mapper = fit_bins(features, n_rows, n_features, 255)
    _clear_env()
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR_MAX_MB", "1")
    var refused = mapper.transform(features, n_rows)
    assert_true(
        not refused.has_row_major(), "auto ignored a budget it exceeded"
    )
    _ = setenv("MOJOTREES_CPU_ROW_MAJOR", "1")
    var forced = mapper.transform(features, n_rows)
    assert_true(forced.has_row_major(), "forced on respected the budget")
    assert_true(
        forced.row_major_bytes() > 1024 * 1024,
        "the forced view is not the one the budget refused",
    )
    _clear_env()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
