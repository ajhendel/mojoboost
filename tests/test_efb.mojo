"""Exclusive feature bundling.

Adversarial where it matters, because every bug EFB can have is a silent one:
a bundled matrix that decodes to the wrong feature, a collision that quietly
eats a value, a plan that changes with the wind. So the checks here are

- conflict measurement, including the case the sparse layout makes easy to
  get wrong: an explicitly stored value that bins to the default is not a
  conflict, and a matrix that stores every such entry must produce the same
  plan and the same bundled matrix as one that stores none;
- exact round-tripping, at both levels: every (feature, local bin) encodes
  and decodes back to itself, and every cell of the bundled matrix decodes
  back to the source cell;
- collision semantics under a non-zero conflict budget: the earliest member
  in bundle order wins the row, the loser is dropped, and the plan says so;
- missing values and the categorical boundary, both of which must keep their
  features out of multi-member bundles unless explicitly allowed;
- determinism, including the documented tie-break, and the memory accounting
  and fallback verdict.

Every fixture bins with a default bin in the *middle* of the range (bin 1 of
3) rather than at 0, because the encoding shifts bins below and above the
default differently and a default of 0 would exercise only one of the two.
"""

from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.binning import BinMapper
from mojotrees.categorical import CategoricalSpec
from mojotrees.efb import (
    EFB_NONE,
    EFB_SHARED_BIN,
    EfbParams,
    FeatureBundling,
    bundle_csc,
    conflict_count,
    feature_bin_count,
    fit_bundles,
    nondefault_rows,
    pairwise_conflict,
    unbundle_histogram,
)
from mojotrees.sparse import SparseBinnedMatrix


def _mapper(
    bins_per_feature: List[Int],
    missing: List[Bool],
    cats: CategoricalSpec = CategoricalSpec.none(),
) raises -> BinMapper:
    """A mapper whose feature f takes exactly `bins_per_feature[f]` bins.

    The edge values are irrelevant here: nothing in efb.mojo bins a raw
    value, it only asks the mapper how wide a feature is. A feature marked
    `missing` spends its top bin on missing values, so it needs two fewer
    edges than bins.
    """
    var n_features = len(bins_per_feature)
    var edges = List[Float64]()
    var offsets = List[Int](capacity=n_features + 1)
    offsets.append(0)
    var missing_bin = List[Int](capacity=n_features)
    var n_bins = 2
    for f in range(n_features):
        var m = bins_per_feature[f]
        if m > n_bins:
            n_bins = m
        if cats.is_cat(f):
            offsets.append(len(edges))
            missing_bin.append(-1)
            continue
        var n_edges = m - 2 if missing[f] else m - 1
        for e in range(n_edges):
            edges.append(0.5 + Float64(e))
        offsets.append(len(edges))
        missing_bin.append(m - 1 if missing[f] else -1)
    return BinMapper(
        edges^, offsets^, n_features, n_bins, cats.copy(), missing_bin^
    )


def _mapper2(
    b0: Int, b1: Int, m0: Bool = False, m1: Bool = False
) raises -> BinMapper:
    var bins: List[Int] = [b0, b1]
    var miss: List[Bool] = [m0, m1]
    return _mapper(bins^, miss^)


def _mapper3(
    b0: Int,
    b1: Int,
    b2: Int,
    m0: Bool = False,
    m1: Bool = False,
    m2: Bool = False,
    cats: CategoricalSpec = CategoricalSpec.none(),
) raises -> BinMapper:
    var bins: List[Int] = [b0, b1, b2]
    var miss: List[Bool] = [m0, m1, m2]
    return _mapper(bins^, miss^, cats)


def _binned(
    dense: List[UInt8],
    default_bin: List[UInt8],
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    store_defaults: Bool,
    missing_bin: List[Int] = [],
    cats: CategoricalSpec = CategoricalSpec.none(),
) raises -> SparseBinnedMatrix:
    """A binned CSC matrix from a column-major dense bin matrix.

    With `store_defaults` every cell is stored, the way a producer that
    never eliminated its explicit zeros would hand it over; without it only
    non-default cells are, which is the compact form. The two must be
    indistinguishable to everything in efb.mojo.
    """
    var row_index = List[Int]()
    var bin = List[UInt8]()
    var col_offsets = List[Int](capacity=n_features + 1)
    col_offsets.append(0)
    for f in range(n_features):
        for r in range(n_rows):
            var b = dense[f * n_rows + r]
            if not store_defaults and b == default_bin[f]:
                continue
            row_index.append(r)
            bin.append(b)
        col_offsets.append(len(bin))
    return SparseBinnedMatrix(
        row_index^,
        bin^,
        col_offsets^,
        default_bin.copy(),
        n_rows,
        n_features,
        n_bins,
        cats.copy(),
        missing_bin.copy(),
    )


def _exclusive_dense() -> List[UInt8]:
    """8 rows, 3 features of 3 bins each, default bin 1, pairwise exclusive.

    Non-default rows: f0 at {0, 1}, f1 at {2, 3}, f2 at {4, 5}, and each
    feature uses a bin below its default and one above it.
    """
    var n_rows = 8
    var out = List[UInt8](capacity=3 * n_rows)
    out.resize(3 * n_rows, 1)
    out[0 * n_rows + 0] = 0
    out[0 * n_rows + 1] = 2
    out[1 * n_rows + 2] = 0
    out[1 * n_rows + 3] = 2
    out[2 * n_rows + 4] = 0
    out[2 * n_rows + 5] = 2
    return out^


def _three_defaults() -> List[UInt8]:
    var out = List[UInt8](capacity=3)
    out.resize(3, 1)
    return out^


def _exclusive_fixture(
    store_defaults: Bool = False,
) raises -> SparseBinnedMatrix:
    return _binned(
        _exclusive_dense(), _three_defaults(), 8, 3, 3, store_defaults
    )


def _three_bin_mapper() raises -> BinMapper:
    return _mapper3(3, 3, 3)


def test_feature_bin_count_reads_the_mapper() raises:
    var mapper = _mapper3(3, 5, 4, False, True, False)
    assert_equal(feature_bin_count(mapper, 0), 3)
    # A reserved missing bin is a bin: 3 edges give ordinary bins 0..3 and
    # the missing bin at 4.
    assert_equal(feature_bin_count(mapper, 1), 5)
    assert_equal(feature_bin_count(mapper, 2), 4)
    with assert_raises():
        _ = feature_bin_count(mapper, 3)
    with assert_raises():
        _ = feature_bin_count(mapper, -1)


def test_conflict_measurement() raises:
    var data = _exclusive_fixture()
    var r0 = nondefault_rows(data, 0)
    assert_equal(len(r0), 2)
    assert_equal(r0[0], 0)
    assert_equal(r0[1], 1)
    assert_equal(pairwise_conflict(data, 0, 1), 0)
    assert_equal(pairwise_conflict(data, 0, 2), 0)
    assert_equal(pairwise_conflict(data, 1, 2), 0)

    # Ascending-list intersection, including the awkward ends.
    assert_equal(conflict_count([], []), 0)
    assert_equal(conflict_count([1, 2, 3], []), 0)
    assert_equal(conflict_count([0, 5, 9], [5, 9, 11]), 2)
    assert_equal(conflict_count([0, 1, 2], [3, 4, 5]), 0)
    assert_equal(conflict_count([7], [7]), 1)

    with assert_raises():
        _ = nondefault_rows(data, 3)


def test_stored_default_is_not_a_conflict() raises:
    """The adversarial version of the previous test: the same logical matrix,
    but with every default-bin cell explicitly stored. A producer that has
    not run `eliminate_zeros()` must not look like a fully dense, fully
    conflicting matrix."""
    var compact = _exclusive_fixture(False)
    var full = _exclusive_fixture(True)
    assert_equal(compact.nnz(), 6)
    assert_equal(full.nnz(), 24)
    for f in range(3):
        var a = nondefault_rows(compact, f)
        var b = nondefault_rows(full, f)
        assert_equal(len(a), len(b))
        for i in range(len(a)):
            assert_equal(a[i], b[i])
    assert_equal(pairwise_conflict(full, 0, 1), 0)
    assert_equal(pairwise_conflict(full, 0, 2), 0)


def test_exclusive_features_share_one_column() raises:
    var data = _exclusive_fixture()
    var plan = fit_bundles(_three_bin_mapper(), data)
    plan.validate()
    assert_equal(plan.n_bundles(), 1)
    assert_equal(plan.bundle_size(0), 3)
    # Every feature is non-default on 2 rows, so the documented tie-break
    # (smaller index first) fixes the member order.
    assert_equal(plan.member_at(0, 0), 0)
    assert_equal(plan.member_at(0, 1), 1)
    assert_equal(plan.member_at(0, 2), 2)
    # 1 shared bin + 3 members * (3 - 1) own bins.
    assert_equal(plan.bundle_bins[0], 7)
    assert_equal(plan.slot_offset[0], 1)
    assert_equal(plan.slot_offset[1], 3)
    assert_equal(plan.slot_offset[2], 5)
    assert_true(plan.is_bundled(0))
    assert_true(plan.is_lossless())
    assert_equal(plan.total_collisions(), 0)
    assert_true(plan.use_bundling)


def test_encode_decode_round_trip() raises:
    var data = _exclusive_fixture()
    var plan = fit_bundles(_three_bin_mapper(), data)
    for f in range(3):
        var b = plan.bundle_of[f]
        var s = plan.slot_of[f]
        for local in range(3):
            var code = plan.encode(f, local)
            if local == plan.slot_default[s]:
                assert_equal(code, EFB_SHARED_BIN)
                continue
            assert_true(code > 0)
            assert_equal(plan.decode_feature(b, code), f)
            assert_equal(plan.decode_bin(b, code), local)
    # The shared bin belongs to every member at once, so it names none.
    assert_equal(plan.decode_feature(0, EFB_SHARED_BIN), EFB_NONE)
    assert_equal(plan.decode_bin(0, EFB_SHARED_BIN), EFB_NONE)

    # The exact offsets, spelled out, so a silent renumbering is caught.
    assert_equal(plan.encode(0, 0), 1)
    assert_equal(plan.encode(0, 2), 2)
    assert_equal(plan.encode(1, 0), 3)
    assert_equal(plan.encode(1, 2), 4)
    assert_equal(plan.encode(2, 0), 5)
    assert_equal(plan.encode(2, 2), 6)

    with assert_raises():
        _ = plan.encode(0, 3)
    with assert_raises():
        _ = plan.encode(0, -1)
    with assert_raises():
        _ = plan.decode_feature(0, 7)
    with assert_raises():
        _ = plan.decode_feature(1, 0)


def test_a_split_is_charged_to_the_member_that_gets_read() raises:
    # `charged_feature` is what a per-feature cost keyed by dataset feature
    # id (cegb.mojo's two vectors and its ledger) asks a bundled search
    # space. Charging the bundle would make one sparse feature's first use
    # pay for every feature bundled with it.
    var data = _exclusive_fixture()
    var fitted = fit_bundles(_three_bin_mapper(), data)

    # In force, whatever this fixture's own verdict was: the question is
    # what a bundled search space answers, not whether this plan pays.
    var plan = fitted.copy()
    plan.use_bundling = True
    for f in range(3):
        var b = plan.bundle_of[f]
        var s = plan.slot_of[f]
        for local in range(3):
            if local == plan.slot_default[s]:
                continue
            assert_equal(plan.charged_feature(b, plan.encode(f, local)), f)

    # The shared bin belongs to every member at once, so it cannot be
    # charged to one feature, and a categorical feature is never bundled at
    # all: both are refused rather than attributed to an arbitrary member.
    with assert_raises():
        _ = plan.charged_feature(0, EFB_SHARED_BIN)
    with assert_raises():
        _ = plan.charged_feature(0, 1, True)
    with assert_raises():
        _ = plan.charged_feature(99, 1)

    # Not in force is the identity: the search space is the dataset, so the
    # feature the scan named is the feature that is read.
    var off = fitted.copy()
    off.use_bundling = False
    assert_equal(off.charged_feature(0, EFB_SHARED_BIN), 0)
    assert_equal(off.charged_feature(2, 5), 2)
    assert_equal(off.charged_feature(99, 1, True), 99)


def test_bundled_matrix_recovers_every_cell() raises:
    """The contract, checked exhaustively: for every row and every feature,
    reading the bundle column and decoding it either names that feature and
    its exact bin, or names something else, in which case the feature was at
    its default."""
    var data = _exclusive_fixture()
    var plan = fit_bundles(_three_bin_mapper(), data)
    var bundled = bundle_csc(data, plan)
    assert_equal(bundled.n_features, 1)
    assert_equal(bundled.n_rows, 8)
    assert_equal(bundled.n_bins, 7)
    assert_equal(Int(bundled.default_bin[0]), EFB_SHARED_BIN)
    # Six non-default cells, one column, canonical CSC.
    assert_equal(bundled.nnz(), 6)
    for i in range(1, bundled.nnz()):
        assert_true(bundled.row_index[i] > bundled.row_index[i - 1])

    for r in range(8):
        var b = Int(bundled.bin_at(r, 0))
        var owner = plan.decode_feature(0, b)
        for f in range(3):
            var source = data.bin_at(r, f)
            if f == owner:
                assert_equal(plan.decode_bin(0, b), source)
            else:
                assert_equal(source, Int(data.default_bin[f]))


def test_stored_defaults_give_an_identical_bundling() raises:
    """Explicit zeros change the input bytes and nothing else."""
    var compact = _exclusive_fixture(False)
    var full = _exclusive_fixture(True)
    var mapper = _three_bin_mapper()
    var pa = fit_bundles(mapper, compact)
    var pb = fit_bundles(mapper, full)
    assert_equal(pa.n_bundles(), pb.n_bundles())
    for i in range(len(pa.members)):
        assert_equal(pa.members[i], pb.members[i])
        assert_equal(pa.slot_offset[i], pb.slot_offset[i])
        assert_equal(pa.slot_bins[i], pb.slot_bins[i])
        assert_equal(pa.slot_default[i], pb.slot_default[i])
    assert_equal(pa.bundled_entries, pb.bundled_entries)
    # Only the source-side accounting differs, which is the point.
    assert_equal(pa.source_entries, 6)
    assert_equal(pb.source_entries, 24)

    var ba = bundle_csc(compact, pa)
    var bb = bundle_csc(full, pb)
    assert_equal(ba.nnz(), bb.nnz())
    for i in range(ba.nnz()):
        assert_equal(ba.row_index[i], bb.row_index[i])
        assert_equal(ba.bin[i], bb.bin[i])
    assert_equal(ba.n_bins, bb.n_bins)


def _colliding_fixture() raises -> SparseBinnedMatrix:
    """8 rows, 2 features of 3 bins, default 1, overlapping at row 1."""
    var n_rows = 8
    var dense = List[UInt8](capacity=2 * n_rows)
    dense.resize(2 * n_rows, 1)
    dense[0 * n_rows + 0] = 0
    dense[0 * n_rows + 1] = 2
    dense[1 * n_rows + 1] = 0
    dense[1 * n_rows + 2] = 2
    var defaults = List[UInt8](capacity=2)
    defaults.resize(2, 1)
    return _binned(dense^, defaults^, n_rows, 2, 3, False)


def test_conflicting_features_do_not_bundle_by_default() raises:
    var data = _colliding_fixture()
    var plan = fit_bundles(_mapper2(3, 3), data)
    plan.validate()
    assert_equal(plan.n_bundles(), 2)
    assert_false(plan.is_bundled(0))
    assert_false(plan.is_bundled(1))
    assert_true(plan.is_lossless())
    assert_false(plan.use_bundling)
    # Singletons are identity encoded, so nothing about them moved.
    for f in range(2):
        for local in range(3):
            assert_equal(plan.encode(f, local), local)
        assert_equal(plan.bundle_default_bin(plan.bundle_of[f]), 1)


def test_collision_gives_the_row_to_the_earlier_member() raises:
    var data = _colliding_fixture()
    # 1/8 of 8 rows is a budget of exactly one colliding row.
    var params = EfbParams(0.125, 256, 0, 0.95, 0.0, False)
    var plan = fit_bundles(_mapper2(3, 3), data, params)
    plan.validate()
    assert_equal(plan.n_bundles(), 1)
    assert_equal(plan.member_at(0, 0), 0)
    assert_equal(plan.member_at(0, 1), 1)
    assert_equal(plan.collisions[0], 1)
    assert_false(plan.is_lossless())
    # Three rows carry a non-default value after bundling: 0, 1, and 2.
    assert_equal(plan.bundled_entries, 3)

    var bundled = bundle_csc(data, plan)
    assert_equal(bundled.nnz(), 3)
    # Row 1: f0 is bin 2 and f1 is bin 0. f0 comes first in member order, so
    # it keeps the row and f1's value is the one that is dropped.
    var b = Int(bundled.bin_at(1, 0))
    assert_equal(plan.decode_feature(0, b), 0)
    assert_equal(plan.decode_bin(0, b), 2)
    # The dropped value reads back as f1's default, which is the documented
    # loss and not a wrong bin.
    assert_equal(data.bin_at(1, 1), 0)
    # Rows 0 and 2 are uncontested and survive intact.
    assert_equal(plan.decode_feature(0, Int(bundled.bin_at(0, 0))), 0)
    assert_equal(plan.decode_bin(0, Int(bundled.bin_at(0, 0))), 0)
    assert_equal(plan.decode_feature(0, Int(bundled.bin_at(2, 0))), 1)
    assert_equal(plan.decode_bin(0, Int(bundled.bin_at(2, 0))), 2)
    # Every other row is shared-default.
    for r in range(3, 8):
        assert_equal(Int(bundled.bin_at(r, 0)), EFB_SHARED_BIN)


def test_conflict_budget_is_a_bundle_total() raises:
    """Three features that each collide with the next, on distinct rows: the
    budget is spent across the whole bundle, not per pair."""
    var n_rows = 10
    var dense = List[UInt8](capacity=3 * n_rows)
    dense.resize(3 * n_rows, 1)
    dense[0 * n_rows + 0] = 0
    dense[0 * n_rows + 1] = 2
    dense[1 * n_rows + 1] = 2
    dense[1 * n_rows + 2] = 0
    dense[2 * n_rows + 2] = 2
    dense[2 * n_rows + 3] = 0
    var defaults = List[UInt8](capacity=3)
    defaults.resize(3, 1)
    var data = _binned(dense^, defaults^, n_rows, 3, 3, False)
    var mapper = _mapper3(3, 3, 3)

    # A budget of one collision admits f1 into f0's bundle but not f2, whose
    # collision with f1 would make two.
    var tight = fit_bundles(mapper, data, EfbParams(0.1, 256, 0, 0.95, 0.0, False))
    tight.validate()
    assert_equal(tight.n_bundles(), 2)
    assert_equal(tight.bundle_size(0), 2)
    assert_equal(tight.collisions[0], 1)

    # A budget of two admits all three.
    var loose = fit_bundles(mapper, data, EfbParams(0.2, 256, 0, 0.95, 0.0, False))
    loose.validate()
    assert_equal(loose.n_bundles(), 1)
    assert_equal(loose.collisions[0], 2)
    assert_equal(loose.bundled_entries, 4)


def test_bin_budget_forces_a_new_bundle() raises:
    """Exclusive features still split apart when the bundle cannot hold their
    bins, because the binned matrix stores bins in a UInt8."""
    var data = _exclusive_fixture()
    var mapper = _three_bin_mapper()
    # Room for the shared bin plus two members' two bins each, not three.
    var params = EfbParams(0.0, 5, 0, 0.95, 0.0, False)
    var plan = fit_bundles(mapper, data, params)
    plan.validate()
    assert_equal(plan.n_bundles(), 2)
    assert_equal(plan.bundle_size(0), 2)
    assert_equal(plan.bundle_bins[0], 5)
    assert_equal(plan.bundle_size(1), 1)
    var bundled = bundle_csc(data, plan)
    assert_equal(bundled.n_bins, 5)
    assert_true(bundled.n_bins <= 256)

    var capped = fit_bundles(mapper, data, EfbParams(0.0, 256, 2, 0.95, 0.0, False))
    assert_equal(capped.n_bundles(), 2)
    assert_equal(capped.bundle_size(0), 2)


def test_dense_feature_is_left_alone() raises:
    """A feature non-default nearly everywhere is not what EFB is for, and
    bundling it would burn a bundle's bin budget for nothing."""
    var n_rows = 10
    var dense = List[UInt8](capacity=2 * n_rows)
    dense.resize(2 * n_rows, 1)
    for r in range(n_rows):
        dense[0 * n_rows + r] = 2
    dense[1 * n_rows + 0] = 0
    var defaults = List[UInt8](capacity=2)
    defaults.resize(2, 1)
    var data = _binned(dense^, defaults^, n_rows, 2, 3, False)
    var params = EfbParams(0.0, 256, 0, 0.5, 0.0, False)
    var plan = fit_bundles(_mapper2(3, 3), data, params)
    plan.validate()
    assert_equal(plan.n_bundles(), 2)
    assert_false(plan.is_bundled(0))


def test_missing_values_stay_out_unless_asked() raises:
    """A feature that reserves a missing bin is a singleton by default,
    because a bundled column carries one missing bin and a node learns one
    default direction."""
    var n_rows = 8
    # f0: 4 bins with the top reserved for missing. f1, f2: plain 3-bin.
    var dense = List[UInt8](capacity=3 * n_rows)
    dense.resize(3 * n_rows, 1)
    dense[0 * n_rows + 0] = 3
    dense[0 * n_rows + 1] = 0
    dense[1 * n_rows + 2] = 0
    dense[1 * n_rows + 3] = 2
    dense[2 * n_rows + 4] = 2
    dense[2 * n_rows + 5] = 0
    var defaults = List[UInt8](capacity=3)
    defaults.resize(3, 1)
    var missing: List[Int] = [3, -1, -1]
    var mapper = _mapper3(4, 3, 3, True, False, False)
    assert_equal(feature_bin_count(mapper, 0), 4)
    var data = _binned(
        dense^, defaults^, n_rows, 3, 4, False, missing.copy()
    )

    var plan = fit_bundles(mapper, data)
    plan.validate()
    assert_equal(plan.n_bundles(), 2)
    assert_false(plan.is_bundled(0))
    assert_true(plan.is_bundled(1))
    # The singleton keeps the missing bin exactly where it was.
    var b0 = plan.bundle_of[0]
    assert_equal(plan.encode(0, 3), 3)
    var mb = plan.missing_bins(b0)
    assert_equal(len(mb), 1)
    assert_equal(mb[0], 3)
    var bundled = bundle_csc(data, plan)
    assert_equal(bundled.missing_bin[b0], 3)
    assert_true(bundled.is_missing(0, b0))
    assert_false(bundled.is_missing(1, b0))

    # Opting in bundles it, and the missing bin follows the encoding like any
    # other non-default bin.
    var opted = fit_bundles(
        mapper, data, EfbParams(0.0, 256, 0, 0.95, 0.0, True)
    )
    opted.validate()
    assert_equal(opted.n_bundles(), 1)
    var s = opted.slot_of[0]
    var coded = opted.encode(0, 3)
    assert_true(coded != EFB_SHARED_BIN)
    assert_equal(opted.decode_feature(0, coded), 0)
    assert_equal(opted.decode_bin(0, coded), 3)
    var opted_missing = opted.missing_bins(0)
    assert_equal(len(opted_missing), 1)
    assert_equal(opted_missing[0], coded)
    var opted_bundled = bundle_csc(data, opted)
    # Exactly one member reserves a missing bin, so the bundled column can
    # still carry it.
    assert_equal(opted_bundled.missing_bin[0], coded)
    assert_true(opted_bundled.is_missing(0, 0))
    _ = s


def test_two_missing_members_report_no_single_missing_bin() raises:
    var n_rows = 8
    var dense = List[UInt8](capacity=2 * n_rows)
    dense.resize(2 * n_rows, 1)
    dense[0 * n_rows + 0] = 3
    dense[1 * n_rows + 1] = 3
    var defaults = List[UInt8](capacity=2)
    defaults.resize(2, 1)
    var missing: List[Int] = [3, 3]
    var mapper = _mapper2(4, 4, True, True)
    var data = _binned(dense^, defaults^, n_rows, 2, 4, False, missing^)
    var plan = fit_bundles(
        mapper, data, EfbParams(0.0, 256, 0, 0.95, 0.0, True)
    )
    plan.validate()
    assert_equal(plan.n_bundles(), 1)
    var mb = plan.missing_bins(0)
    assert_equal(len(mb), 2)
    assert_true(mb[0] != mb[1])
    var bundled = bundle_csc(data, plan)
    # One Int cannot name two bins, so the column declines to and the caller
    # must read `missing_bins` instead.
    assert_equal(bundled.missing_bin[0], EFB_NONE)


def test_categorical_features_are_never_bundled() raises:
    """A categorical feature's bin ids index its category table and its bin 0
    is the reserved unknown bin that routes right. Neither survives being
    offset, so it stays a singleton and stays identity encoded."""
    var n_rows = 8
    var flags: List[Bool] = [True, False, False]
    var codes: List[Int] = [0, 4, 7]
    var cat_offsets: List[Int] = [0, 3, 3, 3]
    var cats = CategoricalSpec(flags^, codes^, cat_offsets^)
    var mapper = _mapper3(4, 3, 3, False, False, False, cats)
    assert_equal(feature_bin_count(mapper, 0), 4)

    var dense = List[UInt8](capacity=3 * n_rows)
    dense.resize(3 * n_rows, 1)
    dense[0 * n_rows + 0] = 0
    dense[0 * n_rows + 1] = 3
    dense[1 * n_rows + 2] = 0
    dense[1 * n_rows + 3] = 2
    dense[2 * n_rows + 4] = 2
    dense[2 * n_rows + 5] = 0
    var defaults = List[UInt8](capacity=3)
    defaults.resize(3, 1)
    var data = _binned(
        dense^, defaults^, n_rows, 3, 4, False, List[Int](), cats
    )

    var plan = fit_bundles(mapper, data)
    plan.validate()
    assert_equal(plan.n_bundles(), 2)
    assert_false(plan.is_bundled(0))
    assert_true(plan.is_bundled(1))
    assert_true(plan.is_bundled(2))
    var b0 = plan.bundle_of[0]
    for local in range(4):
        assert_equal(plan.encode(0, local), local)
        assert_equal(plan.decode_feature(b0, local), 0)
        assert_equal(plan.decode_bin(b0, local), local)

    var bundled = bundle_csc(data, plan)
    assert_true(bundled.cats.is_cat(b0))
    assert_equal(bundled.cats.n_categories(b0), 3)
    assert_equal(bundled.cats.codes[bundled.cats.offsets[b0] + 1], 4)
    assert_equal(bundled.cats.codes[bundled.cats.offsets[b0] + 2], 7)
    # The multi-member bundle is numerical, whichever column it landed in.
    var b1 = plan.bundle_of[1]
    assert_false(bundled.cats.is_cat(b1))
    # Every one of the categorical column's cells still reads back exactly.
    for r in range(n_rows):
        assert_equal(bundled.bin_at(r, b0), data.bin_at(r, 0))


def test_all_singleton_plan_reproduces_the_source() raises:
    """The degenerate plan is the identity, up to dropping stored entries
    that sat at their default bin, which is exactly what the default bin
    already says."""
    var data = _colliding_fixture()
    var plan = fit_bundles(_mapper2(3, 3), data)
    var bundled = bundle_csc(data, plan)
    assert_equal(bundled.n_features, data.n_features)
    assert_equal(bundled.n_bins, 3)
    for f in range(data.n_features):
        var b = plan.bundle_of[f]
        assert_equal(Int(bundled.default_bin[b]), Int(data.default_bin[f]))
        for r in range(data.n_rows):
            assert_equal(bundled.bin_at(r, b), data.bin_at(r, f))


def test_determinism_and_the_tie_break() raises:
    """Same matrix, same plan, every time; and when counts tie, the smaller
    feature index goes first. A count that does not tie wins outright."""
    var data = _exclusive_fixture()
    var mapper = _three_bin_mapper()
    var first = fit_bundles(mapper, data)
    for _ in range(4):
        var again = fit_bundles(mapper, data)
        assert_equal(again.n_bundles(), first.n_bundles())
        assert_equal(len(again.members), len(first.members))
        for i in range(len(first.members)):
            assert_equal(again.members[i], first.members[i])
            assert_equal(again.slot_offset[i], first.slot_offset[i])
        for b in range(first.n_bundles()):
            assert_equal(again.bundle_bins[b], first.bundle_bins[b])
            assert_equal(again.collisions[b], first.collisions[b])
        assert_equal(again.bundled_entries, first.bundled_entries)
        assert_equal(again.use_bundling, first.use_bundling)

    # f2 is non-default on three rows instead of two, so it is visited first
    # and takes the first member slot.
    var n_rows = 8
    var dense = _exclusive_dense()
    dense[2 * n_rows + 6] = 2
    var plan = fit_bundles(
        mapper, _binned(dense^, _three_defaults(), n_rows, 3, 3, False)
    )
    assert_equal(plan.n_bundles(), 1)
    assert_equal(plan.member_at(0, 0), 2)
    assert_equal(plan.member_at(0, 1), 0)
    assert_equal(plan.member_at(0, 2), 1)


def test_memory_accounting_and_fallback() raises:
    var data = _exclusive_fixture()
    var plan = fit_bundles(_three_bin_mapper(), data)
    # Three 3-bin columns of a rectangular histogram against one 7-bin one.
    assert_equal(plan.histogram_slots_unbundled(), 3 * 3)
    assert_equal(plan.histogram_slots_bundled(), 7)
    assert_true(plan.binned_bytes_bundled() < plan.binned_bytes_unbundled())
    assert_true(plan.use_bundling)

    # Demanding a 50% cut of the histogram footprint is more than this plan
    # delivers, so it declines rather than pretending.
    var strict = fit_bundles(
        _three_bin_mapper(), data, EfbParams(0.0, 256, 0, 0.95, 0.5, False)
    )
    assert_equal(strict.n_bundles(), 1)
    assert_false(strict.use_bundling)

    # Nothing bundled at all is never worth it.
    var conflicting = _colliding_fixture()
    var none = fit_bundles(_mapper2(3, 3), conflicting)
    assert_equal(none.n_bundles(), 2)
    assert_false(none.use_bundling)


def _direct_histogram(
    data: SparseBinnedMatrix,
    feature: Int,
    n_local: Int,
    grad: List[Float64],
    hess: List[Float64],
) raises -> List[Float64]:
    """A feature's own per-bin gradient sums, straight off the source matrix.
    Returns 2 * n_local entries: gradients then Hessians, so one helper
    covers both."""
    var out = List[Float64](capacity=2 * n_local)
    out.resize(2 * n_local, 0.0)
    for r in range(data.n_rows):
        var b = data.bin_at(r, feature)
        out[b] += grad[r]
        out[n_local + b] += hess[r]
    return out^


def test_unbundle_histogram_recovers_each_member() raises:
    """The reconstruction that makes bundling free: a member's local
    histogram comes back from the bundle's, with its default bin recovered by
    subtraction."""
    var data = _exclusive_fixture()
    var plan = fit_bundles(_three_bin_mapper(), data)
    var bundled = bundle_csc(data, plan)
    assert_equal(plan.n_bundles(), 1)

    var grad = List[Float64](capacity=8)
    var hess = List[Float64](capacity=8)
    for r in range(8):
        grad.append(0.5 * Float64(r) - 1.25)
        hess.append(1.0 + 0.125 * Float64(r))

    var width = plan.bundle_bins[0]
    var bg = List[Float64](capacity=width)
    bg.resize(width, 0.0)
    var bh = List[Float64](capacity=width)
    bh.resize(width, 0.0)
    var bc = List[Int](capacity=width)
    bc.resize(width, 0)
    for r in range(8):
        var b = bundled.bin_at(r, 0)
        bg[b] += grad[r]
        bh[b] += hess[r]
        bc[b] += 1

    for k in range(3):
        var f = plan.member_at(0, k)
        var local = unbundle_histogram(plan, 0, k, bg, bh, bc, 0)
        assert_equal(local.n_cells(), 3)
        var direct = _direct_histogram(data, f, 3, grad, hess)
        var total = 0
        for b in range(3):
            assert_true(abs(local.grad_at(b) - direct[b]) < 1e-12)
            assert_true(abs(local.hess_at(b) - direct[3 + b]) < 1e-12)
            total += local.count_at(b)
        assert_equal(total, 8)
        # The two non-default bins hold one row each in this fixture.
        assert_equal(local.count_at(0), 1)
        assert_equal(local.count_at(2), 1)
        assert_equal(local.count_at(1), 6)

    with assert_raises():
        _ = unbundle_histogram(plan, 0, 3, bg, bh, bc, 0)
    with assert_raises():
        _ = unbundle_histogram(plan, 1, 0, bg, bh, bc, 0)
    with assert_raises():
        _ = unbundle_histogram(plan, 0, 0, bg, bh, bc, 1)


def test_unbundle_histogram_passes_a_singleton_through() raises:
    var data = _colliding_fixture()
    var plan = fit_bundles(_mapper2(3, 3), data)
    assert_equal(plan.n_bundles(), 2)
    var grad: List[Float64] = [1.0, 2.0, 3.0]
    var hess: List[Float64] = [4.0, 5.0, 6.0]
    var count: List[Int] = [7, 8, 9]
    var local = unbundle_histogram(plan, 0, 0, grad, hess, count, 0)
    for b in range(3):
        assert_equal(local.grad_at(b), grad[b])
        assert_equal(local.hess_at(b), hess[b])
        assert_equal(local.count_at(b), count[b])


def test_params_and_plan_are_validated() raises:
    var data = _exclusive_fixture()
    var mapper = _three_bin_mapper()
    with assert_raises():
        _ = fit_bundles(mapper, data, EfbParams(-0.1, 256, 0, 0.95, 0.0, False))
    with assert_raises():
        _ = fit_bundles(mapper, data, EfbParams(0.0, 257, 0, 0.95, 0.0, False))
    with assert_raises():
        _ = fit_bundles(mapper, data, EfbParams(0.0, 1, 0, 0.95, 0.0, False))
    with assert_raises():
        _ = fit_bundles(mapper, data, EfbParams(0.0, 256, 0, 0.0, 0.0, False))
    with assert_raises():
        _ = fit_bundles(mapper, data, EfbParams(0.0, 256, 0, 0.95, 1.0, False))
    # A mapper that describes a different matrix.
    with assert_raises():
        _ = fit_bundles(_mapper2(3, 3), data)
    # A mapper too narrow for the matrix's own default bin.
    with assert_raises():
        _ = fit_bundles(_mapper3(1, 3, 3), data)

    var plan = fit_bundles(mapper, data)
    plan.validate()
    with assert_raises():
        _ = bundle_csc(_colliding_fixture(), plan)

    var broken = plan.copy()
    broken.slot_offset[1] = 2
    with assert_raises():
        broken.validate()

    var doubled = plan.copy()
    doubled.members[2] = 0
    with assert_raises():
        doubled.validate()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
