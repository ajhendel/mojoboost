"""The packed split-record readback, and the four transports that move it.

`GpuSplitSearcher.download_words` is the round trip this plane's time is made
of: `_device_search_resident` reads one 136-byte record per split, about a
hundred times in a fit. It used to move that record in two `enqueue_copy`
calls into two pinned `HostBuffer`s followed by a `synchronize()`, four
command buffers a trip. It now moves it in one copy out of one device
allocation into ordinary heap memory, two command buffers a trip, and
`gpu_runtime`'s transport table prices both. `pixi run probe-readback`
measured them interleaved in one process on an M4: 202.14 us for the pinned
pair, 124.85 for the packed plain one, against a bare-`synchronize()` floor of
10.59.

Nothing about a record's value is supposed to change, and that is the whole
claim this file tests.

Why the fixture words are all different from each other
------------------------------------------------------
Packing two planes into one buffer and unpacking them again is the kind of
change that is right on a uniform fixture and wrong on a real one. A record
set of zeros, or of one repeated value, passes under a wrong slot stride, a
wrong plane order, and an off-by-one at the boundary between the integer and
float regions, because every wrong answer is spelled the same as the right
one. So every word in `_write_fixture` is distinct from every other word:

- Distinct **across slots**, by a `r * 1000` term, so a transport that read
  slot 1 where it meant slot 0 shows. `max_records` is 3 here for the same
  reason: with one slot a stride error has nothing to be wrong about.
- Distinct **across word indices within a slot**, by an `+ i` term, so a
  transport that transposed two words inside a record shows.
- Distinct **across planes**, by sign: the integer plane is positive and the
  float plane is negative. A read that took the float region at the integer
  region's offset, or that swapped the two, shows in the sign before it shows
  anywhere else.
- Distinct **at the boundary**, which is the one a uniform fixture is least
  able to see. `words_f[0]` is `-2000000` and the integer word immediately
  before it in the packed buffer is `1000022`; an unpack that started the
  float region one word early or one word late produces neither.

Every value is under 2^24, so each is exactly representable as a Float32 and
the float comparisons are `assert_equal` and not a tolerance.

The fixture is written through `rec_i_dev` and `rec_f_dev`, which is the view
every kernel and `gpu_resident_round.mojo` hold, and read back through
`records_dev`, which is the view the packed transports copy from. That
direction is deliberate: it is what proves the two `create_sub_buffer` windows
really do alias the one allocation at the offsets `SPLIT_RECORD_WORDS`
declares. If they did not, the packed arms would return the previous contents
of an unrelated buffer and every assertion here would fail.

What this file does not test, and where that is tested
------------------------------------------------------
It does not test that the unsafe transports are unsafe. That is measured, not
argued, and the measurement is `probes/readback_cost.mojo`, which executes
`pinned_pair_nosync` and `pinned_one_nosync` under a kernel slow enough to
make the race deterministic and reports 34 of 34 words wrong. What this file
tests is that those rows cannot be reached from a fit: `set_readback_transport`
refuses them, and refuses them on a machine with no Metal too, because what
makes them wrong elsewhere is unestablished rather than known to be false.

It also does not test timing. The orchestrator's window measures the arms;
this file asserts they are interchangeable, which is what makes a measurement
of them meaningful.

Four tests run with no accelerator: the table's self-consistency, the default,
the record layout arithmetic, and the shape of the two refusals that do not
need a device. The rest skip (passing).
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojotrees.gpu_runtime import (
    N_READBACK_TRANSPORTS,
    READBACK_DEFAULT,
    READBACK_MAP,
    READBACK_PINNED_ONE_NOSYNC,
    READBACK_PINNED_ONE_SYNC,
    READBACK_PINNED_PAIR_NOSYNC,
    READBACK_PINNED_PAIR_SYNC,
    READBACK_PLAIN_ONE,
    READBACK_PLAIN_PAIR,
    readback_report,
    readback_transport,
    readback_transport_name,
    require_readback_table_consistent,
)
from mojotrees.gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    SPLIT_FWORDS,
    SPLIT_IWORDS,
    SPLIT_RECORD_WORDS,
)


comptime _SLOTS = 3
"""Record slots in the fixture. Three and not one, so that a slot stride has
something to be wrong about, and odd so that a transport that reversed the
slot order would not be hidden by a symmetric count."""


def _implemented_arms() -> List[Int]:
    """The four transports `download_words` executes, cheapest first.

    All four are correct on Metal and all four must return the same words.
    They are not redundant with each other: `plain_one` differs from
    `pinned_pair_sync` in two ways at once (the destination kind and the
    packing), and the two middle arms hold one of those fixed while changing
    the other, so a window can attribute a difference to one of them.
    """
    return [
        READBACK_PLAIN_ONE,
        READBACK_PINNED_ONE_SYNC,
        READBACK_PLAIN_PAIR,
        READBACK_PINNED_PAIR_SYNC,
    ]


# --- Host-side: the table, the default, and the layout --------------------


def test_the_transport_table_is_self_consistent() raises:
    """Every row's destination kind and wait count have to be able to hold at
    once. The rule and the reason are at `require_readback_table_consistent`;
    this is the gate that runs it. No accelerator needed: it reads seven rows
    of a policy table."""
    require_readback_table_consistent()


def test_the_default_transport_is_the_measured_one() raises:
    """`READBACK_DEFAULT` is `plain_one`, and its row says what the probe
    measured about it: two command buffers, an unpinned destination, one
    packed copy, correct on Metal.

    Pinned here would be the dangerous edit and the row is what
    `download_words` branches on, so the field is asserted rather than
    assumed."""
    assert_equal(READBACK_DEFAULT, READBACK_PLAIN_ONE)
    var row = readback_transport(READBACK_DEFAULT)
    assert_equal(row.command_buffers, 2)
    assert_equal(row.host_waits, 1)
    assert_false(row.pinned_destination)
    assert_true(row.packed_source)
    assert_true(row.correct_on_metal)
    # And it is the cheapest correct row, which is the reason it is the
    # default rather than merely a row that happens to work.
    for t in range(N_READBACK_TRANSPORTS):
        var other = readback_transport(t)
        if other.correct_on_metal:
            assert_true(other.command_buffers >= row.command_buffers)


def test_the_pinned_pair_is_still_reachable_as_the_ab_arm() raises:
    """The shape this module shipped until 2026-08-16 has to stay in the
    table and stay correct, or the window has nothing to hold the new default
    against."""
    var row = readback_transport(READBACK_PINNED_PAIR_SYNC)
    assert_equal(row.command_buffers, 4)
    assert_true(row.pinned_destination)
    assert_false(row.packed_source)
    assert_true(row.correct_on_metal)
    assert_equal(readback_transport_name(READBACK_PINNED_PAIR_SYNC),
                 String("pinned_pair_sync"))


def test_the_record_is_thirty_four_words_and_one_hundred_thirty_six_bytes(
) raises:
    """The packed width is the sum of the two planes and nothing else. If
    someone widens a plane, this is the line that says the 136 bytes quoted
    all over this module and `docs/GPU_PORTABILITY.md` moved with it."""
    assert_equal(SPLIT_RECORD_WORDS, SPLIT_IWORDS + SPLIT_FWORDS)
    assert_equal(SPLIT_RECORD_WORDS, 34)
    assert_equal(SPLIT_RECORD_WORDS * 4, 136)


def test_the_report_names_every_transport() raises:
    """`readback_report` is what a probe prints. It has to mention every row,
    so that adding a transport without a measurement is visible."""
    var text = readback_report()
    for t in range(N_READBACK_TRANSPORTS):
        assert_true(text.find(readback_transport_name(t)) >= 0)
    assert_true(text.find("plain") >= 0)
    assert_true(text.find("packed") >= 0)


# --- The fixture ----------------------------------------------------------


def _fixture_int(slot: Int, word: Int) -> Int32:
    """Positive, and distinct for every (slot, word) pair. See the module
    docstring for why every one of those distinctions is load-bearing."""
    return Int32(1000000 + slot * 1000 + word)


def _fixture_float(slot: Int, word: Int) -> Float32:
    """Negative, and distinct for every (slot, word) pair, so the plane a
    value came from is readable off its sign. Magnitudes stay under 2^24, so
    each is exact in Float32 and the comparison needs no tolerance."""
    return Float32(-(2000000 + slot * 1000 + word))


def _write_fixture(mut ctx: DeviceContext, mut searcher: GpuSplitSearcher
) raises:
    """Fill both record planes through the windows the kernels write.

    Through `rec_i_dev` and `rec_f_dev` rather than through `records_dev`: the
    packed transports read the whole allocation, so writing the windows and
    reading the allocation is what checks that the windows land where
    `SPLIT_RECORD_WORDS` says they land. Writing `records_dev` directly would
    only check this file against itself.
    """
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_i = _SLOTS * SPLIT_IWORDS
        var n_f = _SLOTS * SPLIT_FWORDS
        var host_i = List[Int32](capacity=n_i)
        var host_f = List[Float32](capacity=n_f)
        for r in range(_SLOTS):
            for w in range(SPLIT_IWORDS):
                host_i.append(_fixture_int(r, w))
        for r in range(_SLOTS):
            for w in range(SPLIT_FWORDS):
                host_f.append(_fixture_float(r, w))
        ctx.enqueue_copy(
            dst_buf=searcher.rec_i_dev, src_ptr=host_i.unsafe_ptr()
        )
        ctx.enqueue_copy(
            dst_buf=searcher.rec_f_dev, src_ptr=host_f.unsafe_ptr()
        )
        ctx.synchronize()


def _assert_fixture(
    words_i: List[Int32], words_f: List[Float32], arm: String
) raises:
    assert_equal(len(words_i), _SLOTS * SPLIT_IWORDS)
    assert_equal(len(words_f), _SLOTS * SPLIT_FWORDS)
    for r in range(_SLOTS):
        for w in range(SPLIT_IWORDS):
            var got = words_i[r * SPLIT_IWORDS + w]
            var want = _fixture_int(r, w)
            if got != want:
                raise Error(
                    "readback arm ",
                    arm,
                    " returned integer word ",
                    w,
                    " of slot ",
                    r,
                    " as ",
                    Int(got),
                    ", expected ",
                    Int(want),
                )
        for w in range(SPLIT_FWORDS):
            var gotf = words_f[r * SPLIT_FWORDS + w]
            var wantf = _fixture_float(r, w)
            if gotf != wantf:
                raise Error(
                    "readback arm ",
                    arm,
                    " returned float word ",
                    w,
                    " of slot ",
                    r,
                    " as ",
                    Float64(gotf),
                    ", expected ",
                    Float64(wantf),
                )


def test_every_arm_unpacks_the_distinguishable_fixture() raises:
    """The packing test.

    One searcher, one fixture written once, and every implemented transport
    run against it in turn inside the same process. Each has to hand back the
    exact words the fixture put on the device, in the right slot, in the right
    plane, at the right index.

    Running all four against one fixture rather than one each also makes the
    arms interchangeable in the strongest sense the window needs: it is the
    same device state, so a difference between arms could only be the
    transport."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var searcher = GpuSplitSearcher(ctx, 4, 8, max_records=_SLOTS)
        _write_fixture(ctx, searcher)
        var arms = _implemented_arms()
        for i in range(len(arms)):
            searcher.set_readback_transport(arms[i])
            var words_i = List[Int32]()
            var words_f = List[Float32]()
            searcher.download_words(words_i, words_f)
            _assert_fixture(
                words_i, words_f, readback_transport_name(arms[i])
            )
            # And the searcher can say which arm produced them, which is what
            # a benchmark line needs to be attributable at all.
            assert_true(
                searcher.describe_scan().find(
                    readback_transport_name(arms[i])
                )
                >= 0
            )
        assert_equal(len(arms), 4)


def test_the_fixture_would_catch_a_transposed_plane() raises:
    """A guard on the guard.

    `_assert_fixture` is only worth running if the fixture it checks could
    fail. This asserts the three ways a packing bug is spelled are three
    different values in this fixture: the integer word at the boundary, the
    first float word, and the same float word one slot over are mutually
    distinct, and the two planes never share a value. Arithmetic only, so it
    runs everywhere."""
    var last_int = _fixture_int(_SLOTS - 1, SPLIT_IWORDS - 1)
    var first_float = _fixture_float(0, 0)
    assert_true(Float64(last_int) != Float64(first_float))
    assert_true(_fixture_float(0, 0) != _fixture_float(1, 0))
    assert_true(_fixture_float(0, 0) != _fixture_float(0, 1))
    assert_true(_fixture_int(0, 0) != _fixture_int(1, 0))
    assert_true(_fixture_int(0, 0) != _fixture_int(0, 1))
    # Every integer word positive, every float word negative: the sign is the
    # plane, which is what makes a swapped plane visible at a glance.
    for r in range(_SLOTS):
        for w in range(SPLIT_IWORDS):
            assert_true(_fixture_int(r, w) > Int32(0))
        for w in range(SPLIT_FWORDS):
            assert_true(_fixture_float(r, w) < Float32(0.0))


# --- The arms against a real search ---------------------------------------


def _histogram_words(n_features: Int, n_bins: Int) -> List[Int32]:
    """A `[grad | hess | count]` fixed-point buffer with a clear winner, in
    the layout `GpuHistogramBuilder` produces. Both scales are 1.0 at the call
    site, so every quantized word dequantizes to itself."""
    var size = n_features * n_bins
    var words = List[Int32](capacity=3 * size)
    words.resize(3 * size, Int32(0))
    for f in range(n_features):
        for b in range(n_bins):
            var i = f * n_bins + b
            # A monotone gradient ramp per feature, offset by the feature, so
            # the features do not tie and the winner is not slot 0 by default.
            words[i] = Int32((b - n_bins // 2) * (f + 1) * 3)
            words[size + i] = Int32(10 + b)
            words[2 * size + i] = Int32(10 + b)
    return words^


def _assert_same_record(
    a: GpuSplitRecord, b: GpuSplitRecord, arm: String
) raises:
    """Field for field and with no tolerance. Neither arm recomputes a gain,
    so a difference of one bit would mean a transport dropped or moved a
    word, not that something rounded."""
    if a.feature != b.feature:
        raise Error(
            "readback arm ",
            arm,
            " changed the chosen feature: ",
            a.feature,
            " against ",
            b.feature,
        )
    assert_equal(a.bin, b.bin)
    assert_equal(a.ordinal, b.ordinal)
    assert_equal(a.found, b.found)
    assert_equal(a.default_left, b.default_left)
    assert_equal(a.is_categorical, b.is_categorical)
    assert_equal(a.left.count, b.left.count)
    assert_equal(a.right.count, b.right.count)
    assert_equal(a.total.count, b.total.count)
    assert_equal(a.left.grad, b.left.grad)
    assert_equal(a.left.hess, b.left.hess)
    assert_equal(a.right.grad, b.right.grad)
    assert_equal(a.right.hess, b.right.hess)
    assert_equal(a.total.grad, b.total.grad)
    assert_equal(a.total.hess, b.total.hess)
    assert_equal(a.gain, b.gain)
    assert_equal(a.runner_gain, b.runner_gain)
    assert_equal(a.left_value, b.left_value)
    assert_equal(a.right_value, b.right_value)
    assert_equal(a.parent_value, b.parent_value)


def test_every_arm_returns_the_same_record_from_a_real_search() raises:
    """The fixture test proves the transports move words. This proves they
    move the words the *kernels* wrote, through the sub-buffer windows, on a
    searcher driven the way the trainer drives one.

    It is the check the fixture cannot make: a packing that was wrong in the
    same way on the write side and the read side would satisfy the fixture and
    fail here, because the kernel writes through `rec_i_dev` at offsets this
    file never chose."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_features = 5
        var n_bins = 16
        var words = _histogram_words(n_features, n_bins)
        var params = GpuSplitParams.default()
        var ctx = DeviceContext()
        var searcher = GpuSplitSearcher(ctx, n_features, n_bins)
        var hist = ctx.enqueue_create_buffer[DType.int32](len(words))
        ctx.enqueue_copy(dst_buf=hist, src_ptr=words.unsafe_ptr())
        ctx.synchronize()

        var arms = _implemented_arms()
        searcher.set_readback_transport(arms[0])
        searcher.enqueue(hist, params, 1.0, 1.0)
        var reference = searcher.download(0)
        # A search that found nothing would make every comparison below
        # vacuous, which is exactly how a readback test passes while saying
        # nothing.
        assert_true(reference.found)
        for i in range(1, len(arms)):
            searcher.set_readback_transport(arms[i])
            searcher.enqueue(hist, params, 1.0, 1.0)
            var got = searcher.download(0)
            _assert_same_record(
                got, reference, readback_transport_name(arms[i])
            )


def test_the_frontier_download_agrees_across_arms() raises:
    """The same, through `download_frontier`, which is the entry point that
    moves more than one slot and therefore the one a slot stride can break.

    A packed transport reads the whole allocation in one copy, so a stride
    error here would land every record in the wrong slot rather than corrupt
    one of them, and a single-slot test would never see it.

    The three slots are narrowed to three different feature sets so that they
    do not all choose the same split. Three identical records would make a
    slot permutation invisible, which is the same trap a uniform fixture
    is."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_features = 4
        var n_bins = 16
        var words = _histogram_words(n_features, n_bins)
        var params = GpuSplitParams.default()
        var ctx = DeviceContext()
        var searcher = GpuSplitSearcher(
            ctx, n_features, n_bins, max_records=_SLOTS
        )
        var hist = ctx.enqueue_create_buffer[DType.int32](len(words))
        ctx.enqueue_copy(dst_buf=hist, src_ptr=words.unsafe_ptr())
        ctx.synchronize()

        var arms = _implemented_arms()
        var reference = List[GpuSplitRecord]()
        for i in range(len(arms)):
            searcher.set_readback_transport(arms[i])
            for r in range(_SLOTS):
                # Slot r scans feature r alone, so the three records differ in
                # `feature` and a permutation of them is visible.
                var only: List[Int] = [r]
                searcher.set_features(only, record=r)
                searcher.enqueue(hist, params, 1.0, 1.0, record=r)
                # The staging contract: a node's `enqueue` is followed by its
                # download or by an explicit synchronize before the next node
                # stages, because the per-node tables cross in one shared
                # pinned buffer. See `GpuSplitSearcher.enqueue`.
                searcher.synchronize()
            var got = searcher.download_frontier(_SLOTS)
            assert_equal(len(got), _SLOTS)
            if i == 0:
                assert_true(got[0].found)
                # The slots have to actually differ, or this test is three
                # copies of the previous one.
                assert_true(got[0].feature != got[1].feature)
                assert_true(got[1].feature != got[2].feature)
                reference = got.copy()
            else:
                for r in range(_SLOTS):
                    _assert_same_record(
                        got[r],
                        reference[r],
                        readback_transport_name(arms[i]),
                    )


# --- What must not be reachable -------------------------------------------


def test_the_unsafe_and_unimplemented_arms_are_refused() raises:
    """`map` is refused because it is the slowest transport measured; the two
    `nosync` rows are refused because the probe measured 34 of 34 words wrong.

    Refused on every backend and not only on Metal. What makes the `nosync`
    rows wrong elsewhere is unestablished rather than known to be false, and a
    fit is not where that should be discovered."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var ctx = DeviceContext()
        var searcher = GpuSplitSearcher(ctx, 3, 8)
        var refused: List[Int] = [
            READBACK_MAP,
            READBACK_PINNED_PAIR_NOSYNC,
            READBACK_PINNED_ONE_NOSYNC,
        ]
        for i in range(len(refused)):
            var raised = False
            try:
                searcher.set_readback_transport(refused[i])
            except:
                raised = True
            if not raised:
                raise Error(
                    "readback transport ",
                    readback_transport_name(refused[i]),
                    " was accepted by the split searcher and must not be",
                )
            # And the refusal left the live arm alone, so a caller that
            # ignores the error does not silently run the refused one.
            assert_true(searcher.readback != refused[i])


def test_the_unsafe_rows_are_marked_wrong_in_the_table() raises:
    """The refusal above is only as good as the column it fires on. These are
    the two rows the probe measured wrong, and they must stay marked wrong;
    `require_readback_correct` reads nothing else. Runs with no accelerator."""
    assert_false(
        readback_transport(READBACK_PINNED_PAIR_NOSYNC).correct_on_metal
    )
    assert_false(
        readback_transport(READBACK_PINNED_ONE_NOSYNC).correct_on_metal
    )
    assert_equal(readback_transport(READBACK_PINNED_PAIR_NOSYNC).host_waits, 0)
    assert_equal(readback_transport(READBACK_PINNED_ONE_NOSYNC).host_waits, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
