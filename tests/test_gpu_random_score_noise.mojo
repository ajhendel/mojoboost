"""`random_strength` on an accelerator: the half that needs one.

The companion to `tests/test_random_score_noise.mojo`, which holds everything
about this feature that a CPU-only runner can check. What is left here is the
one assertion this lane exists to make and the one machine that can make it:

    **the device's key and the host's key are the same 64-bit word.**

That is stage A of the draw, it is pure integer arithmetic, and it is the
half of `random_strength` that crosses to a Float32 GPU intact. Stage B --
Marsaglia's polar method in Float64 -- does not cross and is not asked to:
Apple GPUs have no Float64 at all, and the polar method's rejection test
would accept a pair on one backend and reject it on the other within an ulp
of the unit circle, which sends the two backends onto entirely different
draws rather than draws a few ulps apart. So the normal is drawn once on the
host and the plane is uploaded, and what is asserted below is that the device
consuming that plane picks the split the replica picks.

Both tables of pinned keys are imported from the CPU file rather than copied,
so the host and the device cannot be pinned to different tuples.

Skips (passing) with no accelerator, as every `test_gpu_*` file does.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from max.gpu.host import DeviceBuffer, DeviceContext

from mojotrees.gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    SplitNodeRequest,
    gpu_random_score_stream,
    random_score_key_probe,
    random_score_plane,
    reference_search,
)
from mojotrees.monotone import OutputBounds

from test_random_score_noise import KEY_QUERIES, KEY_WORDS


def _zeroed(n: Int) -> List[Int32]:
    var out = List[Int32](capacity=n)
    out.resize(n, Int32(0))
    return out^


def _two_feature_words(n_bins: Int) -> List[Int32]:
    """The same histogram shape the CPU file uses: two ordinal features whose
    best splits sit at different bins."""
    var size = 2 * n_bins
    var words = _zeroed(3 * size)
    for f in range(2):
        for b in range(n_bins):
            var turn = n_bins // 3 if f == 0 else (2 * n_bins) // 3
            var i = f * n_bins + b
            words[i] = Int32(6 - 2 * b if b < turn else b - 3)
            words[size + i] = Int32(2 + (b % 3))
            words[2 * size + i] = Int32(5 + (b % 4))
    return words^


def _plain_params() -> GpuSplitParams:
    return GpuSplitParams.default()


def _assert_same_decision(
    got: GpuSplitRecord, want: GpuSplitRecord
) raises:
    """Everything discrete exactly; the gain to Float32 precision, which is
    the standing contract between this module's replica and its kernels."""
    assert_equal(got.found, want.found)
    assert_equal(got.feature, want.feature)
    assert_equal(got.bin, want.bin)
    assert_equal(got.ordinal, want.ordinal)
    assert_equal(got.default_left, want.default_left)
    assert_equal(got.left.count, want.left.count)
    assert_equal(got.right.count, want.right.count)
    assert_almost_equal(got.gain, want.gain, atol=1e-4, rtol=1e-4)


def test_device_key_matches_the_host_key_bit_for_bit() raises:
    """**The lane's assertion.** `gpu_random_score_stream` evaluated on the
    accelerator returns the identical 64-bit word it returns on the host, for
    every pinned tuple, including a negative seed and a tuple at the far end
    of the ranges.

    This is what makes `random_strength` a shared rule rather than two
    similar ones. Everything downstream of the key is either host-computed
    (the normal, and therefore the plane) or already inside this module's
    documented Float32 contract (the gain the plane is added to), so with the
    key identical and the plane uploaded, the two backends noise the same
    candidate by the same amount.
    """
    if not has_accelerator():
        return
    var queries = KEY_QUERIES()
    var want = KEY_WORDS()
    var device = random_score_key_probe(queries)
    assert_equal(len(device), len(want))
    for i in range(len(want)):
        var b = 5 * i
        # Device against host, which is the crossing.
        assert_equal(
            device[i],
            gpu_random_score_stream(
                queries[b],
                queries[b + 1],
                queries[b + 2],
                queries[b + 3],
                queries[b + 4],
            ),
        )
        # Device against the pinned literal, so a failure says whether the
        # device moved or the host did.
        assert_equal(device[i], want[i])


def test_device_key_probe_refuses_a_ragged_query_list() raises:
    if not has_accelerator():
        return
    with assert_raises(contains="five ints per query"):
        _ = random_score_key_probe(List[Int]([1, 2, 3]))


def test_noise_off_leaves_the_record_alone() raises:
    """A searcher that never sets a strength, and one that sets it to zero,
    return the same record: the default is a strict no-op and not a small
    perturbation."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 12
        var words = _two_feature_words(n_bins)
        var searcher = GpuSplitSearcher(2, n_bins)
        searcher.upload_histogram(words)
        var untouched = searcher.search(_plain_params(), 1.0, 1.0)
        searcher.set_random_score(0.0, 20260816, 3)
        var zeroed = searcher.search(_plain_params(), 1.0, 1.0)
        assert_equal(searcher.random_score_stdev(), 0.0)
        assert_equal(zeroed.gain, untouched.gain)
        assert_equal(zeroed.bin, untouched.bin)
        assert_equal(zeroed.feature, untouched.feature)
        assert_true(untouched.found)


def test_device_under_noise_matches_the_replica() raises:
    """The device search with a staged plane picks the split the replica
    picks with the same plane, over several nodes and two seeds.

    The plane is built once, by `random_score_plane`, and handed to both, so
    what is compared is the two scans -- the kernel's placement of the addend
    against the replica's -- and not two draws."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 12
        var words = _two_feature_words(n_bins)
        var features = List[Int]([0, 1])
        var searcher = GpuSplitSearcher(2, n_bins)
        searcher.upload_histogram(words)
        for seed in range(2):
            for node in range(4):
                var stdev = 0.25 + 0.5 * Float64(seed)
                searcher.set_random_score(stdev, 7 + seed, 2)
                var got = searcher.search(
                    _plain_params(),
                    1.0,
                    1.0,
                    OutputBounds.unbounded(),
                    0,
                    node,
                )
                var plane = random_score_plane(
                    stdev, 7 + seed, 2, node, features, n_bins
                )
                var want = reference_search(
                    words,
                    2,
                    n_bins,
                    1.0,
                    1.0,
                    _plain_params(),
                    noise=plane,
                )
                _assert_same_decision(got, want)


def test_device_under_noise_is_deterministic() raises:
    """Run to run, the same node under the same seed returns the same
    record. The GPU backend's standing guarantee, which the noise must not
    weaken: the draw is keyed and the plane is a pure function of the key, so
    nothing here depends on launch order."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 12
        var words = _two_feature_words(n_bins)
        var searcher = GpuSplitSearcher(2, n_bins)
        searcher.upload_histogram(words)
        searcher.set_random_score(0.75, 4242, 11)
        var first = searcher.search(
            _plain_params(), 1.0, 1.0, OutputBounds.unbounded(), 0, 6
        )
        var second = searcher.search(
            _plain_params(), 1.0, 1.0, OutputBounds.unbounded(), 0, 6
        )
        assert_equal(first.gain, second.gain)
        assert_equal(first.bin, second.bin)
        assert_equal(first.feature, second.feature)
        assert_equal(first.ordinal, second.ordinal)


def test_a_big_enough_plane_moves_the_device_decision() raises:
    """The regularizer actually reaches the kernel: a strength large against
    this histogram's gains changes the split the device returns, and the
    replica agrees on which one it changed to. A test that only checked
    agreement could pass with the noise never read."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 12
        var words = _two_feature_words(n_bins)
        var features = List[Int]([0, 1])
        var searcher = GpuSplitSearcher(2, n_bins)
        searcher.upload_histogram(words)
        var plain = searcher.search(_plain_params(), 1.0, 1.0)
        assert_true(plain.found)
        var moved = 0
        for node in range(12):
            searcher.set_random_score(8.0, 31337, 0)
            var got = searcher.search(
                _plain_params(),
                1.0,
                1.0,
                OutputBounds.unbounded(),
                0,
                node,
            )
            var plane = random_score_plane(
                8.0, 31337, 0, node, features, n_bins
            )
            var want = reference_search(
                words, 2, n_bins, 1.0, 1.0, _plain_params(), noise=plane
            )
            _assert_same_decision(got, want)
            if got.ordinal != plain.ordinal:
                moved += 1
        assert_true(moved > 0)


def test_every_scan_kernel_reads_the_same_plane() raises:
    """All three scan shapes -- the serial one, the wide one, and the wide
    one written with `block` collectives -- return the same record under the
    same noise plane, and the replica's.

    The wide scan is the reason the draw is keyed by bin rather than streamed
    in scan order. A thread there starts in the middle of a feature's bin
    range, so a stream that advanced per candidate would hand it the wrong
    draws; a key cannot, and this is where that shows."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 12
        var words = _two_feature_words(n_bins)
        var features = List[Int]([0, 1])
        var plane = random_score_plane(1.5, 606, 8, 2, features, n_bins)
        var want = reference_search(
            words, 2, n_bins, 1.0, 1.0, _plain_params(), noise=plane
        )
        for arm in range(3):
            var searcher = GpuSplitSearcher(2, n_bins)
            searcher.wide_scan = arm > 0
            searcher.set_primitives(arm == 2)
            searcher.upload_histogram(words)
            searcher.set_random_score(1.5, 606, 8)
            var got = searcher.search(
                _plain_params(), 1.0, 1.0, OutputBounds.unbounded(), 0, 2
            )
            _assert_same_decision(got, want)


def test_a_frontier_carries_one_plane_per_node() raises:
    """Batched, each record gets its own node's draw. A frontier that shared
    one plane would give two leaves the same noise, which is exactly the
    failure keying by node id prevents."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 12
        var words = _two_feature_words(n_bins)
        var features = List[Int]([0, 1])
        var ctx = DeviceContext()
        var hist = ctx.enqueue_create_buffer[DType.int32](len(words))
        ctx.enqueue_copy(dst_buf=hist, src_ptr=words.unsafe_ptr())
        ctx.synchronize()
        var searcher = GpuSplitSearcher(ctx, 2, n_bins, max_records=3)
        searcher.set_random_score(2.0, 909, 5)
        var nodes = List[SplitNodeRequest]()
        for i in range(3):
            nodes.append(SplitNodeRequest(node=1 + i))
        var got = searcher.search_frontier(
            hist, nodes, _plain_params(), 1.0, 1.0
        )
        assert_equal(len(got), 3)
        for i in range(3):
            var plane = random_score_plane(
                2.0, 909, 5, 1 + i, features, n_bins
            )
            var want = reference_search(
                words, 2, n_bins, 1.0, 1.0, _plain_params(), noise=plane
            )
            _assert_same_decision(got[i], want)


def test_a_frontier_without_node_ids_is_refused() raises:
    """-1 is "not supplied" and the batch is refused rather than drawing
    every leaf of the level from one stream."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 8
        var words = _two_feature_words(n_bins)
        var ctx = DeviceContext()
        var hist = ctx.enqueue_create_buffer[DType.int32](len(words))
        ctx.enqueue_copy(dst_buf=hist, src_ptr=words.unsafe_ptr())
        ctx.synchronize()
        var searcher = GpuSplitSearcher(ctx, 2, n_bins, max_records=2)
        searcher.set_random_score(1.0, 1, 1)
        var nodes = List[SplitNodeRequest]()
        nodes.append(SplitNodeRequest())
        nodes.append(SplitNodeRequest())
        with assert_raises(contains="node id"):
            _ = searcher.search_frontier(
                hist, nodes, _plain_params(), 1.0, 1.0
            )


def test_a_search_without_a_node_id_is_refused() raises:
    """The single-node entry point takes the same rule."""
    if not has_accelerator():
        return
    comptime if not has_accelerator():
        raise Error("no accelerator")
    else:
        var n_bins = 8
        var words = _two_feature_words(n_bins)
        var searcher = GpuSplitSearcher(2, n_bins)
        searcher.upload_histogram(words)
        searcher.set_random_score(1.0, 2, 3)
        with assert_raises(contains="node id"):
            _ = searcher.search(_plain_params(), 1.0, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
