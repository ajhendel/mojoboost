"""Collective communication contract for distributed training.

This module is the whole interface between distributed tree growth and
however ranks actually talk to each other. Tree logic depends on the
`Collective` trait and on nothing else: no socket, no message, and no rank
id appears in the growth code except through this trait. Adding a real
transport means writing one struct here, not editing the trainer.

See docs/distributed.md for the design, and section 5 of it for the five
requirements a conforming transport has to satisfy. Two of them are load
bearing enough to repeat:

- an all-reduce delivers bit-identical bytes to every rank, so no rank can
  reach a different conclusion from the same reduction
- the sum is accumulated in ascending rank order, which is what makes an
  exactly representable dataset produce the same model distributed as on
  one node

`n_local_ranks` and `local_rank` exist so one implementation can host
several ranks in a single process. A real transport returns 1 and its own
rank; `LocalCollective` returns the whole world. The growth code loops over
local ranks and accumulates their contributions in ascending rank order
before calling the all-reduce, so under a real transport that loop runs
once and the code is otherwise identical.

Failure handling lives here too, in `agree_status`. Validation errors are
recorded as status codes rather than raised, reduced across ranks, and then
raised identically on every rank. A run where one rank raises while the
others block forever in a collective it will never call is the failure mode
this exists to prevent. Codes are coarse because a status code is reducible
and a message is not; the code identifies the class of failure and the rank
that hit it.
"""

comptime STATUS_OK = 0
comptime STATUS_SHAPE_MISMATCH = 1
comptime STATUS_INVALID_TARGET = 2
comptime STATUS_INVALID_WEIGHT = 3
comptime STATUS_UNSUPPORTED = 4
comptime STATUS_LAYOUT_MISMATCH = 5
comptime STATUS_INVALID_PARAM = 6


def status_message(code: Int) -> String:
    """Text for a status code. Deliberately coarse: every rank must produce
    the same message from the same reduced code, so the message cannot carry
    rank-local detail."""
    if code == STATUS_OK:
        return "no failure"
    if code == STATUS_SHAPE_MISMATCH:
        return "shard array lengths do not match the shard row count"
    if code == STATUS_INVALID_TARGET:
        return "shard target values are outside the objective's domain"
    if code == STATUS_INVALID_WEIGHT:
        return "shard sample weights are negative or the wrong length"
    if code == STATUS_UNSUPPORTED:
        return "the configuration is not supported by distributed training"
    if code == STATUS_LAYOUT_MISMATCH:
        return "shards disagree about the binned feature layout"
    if code == STATUS_INVALID_PARAM:
        return "the training parameters are invalid"
    return "unrecognized failure code"


trait Collective:
    """Bulk-synchronous collectives over a fixed set of ranks.

    Every rank calls every collective the same number of times and in the
    same order. An all-reduce either delivers the complete result to every
    rank or fails on every rank; there is no partial completion.
    """

    def world_size(self) -> Int:
        """Total number of ranks across all processes."""
        ...

    def rank(self) -> Int:
        """This process's first rank."""
        ...

    def n_local_ranks(self) -> Int:
        """How many ranks this process hosts. 1 for a real transport."""
        ...

    def local_rank(self, index: Int) -> Int:
        """Global rank id of this process's `index`-th local rank."""
        ...

    def allreduce_sum_f64(mut self, mut buf: List[Float64]) raises:
        """Sum `buf` element-wise across ranks, in ascending rank order,
        leaving the identical result in every rank's `buf`."""
        ...

    def allreduce_sum_int(mut self, mut buf: List[Int]) raises:
        """Integer element-wise sum across ranks. Exact at any world size."""
        ...

    def allreduce_max_int(mut self, mut buf: List[Int]) raises:
        """Integer element-wise maximum across ranks."""
        ...

    def barrier(mut self) raises:
        """Return only once every rank has called it."""
        ...


@fieldwise_init
struct LocalCollective(Collective, Copyable, Movable):
    """Every rank of the world hosted in the calling process.

    The all-reduces are the identity, and that is the correct implementation
    of the contract here rather than a stub: the caller has already
    accumulated all `world_size` local contributions in ascending rank
    order, and there is no other process to combine with. By the ascending
    rank order requirement, the values are the ones a conforming
    multi-process transport would produce for the same world size.

    `calls` and `elements` count the reductions issued and the buffer
    elements reduced, which is how the communication schedule is tested
    without a network.
    """

    var n: Int
    var calls: Int
    var elements: Int

    def __init__(out self, world_size: Int):
        self.n = world_size if world_size > 0 else 1
        self.calls = 0
        self.elements = 0

    def world_size(self) -> Int:
        return self.n

    def rank(self) -> Int:
        return 0

    def n_local_ranks(self) -> Int:
        return self.n

    def local_rank(self, index: Int) -> Int:
        return index

    def allreduce_sum_f64(mut self, mut buf: List[Float64]) raises:
        self.calls += 1
        self.elements += len(buf)

    def allreduce_sum_int(mut self, mut buf: List[Int]) raises:
        self.calls += 1
        self.elements += len(buf)

    def allreduce_max_int(mut self, mut buf: List[Int]) raises:
        self.calls += 1
        self.elements += len(buf)

    def barrier(mut self) raises:
        pass


def zeros_f64(size: Int) -> List[Float64]:
    var out = List[Float64](capacity=size)
    out.resize(size, 0.0)
    return out^


def zeros_int(size: Int) -> List[Int]:
    var out = List[Int](capacity=size)
    out.resize(size, 0)
    return out^


def add_into_f64(mut acc: List[Float64], src: List[Float64]) raises:
    """Accumulate one rank's contribution into this process's buffer. Called
    in ascending local rank order, so the association matches the ascending
    rank order a conforming transport uses across processes."""
    if len(acc) != len(src):
        raise Error("contribution length must match the accumulator")
    for i in range(len(acc)):
        acc[i] += src[i]


def add_into_int(mut acc: List[Int], src: List[Int]) raises:
    if len(acc) != len(src):
        raise Error("contribution length must match the accumulator")
    for i in range(len(acc)):
        acc[i] += src[i]


def agree_status[C: Collective](mut comm: C, statuses: List[Int]) raises:
    """Reduce per-rank validation statuses and raise identically everywhere.

    `statuses[i]` is the status of this process's `i`-th local rank, with
    `STATUS_OK` meaning the rank found nothing wrong. If any rank in the
    world reported a failure, every rank raises the same error naming the
    lowest-numbered failing rank and its reason, including ranks that were
    themselves fine.
    """
    if len(statuses) != comm.n_local_ranks():
        raise Error("agree_status needs one status per local rank")
    var world = comm.world_size()
    var vec = zeros_int(world)
    for i in range(len(statuses)):
        var r = comm.local_rank(i)
        if r < 0 or r >= world:
            raise Error("local rank id out of range")
        if statuses[i] > vec[r]:
            vec[r] = statuses[i]
    comm.allreduce_max_int(vec)
    for r in range(world):
        if vec[r] != STATUS_OK:
            raise Error(
                "distributed training failed on rank ",
                r,
                ": ",
                status_message(vec[r]),
            )


def agree_equal_ints[
    C: Collective
](mut comm: C, values: List[Int]) raises -> Int:
    """Check that every rank passed the same integers. Returns the index of
    the first value the ranks disagree about, or -1 when they all agree.

    Every rank gets the same answer, so the caller can turn a disagreement
    into an error that every rank raises. Implemented with a single maximum
    reduction over each value and its negation: the ranks agree about
    `values[i]` exactly when the maximum of the value equals the negated
    maximum of its negation.
    """
    var n = len(values)
    var buf = zeros_int(2 * n)
    for i in range(n):
        buf[2 * i] = values[i]
        buf[2 * i + 1] = -values[i]
    comm.allreduce_max_int(buf)
    for i in range(n):
        if buf[2 * i] != -buf[2 * i + 1]:
            return i
    return -1
