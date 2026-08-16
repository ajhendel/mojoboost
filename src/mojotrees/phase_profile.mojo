"""Where a boosting round's wall time goes, split by phase and by how big the
node was. One vocabulary, both backends.

This module is an instrument, not an optimization. Nothing in it has been
measured, nothing in it makes a claim about any hardware, and no coefficient
in it came from a benchmark: it reads clocks and increments integers. It
exists because the two things this repository most wants to fix are both
un-attributed. The five-lane GPU round netted three to four percent end to
end, so the obvious histogram work has stopped paying; and on a matched
shape the CPU backend loses to LightGBM running on a single thread. Neither
number tells anyone where the time is. The next change has to be argued from
attribution.

The question, stated exactly: for one boosting round, how is wall time
divided between histogram construction, the active-row partition, split
search, the per-round score update and gradient fill, and (on the device)
transfers plus host synchronization -- **broken down by node size class**.

The hypothesis it was built to test, and can refute
---------------------------------------------------
On the device the suspicion is that large nodes are fine and the tail of
small nodes is dominated by per-launch and per-synchronization cost rather
than by row work. That is a hypothesis and this module does not assume it.
What makes it refutable is that every bucket carries **counts as well as
time**: dispatches and host synchronizations per (phase, class), and rows,
row slots, and histogram cells per (phase, class) so a cost per unit can be
divided out. If the small classes hold most of the dispatches and most of
the time while holding almost none of the rows, the hypothesis survives; if
the per-row cost is flat across the classes, it is dead, and the counts say
which without any further run. Time alone would say neither.

On the host the suspicion is different -- a serial full-tree score update,
and a three-plane histogram against LightGBM's two -- and the same counters
answer it. `PROF_SCORE_UPDATE` is its own phase, so it is confirmed or
refuted on its own line. The three-plane question becomes a bytes-per-second
figure: `cells` counts the `n_features * n_bins` cells a build touched, and a
`Histogram` cell is a Float64 gradient, a Float64 hessian, and an Int count,
so `cells * 24` is bytes written and `nanos` is what it took. What this
module deliberately does **not** do is time the three plane writes
separately. That would mean a clock read inside the accumulation loop, which
runs under `sync_parallelize` where a shared counter is a data race and a
per-iteration clock read would cost more than the write it timed. The cell
count is the honest instrument for that question; the plane split is not
available without changing the thing being measured.

What "phase" means here, and how it relates to the models next door
-------------------------------------------------------------------
`gpu_runtime.PhaseCounters` counts a *session's* phases -- compile,
allocation, transfer, kernel, synchronization, cleanup -- across a whole
`DeviceContext` lifetime, with no notion of a node and no notion of the host
backend. It answers "what did this session spend its life doing". This
module answers "what did this round spend its time on, and for which nodes",
so it has its own grower-shaped vocabulary and does not extend that one.
Neither is derivable from the other.

`bench/bench_profile.mojo` is the third relative and the closest one. It
times each CPU stage twice on synthetic inputs, serial against auto, and
reports the multicore speedup of the stage in isolation. That is a
microbenchmark: it says what a stage costs when run on its own, not what
share of a real fit it takes, and it never sees a small node because it
builds a root, a half, and a tenth. This module is the in-run complement --
same stages, real trees, every node, and a size breakdown -- and neither
replaces the other.

`HybridCosts`, in the hybrid leaf scheduler deleted on 2026-08-16, was the
fourth, and the buckets here were made deliberately legible against it. Its
calibrated coefficients survive in
`bench/results/apple_m4_hybrid_costs_2026-08-15.md`. That model priced a
node's build as a fixed part
(`launch_nanos`, `sync_nanos`, `host_fixed_nanos`), a per-cell part
(`convert_nanos_per_kcell`, `host_zero_nanos_per_kcell`) and a per row-slot
part (`device_nanos_per_krow_slot`, `host_nanos_per_krow_slot`), where a *row
slot* is one (row, active feature) pair. So this module records rows, row
slots, and cells side by side and reports nanoseconds per thousand of each,
which is the unit those coefficients are stated in, and a profile row can
still be read against that recorded calibration without conversion. The
comparison was never done, and nothing here says how it would come out.

`PROF_HIST_ALLOC`, `PROF_SUBTRACT`, and `PROF_CONVERT` are separate from
`PROF_HISTOGRAM` for the same reason: each is a per-cell cost where the
accumulate is a per-row-slot cost, and the ratio between those two is exactly
what the small-node question turns on. Folding them together would hide it.

The size classes are a reporting choice, not a measured crossover
-----------------------------------------------------------------
`classify_node` puts a node in one of five classes from its row count as a
fraction of the tree's root row count. **The boundaries are octaves picked so
that a default 31-leaf leaf-wise tree spreads across them, and nothing has
measured a crossover at any of them.** They are not thresholds and no
decision anywhere is taken from one; changing them changes how a profile
reads and changes no fit. `device_policy.crossover_rules()` and
`apple_gpu_policy.CrossoverInputs` are where this repository puts numbers a
measurement would have to license, and none of these is one of those.

The denominator is the *tree's* root row count, not `data.n_rows`. Under
bagging or GOSS the root holds the sample and every node beneath it is a
fraction of the sample, so classifying against the dataset would push a whole
sampled tree down a class or two and make two runs at different
`bagging_fraction` incomparable. `root_rows` is what the tree actually
partitions. Round-level work that belongs to no node (the gradient fill, the
score update) is charged at `root_rows` and so lands in `CLASS_ROOT`, which
is stated here because a reader will otherwise wonder why the root class
holds work no node did.

The classes partition the row counts with no gap and no overlap: for any
`node_rows` in `[0, root_rows]` exactly one class matches, which
`tests/test_gpu_phase_profile.mojo` asserts exhaustively rather than by
sampling.

Dispatch and launch counts
--------------------------
`dispatches` is one counter with two readings, which is what makes the
vocabulary shared: on the device it counts kernel launches, and on the host
it counts `parallel.dispatch_*` calls, which is one `sync_parallelize` fan-out
each. Both are the same thing for the purpose at hand -- a fixed cost paid
per unit of work regardless of how much work the unit holds -- and both are
what a small-node tail would be bound by.

What one host dispatch costs before it runs anything is worth writing down
once, because the profile reports the count and not that arithmetic:
`parallel.plan_tasks` reads `MOJOTREES_NUM_WORKERS` and, in auto mode,
`MOJOTREES_PARALLEL_MIN_OPS`, and then calls `apple_cpu_policy.cpu_profile()`,
which is `CpuProfile.detect()` and re-reads the machine's core counts every
time. So a host dispatch in auto mode is at least two `getenv` calls and one
core re-detection on top of the fan-out. Multiply the reported dispatch count
by that to price it. This module counts the dispatches and does not count the
`getenv` calls, because counting those would need mutable state inside
`parallel.mojo`, Mojo 1.0 has no module-level mutable state, and threading a
counter through every dispatch call site in the package is a change to the
thing being measured. That is a real limitation and it is stated rather than
papered over.

The counts themselves are structural: they are read off the `enqueue_function`
and `dispatch_*` calls in the source and charged by the call site. See
`PARTITION_LAUNCHES` and the constants beside it. **If those functions change
their launch structure, these constants become wrong and no test will catch
it**, because nothing here can observe a driver or a thread pool. They are
written in one place so there is one place to fix.

Free when off
-------------
`MOJOTREES_PHASE_PROFILE` is unset by default and an unset run pays nothing.
The structural argument, which is what makes that checkable rather than
asserted:

- `clock()` tests one Bool and returns 0 without reading `perf_counter_ns`,
  so a disabled profile performs no clock read anywhere.
- `charge()` returns on the same Bool **before** it classifies, before it
  indexes, and before it adds, so a disabled profile performs no counter
  update either. This is deliberately unlike `gpu_runtime.PhaseCounters`,
  whose `record` keeps counting calls when disabled because its lifecycle
  tests read those counts; nothing reads these counts unless the profile is
  on, so there is no reason to pay for them.
- Every argument a call site computes *for* a charge is either already in
  hand at that point in the grower (the node's row count, the active feature
  count, the bin count) or is itself guarded by `profile.enabled()`. The one
  guarded case is the device launch count, which costs a policy derivation,
  and it is guarded at its call site in `train_gpu.mojo`.
- The buckets are allocated in `__init__` whether or not the profile is on:
  `10 * 5` integers seven times over, 350 words, once per fit or per tree.
  That and one `getenv` is the entire cost of an off profile.

`tests/test_gpu_phase_profile.mojo` asserts the counter half of that
directly, by charging an off profile a large duration and a large count and
requiring every bucket to stay zero. The clock half is structural and is
visible in `clock` in four lines.

Modes
-----
`MOJOTREES_PHASE_PROFILE` takes a word, following
`MOJOTREES_GPU_HIST_STRATEGY` and `MOJOTREES_GPU_MIN_TILES` rather than
`_env_int`, because the choice is not a number and folding it through an
integer parse would turn a typo into a default:

- unset, `0`, or `off` -- off, and the run is the run that shipped.
- `1` or `async` -- on, with no fences added. This is the honest default
  because it perturbs nothing. On the host every phase is synchronous and the
  clocks mean what they say. On the device the queue is in order and kernel
  launches are asynchronous, so a phase's clock measures the host-side
  enqueue plus whatever device execution happened to drain into it, and the
  phases that wait absorb the rest. That is not a defect for the question
  being asked -- if the small-node tail is launch-bound, enqueue time and
  dispatch count are precisely the evidence -- but a reader must not read
  `PROF_HISTOGRAM` on a device arm under `async` as device accumulation time.

  Two Metal-specific corrections to that picture, both measured by
  disassembly and written out in `docs/GPU_PORTABILITY.md` section 6. First,
  the download is not the only phase that drains: `enqueue_copy` is a
  synchronous full-queue drain in *both* directions, so every upload charged
  to `PROF_TRANSFER` fences too, and an `async` profile on Metal is already
  partly fenced by its own uploads. That is a statement about *ordering*, not
  about cost: section 6.1.1 records that a drain of a queue holding nothing
  costs nothing, and that the download is a **round trip** where the uploads
  are not. So an upload's fencing effect is what shifts device time into
  whichever phase's clock closes over it; it is not itself a charge, and a
  `PROF_TRANSFER` total should not be read as a count of copies times a
  per-copy price. Second, a launch stream longer
  than 64 command buffers backpressures inside `objc_msgSend`, which this
  profiler counts as enqueue time with no attribution, so a rising enqueue
  clock on a long unwaited stream is the host blocking and not the device
  slowing down. Neither is a reason to distrust the counters; both are
  reasons not to read a clock as device time.
- `fenced` -- on, and the device growers drain the queue after the partition
  and after the histogram enqueue so each phase's clock closes over its own
  device execution. This buys per-phase device time and pays two extra host
  synchronizations per split for it, which against the launch-cost figures
  quoted in `train_gpu._device_search_resident` is a real perturbation of the
  schedule. A fenced profile and an async profile are not comparable to each
  other, and the report names which mode produced it. On the host backend
  `fenced` and `async` are the same run, since there is no queue to drain.

Neither mode changes a model. A fence is a wait; a charge is an integer add.
`tests/test_gpu_phase_profile.mojo` asserts that predictions are byte
identical across off, async, and fenced.

Backends
--------
The vocabulary is backend-neutral by construction: this module imports no GPU
module, opens no `DeviceContext`, and names nothing only a device has.
`PROF_TRANSFER`, `PROF_HOST_SYNC`, and `PROF_CONVERT` are simply zero on a
host fit, and the rest are the same phases both growers walk. Which entry
points are wired is recorded on the report's `label` and in this lane's
commit message, not asserted here.
"""

from std.os import getenv
from std.time import perf_counter_ns


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

comptime PROF_HISTOGRAM = 0
"""Histogram accumulation over a node's rows: the device kernels a build
enqueues (`histogram_gpu.enqueue_leaf`), or the host scatter-add
(`tree._hist_full` / `tree._hist_subset`). Not the buffer, not the download,
not the conversion, not the subtraction -- each of those is its own phase, so
that a per-cell cost is never mistaken for a per-row one."""

comptime PROF_HIST_ALLOC = 1
"""Getting a histogram buffer and putting it in a known state: `_HistPool.take`
on the host, including the `Histogram.zeroed` allocation when the pool is
empty and the `reset` pass when the caller needs one. Independent of the
node's rows and proportional to `n_features * n_bins`, which is why it is not
inside `PROF_HISTOGRAM`."""

comptime PROF_SUBTRACT = 2
"""Deriving a sibling histogram from its parent and the built child
(`histogram.subtract_histogram_into`). Per cell, not per row, and it is what
the subtraction trick buys the second child instead of an accumulate."""

comptime PROF_PARTITION = 3
"""Routing a node's rows to its two children: `tree.partition_rows_into`,
including the two child row lists it allocates and fills, or
`gpu_active_rows.enqueue_partition` on the device. The allocation is inside
this phase on purpose -- on the host it is two `List[Int]` of the parent's
size per split, and separating it from the routing would need a change inside
`partition_rows_into`."""

comptime PROF_SPLIT_SEARCH = 4
"""Choosing a split from a histogram: the host scan in `tree._search`, or the
device scan and reduction in `gpu_split_search`."""

comptime PROF_CONVERT = 5
"""Host work over downloaded histogram cells: `histogram_gpu.histogram_from_host`
dequantizing the fixed-point planes into Float64. Device paths only; zero on a
host fit."""

comptime PROF_TRANSFER = 6
"""Bytes crossing the bus, and the wait that copy carries with it.
`histogram_gpu.download_raw` and `gpu_split_search.download_frontier` each
enqueue a copy and then synchronize inside one call, so their time lands here
as a single charge and the separate `syncs` counter is what says a wait
happened. Splitting the copy from its wait would need a change inside those
two methods, which this lane did not make."""

comptime PROF_HOST_SYNC = 7
"""A host synchronization that is not part of a transfer: a fence this
instrument inserted under `fenced`, or a drain a grower performs on its own
account. Under `async` this is normally zero, and that it is zero is
information."""

comptime PROF_GRAD_FILL = 8
"""The round's per-row gradients and hessians (`boosting._fill_grad_hess`,
plus any GOSS rescaling that follows it). Round-level, charged at the root's
row count."""

comptime PROF_SCORE_UPDATE = 9
"""The round's raw-score update: walking the finished tree for every training
row and adding `learning_rate * value`. Round-level, charged at the root's
row count, and its own phase because a full-tree traversal per row per round
is a cost that belongs to no node and would otherwise disappear into the
unattributed remainder."""

comptime N_PROFILE_PHASES = 10


def profile_phase_name(phase: Int) -> String:
    if phase == PROF_HISTOGRAM:
        return String("histogram")
    if phase == PROF_HIST_ALLOC:
        return String("hist_alloc")
    if phase == PROF_SUBTRACT:
        return String("subtract")
    if phase == PROF_PARTITION:
        return String("partition")
    if phase == PROF_SPLIT_SEARCH:
        return String("split_search")
    if phase == PROF_CONVERT:
        return String("convert")
    if phase == PROF_TRANSFER:
        return String("transfer")
    if phase == PROF_HOST_SYNC:
        return String("host_sync")
    if phase == PROF_GRAD_FILL:
        return String("grad_fill")
    if phase == PROF_SCORE_UPDATE:
        return String("score_update")
    return String("unknown")


# ---------------------------------------------------------------------------
# Node size classes
# ---------------------------------------------------------------------------
#
# A reporting choice. See the module docstring: these boundaries are octaves,
# nothing has measured a crossover at any of them, and no decision anywhere in
# this repository is taken from one.

comptime CLASS_ROOT = 0
"""Every row the tree was given, and the class round-level work is charged
at. Defined by an equality rather than a fraction, because the root is the
one node whose cost is not in the tail by construction, and lumping it in
with `large` would let one node's row work drown out the thirty after it."""

comptime CLASS_LARGE = 1
"""More than an eighth of the root's rows, but not all of them."""

comptime CLASS_MEDIUM = 2
"""More than a sixty-fourth of the root's rows, up to an eighth."""

comptime CLASS_SMALL = 3
"""More than a five-hundred-and-twelfth of the root's rows, up to a
sixty-fourth."""

comptime CLASS_TINY = 4
"""A five-hundred-and-twelfth of the root's rows or fewer, including a node
with no rows at all. This is where a launch-bound or dispatch-bound tail
would live."""

comptime N_NODE_CLASSES = 5

# The boundaries, as the inverse of the fraction of `root_rows` at which each
# class opens. Written as inverses so `classify_node` compares integers and
# two runs on two machines classify identically, with no float rounding
# anywhere in the path.
comptime LARGE_MIN_INVERSE = 8
comptime MEDIUM_MIN_INVERSE = 64
comptime SMALL_MIN_INVERSE = 512


def node_class_name(cls: Int) -> String:
    if cls == CLASS_ROOT:
        return String("root")
    if cls == CLASS_LARGE:
        return String("large")
    if cls == CLASS_MEDIUM:
        return String("medium")
    if cls == CLASS_SMALL:
        return String("small")
    if cls == CLASS_TINY:
        return String("tiny")
    return String("unknown")


def node_class_bounds() -> String:
    """The boundaries, as the report prints them. Written once so the header
    line and the class docstrings cannot drift from `classify_node`."""
    return String(
        "root=all large>1/",
        LARGE_MIN_INVERSE,
        " medium>1/",
        MEDIUM_MIN_INVERSE,
        " small>1/",
        SMALL_MIN_INVERSE,
        " tiny<=1/",
        SMALL_MIN_INVERSE,
    )


def classify_node(node_rows: Int, root_rows: Int) -> Int:
    """Which size class a node of `node_rows` rows falls in, against a tree
    whose root held `root_rows`.

    Total and disjoint over `[0, root_rows]`: every row count matches exactly
    one class, and the class is monotone non-decreasing as the row count falls,
    so a child is never in a larger class than its parent. A `node_rows` above
    `root_rows` cannot happen in a grower, since a child holds a subset of its
    parent's rows, and is classified `CLASS_ROOT` rather than raising: an
    instrument that can abort a fit is worse than one that mis-files a row it
    should never have seen.

    A `root_rows` of zero or less has no fractions to take, so everything is
    the root. That reaches here only from a degenerate tree.
    """
    if root_rows <= 0:
        return CLASS_ROOT
    if node_rows >= root_rows:
        return CLASS_ROOT
    if node_rows * LARGE_MIN_INVERSE > root_rows:
        return CLASS_LARGE
    if node_rows * MEDIUM_MIN_INVERSE > root_rows:
        return CLASS_MEDIUM
    if node_rows * SMALL_MIN_INVERSE > root_rows:
        return CLASS_SMALL
    return CLASS_TINY


# ---------------------------------------------------------------------------
# Structural dispatch counts
# ---------------------------------------------------------------------------
#
# Counts of `enqueue_function` and `dispatch_*` calls, read off the code that
# makes them. Nothing here is timed and nothing here is a rate. Every one of
# these can go stale silently if the function it counts is restructured; they
# live here so there is one place to fix.

comptime PARTITION_LAUNCHES = 4
"""Kernel launches one device row partition costs.

`gpu_active_rows.enqueue_partition`: a flag-and-scan pass, a block-sum scan, a
scatter, and a copy back. Both scan arms issue two launches for the scan --
the `block.prefix_sum` primitive arm replaces two kernels with two others --
so the count does not depend on `scan_primitives`, which is why this is a
constant rather than a query."""

comptime SPLIT_SEARCH_DEVICE_LAUNCHES = 2
"""Kernel launches one device split-search *batch* costs: one scan over the
slots and one reduction over the records
(`gpu_split_search.GpuSplitSearcher._launch`). Per batch, not per node, which
is the point of `enqueue_frontier`; a leaf-wise tree makes those the same
number and a depth-wise level does not."""

comptime HOST_HIST_DISPATCHES = 2
"""`parallel.dispatch_*` calls one host histogram build makes: the zeroing
pass (`histogram._zero_range`) and the accumulation
(`build_histogram_into` / `build_histogram_subset_into`, one
`dispatch_feature_ranges` each).

Not three. `build_histogram_subset_into_scratch` adds a `dispatch_rows` to
gather the subset's gradient/hessian pairs, but only when the compaction
policy elects it, and this constant is the shape of the path that always
runs. A compacted build therefore under-reports its dispatches by one, which
is stated here rather than guessed at, and is the sort of thing a reader
should check before pricing the count."""

comptime HOST_SUBTRACT_DISPATCHES = 1
"""`histogram.subtract_histogram_into` fans its element-wise subtraction out
once (`dispatch_rows`)."""

comptime HOST_PARTITION_DISPATCHES = 0
"""`tree.partition_rows_into` runs serially: it imports no dispatcher and
makes no fan-out. Zero is the measurement-relevant fact here, not an absence
of instrumentation, which is why it is written down."""

comptime HOST_SPLIT_SEARCH_DISPATCHES = 0
"""`split.find_best_split` and `tree._search` run serially: split.mojo imports
nothing from parallel.mojo."""


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

comptime PROFILE_OFF = 0
comptime PROFILE_ASYNC = 1
comptime PROFILE_FENCED = 2


def profile_mode_name(mode: Int) -> String:
    if mode == PROFILE_OFF:
        return String("off")
    if mode == PROFILE_ASYNC:
        return String("async")
    if mode == PROFILE_FENCED:
        return String("fenced")
    return String("unknown")


def env_profile_mode() raises -> Int:
    """`MOJOTREES_PHASE_PROFILE`, as a mode.

    Unset, empty, `0`, and `off` are off. `1` and `async` are the unfenced
    instrument. `fenced` adds the device drains described in the module
    docstring. Anything else raises rather than defaulting to off: a caller
    who set the variable meant to profile, and silently handing them an
    unprofiled run because they typed `asynch` would waste the run they set it
    for. Same reason `histogram_gpu`'s transfer route raises on an unknown
    word rather than falling back.
    """
    var raw = getenv("MOJOTREES_PHASE_PROFILE")
    if raw == "" or raw == "0" or raw == "off":
        return PROFILE_OFF
    if raw == "1" or raw == "async":
        return PROFILE_ASYNC
    if raw == "fenced":
        return PROFILE_FENCED
    raise Error(
        "MOJOTREES_PHASE_PROFILE must be one of off, async (or 1), or fenced;"
        " got '",
        raw,
        "'",
    )


# ---------------------------------------------------------------------------
# Scope: what one report covers
# ---------------------------------------------------------------------------

comptime SCOPE_TREE = 0
comptime SCOPE_FIT = 1


def profile_scope_name(scope: Int) -> String:
    if scope == SCOPE_TREE:
        return String("tree")
    if scope == SCOPE_FIT:
        return String("fit")
    return String("unknown")


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


def _pct(numerator: Int, denominator: Int) -> Float64:
    """`numerator / denominator` as a percentage, three decimal places. Zero
    when the denominator is zero, which is what an off or empty profile
    reports and is not a measurement."""
    if denominator == 0:
        return 0.0
    var scaled = Float64(numerator) * 100000.0 / Float64(denominator)
    return Float64(Int(scaled + 0.5)) / 1000.0


def _per_thousand(nanos: Int, units: Int) -> Float64:
    """Nanoseconds per thousand units, three decimal places, or 0.0 when no
    units were counted.

    Per *thousand* because that is the unit every rate in the recorded M4
    calibration (`bench/results/apple_m4_hybrid_costs_2026-08-15.md`) is
    stated in, so a profile row and a calibrated coefficient can be read
    against each other with no conversion.
    """
    if units == 0:
        return 0.0
    var scaled = Float64(nanos) * 1000000.0 / Float64(units)
    return Float64(Int(scaled + 0.5)) / 1000.0


struct PhaseProfile(Copyable, Movable):
    """Wall time and counts per (phase, node size class), for one tree or one
    fit.

    Seven parallel bucket arrays rather than an array of a seven-field struct:
    a charge writes at most seven words at one index, and keeping the columns
    apart lets `report` sum one without walking a struct. There are
    `N_PROFILE_PHASES * N_NODE_CLASSES` buckets, which is fifty.

    Every mutator returns immediately when `mode` is `PROFILE_OFF`, checked
    before any other work, so an unprofiled fit pays one Bool test per call
    site and nothing else. See the module docstring for why that is stronger
    than what `gpu_runtime.PhaseCounters` does, and for what it is that makes
    the claim structural rather than asserted.
    """

    var mode: Int
    var scope: Int
    var label: String
    """What produced this report: a trainer name, an arm name, whatever tells
    two blocks apart in a diff."""

    var root_rows: Int
    """The denominator every classification in the current tree is taken
    against, set by `begin_tree`. A fit-scope profile carries the last tree's;
    under the trainers wired here every tree in a fit has the same root, being
    the full dataset or a sample of fixed fraction."""

    var dataset_rows: Int
    var trees: Int
    var nodes: Int
    """Nodes charged at least one histogram build, so a report can say how
    many nodes a class held rather than only how many charges it took."""

    var wall_nanos: Int
    """Wall time of everything the profile was open across, summed over
    `begin_tree`/`end_tree` pairs and any `note_wall` the caller adds. What no
    phase claimed is the difference between this and the attributed total, and
    the report prints that difference rather than hiding it."""

    var nanos: List[Int]
    var calls: List[Int]
    var dispatches: List[Int]
    """Kernel launches on a device path, `parallel.dispatch_*` fan-outs on a
    host path. See the module docstring: one counter, two readings, and both
    are the fixed cost a unit of work pays regardless of its size."""

    var syncs: List[Int]
    var rows: List[Int]
    var slots: List[Int]
    """(row, active feature) pairs: `rows * n_active_features`."""

    var cells: List[Int]
    """Histogram cells touched: `n_features * n_bins`. A `Histogram` cell is a
    Float64 gradient, a Float64 hessian, and an Int count, so `cells * 24` is
    the bytes a full pass writes."""

    def __init__(
        out self,
        mode: Int = PROFILE_OFF,
        scope: Int = SCOPE_TREE,
        var label: String = String(""),
    ):
        self.mode = mode
        self.scope = scope
        self.label = label^
        self.root_rows = 0
        self.dataset_rows = 0
        self.trees = 0
        self.nodes = 0
        self.wall_nanos = 0
        var n = N_PROFILE_PHASES * N_NODE_CLASSES
        self.nanos = List[Int](capacity=n)
        self.calls = List[Int](capacity=n)
        self.dispatches = List[Int](capacity=n)
        self.syncs = List[Int](capacity=n)
        self.rows = List[Int](capacity=n)
        self.slots = List[Int](capacity=n)
        self.cells = List[Int](capacity=n)
        for _ in range(n):
            self.nanos.append(0)
            self.calls.append(0)
            self.dispatches.append(0)
            self.syncs.append(0)
            self.rows.append(0)
            self.slots.append(0)
            self.cells.append(0)

    @staticmethod
    def from_env(
        scope: Int = SCOPE_TREE, var label: String = String("")
    ) raises -> PhaseProfile:
        """A profile configured by `MOJOTREES_PHASE_PROFILE`. One `getenv` per
        fit on the threaded paths, one per tree on the rest."""
        return PhaseProfile(env_profile_mode(), scope, label^)

    def enabled(self) -> Bool:
        return self.mode != PROFILE_OFF

    def fenced(self) -> Bool:
        """True when a device call site should drain the queue between phases
        so each phase's clock closes over its own execution. Costs extra
        waits; see the module docstring. Always False on a host path, which
        has no queue, so a host grower need not consult it."""
        return self.mode == PROFILE_FENCED

    def clock(self) -> Int:
        """A start timestamp, or 0 when the profile is off.

        The only clock read in this module's hot path, and it is behind the
        mode test, which is what makes an off profile free of clock reads
        rather than merely cheap in them."""
        if self.mode == PROFILE_OFF:
            return 0
        return Int(perf_counter_ns())

    def begin_tree(mut self, root_rows: Int, dataset_rows: Int):
        """Open a tree. `root_rows` becomes the classification denominator for
        every charge until the next call."""
        if self.mode == PROFILE_OFF:
            return
        self.root_rows = root_rows
        self.dataset_rows = dataset_rows
        self.trees += 1

    def note_wall(mut self, started: Int):
        """Add the interval since `started` to the wall total, without
        charging it to any phase. `end_tree` is this under its own name; a
        caller that brackets a round rather than a tree uses this."""
        if self.mode == PROFILE_OFF or started <= 0:
            return
        var elapsed = Int(perf_counter_ns()) - started
        if elapsed > 0:
            self.wall_nanos += elapsed

    def end_tree(mut self, started: Int):
        """Close the tree `begin_tree` opened, charging its whole wall time so
        the report can name what no phase claimed."""
        self.note_wall(started)

    def note_node(mut self):
        """One node's histogram was built. Counted apart from `calls` because
        one node can be charged several times in one phase."""
        if self.mode == PROFILE_OFF:
            return
        self.nodes += 1

    def charge(
        mut self,
        phase: Int,
        node_rows: Int,
        started: Int,
        dispatches: Int = 0,
        syncs: Int = 0,
        slots_per_row: Int = 0,
        cells: Int = 0,
    ) raises:
        """Charge one operation to `phase`, filed under the class `node_rows`
        falls in.

        `started` came from `clock()`; a 0 (an off profile, or a caller that
        did not time this one) charges the counts and no time, which is
        deliberate -- a dispatch count is still true when nobody held a
        stopwatch.

        `slots_per_row` is the active feature count, so `node_rows *
        slots_per_row` is the (row, active feature) pairs the work covered.
        Passed as the per-row factor rather than the product because the
        grower has the feature count in hand and the multiplication is this
        module's business.

        Raises only on an unknown phase, which is a programming error at a
        call site and not a runtime condition.
        """
        if self.mode == PROFILE_OFF:
            return
        if phase < 0 or phase >= N_PROFILE_PHASES:
            raise Error("unknown profile phase ", phase)
        var cls = classify_node(node_rows, self.root_rows)
        var i = phase * N_NODE_CLASSES + cls
        self.calls[i] += 1
        self.dispatches[i] += dispatches
        self.syncs[i] += syncs
        if cells > 0:
            self.cells[i] += cells
        if node_rows > 0:
            self.rows[i] += node_rows
            if slots_per_row > 0:
                self.slots[i] += node_rows * slots_per_row
        if started > 0:
            var elapsed = Int(perf_counter_ns()) - started
            if elapsed > 0:
                self.nanos[i] += elapsed

    # -- reading ----------------------------------------------------------

    def _at(self, values: List[Int], phase: Int, cls: Int) -> Int:
        if phase < 0 or phase >= N_PROFILE_PHASES:
            return 0
        if cls < 0 or cls >= N_NODE_CLASSES:
            return 0
        return values[phase * N_NODE_CLASSES + cls]

    def nanos_of(self, phase: Int, cls: Int) -> Int:
        return self._at(self.nanos, phase, cls)

    def calls_of(self, phase: Int, cls: Int) -> Int:
        return self._at(self.calls, phase, cls)

    def dispatches_of(self, phase: Int, cls: Int) -> Int:
        return self._at(self.dispatches, phase, cls)

    def syncs_of(self, phase: Int, cls: Int) -> Int:
        return self._at(self.syncs, phase, cls)

    def rows_of(self, phase: Int, cls: Int) -> Int:
        return self._at(self.rows, phase, cls)

    def slots_of(self, phase: Int, cls: Int) -> Int:
        return self._at(self.slots, phase, cls)

    def cells_of(self, phase: Int, cls: Int) -> Int:
        return self._at(self.cells, phase, cls)

    def _phase_sum(self, values: List[Int], phase: Int) -> Int:
        var total = 0
        for c in range(N_NODE_CLASSES):
            total += self._at(values, phase, c)
        return total

    def _class_sum(self, values: List[Int], cls: Int) -> Int:
        var total = 0
        for p in range(N_PROFILE_PHASES):
            total += self._at(values, p, cls)
        return total

    def phase_nanos(self, phase: Int) -> Int:
        return self._phase_sum(self.nanos, phase)

    def phase_calls(self, phase: Int) -> Int:
        return self._phase_sum(self.calls, phase)

    def phase_dispatches(self, phase: Int) -> Int:
        return self._phase_sum(self.dispatches, phase)

    def phase_syncs(self, phase: Int) -> Int:
        return self._phase_sum(self.syncs, phase)

    def class_nanos(self, cls: Int) -> Int:
        return self._class_sum(self.nanos, cls)

    def class_calls(self, cls: Int) -> Int:
        return self._class_sum(self.calls, cls)

    def class_dispatches(self, cls: Int) -> Int:
        return self._class_sum(self.dispatches, cls)

    def class_syncs(self, cls: Int) -> Int:
        return self._class_sum(self.syncs, cls)

    def _total_of(self, values: List[Int]) -> Int:
        var total = 0
        for i in range(len(values)):
            total += values[i]
        return total

    def attributed_nanos(self) -> Int:
        return self._total_of(self.nanos)

    def total_dispatches(self) -> Int:
        return self._total_of(self.dispatches)

    def total_syncs(self) -> Int:
        return self._total_of(self.syncs)

    def total_calls(self) -> Int:
        return self._total_of(self.calls)

    def merge(mut self, other: PhaseProfile):
        """Fold another profile's buckets into this one.

        For a caller that profiles a sub-unit separately and wants one report.
        The mode, scope, and label stay this profile's, because they describe
        the report and not the arithmetic; a merge into an off profile is a
        no-op, so a disabled accumulator cannot pick up counts from an enabled
        part.
        """
        if self.mode == PROFILE_OFF:
            return
        for i in range(len(self.nanos)):
            self.nanos[i] += other.nanos[i]
            self.calls[i] += other.calls[i]
            self.dispatches[i] += other.dispatches[i]
            self.syncs[i] += other.syncs[i]
            self.rows[i] += other.rows[i]
            self.slots[i] += other.slots[i]
            self.cells[i] += other.cells[i]
        self.trees += other.trees
        self.nodes += other.nodes
        self.wall_nanos += other.wall_nanos
        if self.root_rows == 0:
            self.root_rows = other.root_rows
        if self.dataset_rows == 0:
            self.dataset_rows = other.dataset_rows

    def reset(mut self):
        for i in range(len(self.nanos)):
            self.nanos[i] = 0
            self.calls[i] = 0
            self.dispatches[i] = 0
            self.syncs[i] = 0
            self.rows[i] = 0
            self.slots[i] = 0
            self.cells[i] = 0
        self.trees = 0
        self.nodes = 0
        self.wall_nanos = 0

    # -- reporting --------------------------------------------------------

    def report(self) -> String:
        """The table, as lines.

        Readable and parseable, in that order of compromise: every line begins
        with the literal `phase_profile`, so one `grep` lifts the whole block
        out of a benchmark's output; every line's second word is its record
        kind; and every field after that is positional and separated by one
        space. The row block is emitted in full -- all fifty (phase, class)
        cells, zeros included -- so two reports of the same shape diff line
        for line instead of shifting when a class empties.

        Times are nanoseconds, integers, exactly as counted. The three derived
        columns are nanoseconds per thousand rows, per thousand row slots, and
        per thousand cells, which are the units the recorded M4 calibration
        (`bench/results/apple_m4_hybrid_costs_2026-08-15.md`) states its
        coefficients in. Each
        is 0.0 where nothing was counted, and a 0.0 there is an absence rather
        than a measured zero. `pct` is the cell's share of attributed time.

        The last line names what no phase claimed. A large unattributed share
        means the phases do not cover the round and the reader must not divide
        the ones that do into the whole.
        """
        var attributed = self.attributed_nanos()
        var out = String("phase_profile begin label=")
        out += self.label if self.label.byte_length() > 0 else String("-")
        out += " scope=" + profile_scope_name(self.scope)
        out += " mode=" + profile_mode_name(self.mode)
        out += " trees=" + String(self.trees)
        out += " nodes=" + String(self.nodes)
        out += " root_rows=" + String(self.root_rows)
        out += " dataset_rows=" + String(self.dataset_rows) + "\n"
        out += "phase_profile classes " + node_class_bounds() + "\n"
        out += (
            "phase_profile columns kind phase class calls dispatches syncs"
            " rows slots cells nanos ns_per_krow ns_per_kslot ns_per_kcell"
            " pct\n"
        )
        for p in range(N_PROFILE_PHASES):
            for c in range(N_NODE_CLASSES):
                var i = p * N_NODE_CLASSES + c
                out += "phase_profile row "
                out += profile_phase_name(p) + " " + node_class_name(c)
                out += " " + String(self.calls[i])
                out += " " + String(self.dispatches[i])
                out += " " + String(self.syncs[i])
                out += " " + String(self.rows[i])
                out += " " + String(self.slots[i])
                out += " " + String(self.cells[i])
                out += " " + String(self.nanos[i])
                out += " " + String(_per_thousand(self.nanos[i], self.rows[i]))
                out += " " + String(
                    _per_thousand(self.nanos[i], self.slots[i])
                )
                out += " " + String(
                    _per_thousand(self.nanos[i], self.cells[i])
                )
                out += " " + String(_pct(self.nanos[i], attributed)) + "\n"
        for p in range(N_PROFILE_PHASES):
            var pn = self.phase_nanos(p)
            var pr = self._phase_sum(self.rows, p)
            var ps = self._phase_sum(self.slots, p)
            var pc = self._phase_sum(self.cells, p)
            out += "phase_profile phase " + profile_phase_name(p) + " all"
            out += " " + String(self.phase_calls(p))
            out += " " + String(self.phase_dispatches(p))
            out += " " + String(self.phase_syncs(p))
            out += " " + String(pr)
            out += " " + String(ps)
            out += " " + String(pc)
            out += " " + String(pn)
            out += " " + String(_per_thousand(pn, pr))
            out += " " + String(_per_thousand(pn, ps))
            out += " " + String(_per_thousand(pn, pc))
            out += " " + String(_pct(pn, attributed)) + "\n"
        for c in range(N_NODE_CLASSES):
            var cn = self.class_nanos(c)
            var cr = self._class_sum(self.rows, c)
            var cs = self._class_sum(self.slots, c)
            var cc = self._class_sum(self.cells, c)
            out += "phase_profile class all " + node_class_name(c)
            out += " " + String(self.class_calls(c))
            out += " " + String(self.class_dispatches(c))
            out += " " + String(self.class_syncs(c))
            out += " " + String(cr)
            out += " " + String(cs)
            out += " " + String(cc)
            out += " " + String(cn)
            out += " " + String(_per_thousand(cn, cr))
            out += " " + String(_per_thousand(cn, cs))
            out += " " + String(_per_thousand(cn, cc))
            out += " " + String(_pct(cn, attributed)) + "\n"
        var unattributed = self.wall_nanos - attributed
        if unattributed < 0:
            # A fenced run can charge slightly more than the wall clock the
            # caller bracketed, because a phase's clock closes after a drain
            # the wall clock opened before. Printing a negative would be noise
            # dressed as a finding.
            unattributed = 0
        out += "phase_profile totals attributed_ns=" + String(attributed)
        out += " wall_ns=" + String(self.wall_nanos)
        out += " unattributed_ns=" + String(unattributed)
        out += " unattributed_pct=" + String(
            _pct(unattributed, self.wall_nanos)
        )
        out += " dispatches=" + String(self.total_dispatches())
        out += " syncs=" + String(self.total_syncs())
        out += " calls=" + String(self.total_calls()) + "\n"
        out += "phase_profile end\n"
        return out^

    def print_report(self):
        """`report()` to stdout, and nothing at all when the profile is off.

        The one place a call site has to remember nothing: an off profile
        prints no header, no empty table, and no blank line, so a run with the
        instrument off is byte identical on stdout too."""
        if self.mode == PROFILE_OFF:
            return
        print(self.report(), end="")
