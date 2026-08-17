"""Thread scaling: the same fit at one worker and at auto, against LightGBM
at one thread and at its own default, all in one process and interleaved.

The number this file exists to move is a **ratio of ratios**. The serial gap
against LightGBM is closed (1.05x behind, and 1.02x on the hot arm). The
parallel gap is not: at ten threads the same fit sits 1.38x behind. Dividing
those says the loss is in *scaling* and not in the kernel, and a scaling loss
cannot be read off one arm. It needs four cells -- ours and theirs, at one
thread and at many -- and it needs them close enough together in time that a
thermal regime cannot sit between two of them.

So every cell is an arm of one interleaved loop. Within a repeat the order is
fixed and every arm runs once, which is what makes a *paired* reduction legal:
repeat `r` of two arms saw the same machine, so `a[r] / b[r]` is a ratio the
window's drift largely cancels out of, and the count of repeats in which one
arm won is a statistic that survives a drifting window when a median does not.
Both are printed, because the serial lane's window climbed for its whole
length and its spread was drift rather than noise, and one of its repeats
recorded **579 seconds** from an operating system stall -- a sample that a
median absorbs and a spread does not.

## What LightGBM actually does at this shape, measured and not assumed

Established out of band, before this file was written, by training the
comparator for two rounds at `verbosity=1` on this exact dataset at both
thread counts. LightGBM printed, at **one** thread and at its **default**
thread count alike:

    [LightGBM] [Info] Auto-choosing col-wise multi-threading

So the private-buffer-plus-merge builder **does not run in this comparison at
all**. That builder is the row-wise one, and everything usually said about
LightGBM's histogram reduction is about it: per-block private histograms sized
`n_data_block * num_bin_aligned * 2`
(`src/io/train_share_states.cpp:211-222`), a block count that is
`min(num_threads, num_data / min_block_size)` and therefore never exceeds the
thread count (`include/LightGBM/train_share_states.h:63`), and a merge
partitioned by **bin** range so no two threads write one cell
(`src/io/train_share_states.cpp:120-190`). None of it is reached here.

What is reached is `Dataset::ConstructHistogramsInner`, and it has **no
private buffers and no reduction of any kind**. One `#pragma omp parallel for
schedule(static)` over the used dense feature groups
(`src/io/dataset.cpp:1385-1386`); each iteration memsets its own slice of the
destination at `hist_data + group_bin_boundaries_[group] * 2` and accumulates
straight into it (`src/io/dataset.cpp:1439-1450`). The destinations are
disjoint by construction, so there is nothing to merge. With
`enable_bundle=false` in the comparator and 25,500 total bins over 100
features, **every group is one feature: 100 dispatch units, each one feature
over all of the node's rows, at every node size.** The only node-size
adaptivity anywhere in that path is the `if (num_data >= 1024)` guard on the
ordered-gradient gather (`src/io/dataset.cpp:1368`). The column-wise versus
row-wise choice itself is made **once**, at the root, by timing one
construction of each (`src/io/dataset.cpp:694-727`), and is never revisited
per node.

## What the arms vary, and why these knobs

Blocks one and two change nothing in `src`: every arm there is an environment
setting the policy already read, which is what makes those sweeps a
measurement rather than a change with a benchmark attached.

**Block three is not like that and the difference is worth naming
rather than discovering.** `wA_minops`, `wA_mintask` and `wA_floor0` are still
pure environment. `wA_rowsmall` is new code -- a per-node bin-layout rule --
behind its own off-by-default switch, so `wA_base` in that block is still byte
for byte the program blocks one and two measured. It is bit-neutral by
construction and the determinism section below checks it against the same
digest every other arm produces. The winner becomes a default afterwards, in a
separate commit, with this transcript as its justification.

Ours differs from the comparator on exactly one axis: we cut the node's rows
into private-histogram blocks and fold them, and LightGBM does not cut rows at
all. So the arms walk that axis to nothing and then walk the feature axis past
it.

- `MOJOTREES_CPU_ROW_BLOCKS=1` (`noblk`). The documented off switch for row
  blocking, which "reproduces the pre-blocking accumulation exactly": one
  block, no private histograms, no fold, the kernel writing each feature's
  slice of the output directly. The interleave width is still derived, which
  at one block and 100 features is 4, so this is 25 dispatch units.

- `MOJOTREES_CPU_ROW_BLOCKS=1` **and** `MOJOTREES_CPU_FEATURE_GROUP=1`
  (`lgbmlike`). The same, at width 1: one unit per feature, 100 units, no
  private histogram beyond the output slice itself and no reduction. **This is
  LightGBM's column-wise partition, feature for feature.** It is the arm that
  says whether the shape is the gap or whether the shape is already fine and
  something else is.

  It is not free, and the price is worth stating because it predicts the
  result. A group re-walks its rows, so the gradient stream is read
  `ceil(n_active / width)` times per node: 13 passes at the shipped width 8,
  **100 at width 1**. LightGBM pays those 100 passes. If gradient traffic were
  what binds, we would already be ahead on it by 7.7x and we are not, so the
  prediction registered here before the run is that this arm does **not** win
  on gradient traffic and that whatever it moves is the row-block overhead.

- `MOJOTREES_CPU_FEATURE_GROUP=16` (`g16`). The width is clamped to 8 by an
  assumed 64 KB L1 that `apple_cpu_policy` documents as a portable floor
  rather than this machine's cache, and says in as many words is unmeasured in
  both directions and settled only by an interleaved wall-time A/B. This is
  that A/B, in the other direction from `lgbmlike`: 7 passes of the gradient
  stream instead of 13, and 7 units instead of 13 per block.

The three above are the *partition* arms and they are one block of the run.
The second block is the **fan-out**, which is the half of scheduling that no
partition arm can reach, and it exists because this machine is four
performance cores and six efficiency ones and every rule in `apple_cpu_policy`
prices a core as if they were equal. `dispatch_rounds` says so about itself:
"Real cores here are not equal, so this understates the cost of a partial
round rather than overstating it."

- `MOJOTREES_CPU_TASKS_PER_CORE` at 1 and at 16 against the shipped 4
  (`tpc1`, `tpc16`). The oversubscription factor, and the only lever that
  decides whether a performance core has somewhere to go while an efficiency
  core is still finishing its share of a statically split range. The shipped
  4 is documented in `apple_cpu_policy` as "a starting point, not a measured
  optimum", which is a request for exactly this arm.
- `MOJOTREES_CPU_CORE_POOL=performance` (`pool`), which plans against four
  cores instead of ten. The blunter form of the same question: if the
  accumulation is memory-bound before ten cores are busy, the six efficiency
  cores are contributing contention rather than throughput.

  **Read that arm narrowly, because the knob does less than its name says.**
  `apple_cpu_policy` uses "no affinity, no QoS class, no thread pinning", so
  the pool setting cannot keep work off an efficiency core. All it changes is
  `dispatch_cores`, which is the number the fan-out rule plans against: the
  task count drops from `4 * 10` to `4 * 4`, and the runtime still spreads
  those sixteen tasks over whatever cores it likes. So this arm measures
  **fewer, larger tasks**, not four-cores-only, and a win here would be a
  statement about fan-out cost rather than about efficiency cores. Stated here
  because the arm's name invites the other reading and the other reading is
  the one that would be wrong.

Both are pure schedule. They change how many tasks the same units are handed
to and which core count the decision is made against, and neither can move a
summation order, so **every one of them must digest identically to `base`**
and the run says so arm by arm.

The knobs are schedule-only in one respect and not the other, and the
difference is the whole of this file's determinism argument:

- **The width is bit-neutral.** It changes how many features share one walk of
  the rows, never the order in which one feature's bins are summed. `w1_g16`
  must therefore digest identically to `w1_base`, and the run checks it.
- **The block count is not.** It changes which rows are summed together before
  their partials are folded, so `noblk` is a different Float64 sequence from
  `base`. What it must still be is **independent of the worker count**: the
  fold walks blocks in ascending order inside a task that owns a whole slot,
  so no partition of the units can reach it. That is the property under test,
  and it is tested by equality on a `bitcast` digest rather than by a
  tolerance, at one worker and at auto, for every knob setting.

## What is NOT measured here, stated so it is not inferred

This is whole-fit wall time. It cannot attribute a difference to the
histogram, and the in-run phase profiler (`MOJOTREES_PHASE_PROFILE`) is not
enabled, because an instrument inside a timed region is a different
experiment. The attribution this sweep leans on was measured separately and is
recorded in `bench/results/cpu_round1_2026-08-16/RESULTS.md`: histogram
accumulation is 83.9 percent of the serial round and scales at 2.13x, against
2.69x at the root and 1.20x at small nodes.

## The LightGBM arm, and the one thing it is asked that it is not usually asked

One `InterleavedArm` is constructed, not two: the dataset is 640 MB of
Float64 and a second copy for a second thread count is memory this box does
not need to spend. The thread count is moved by writing `num_threads` into the
arm's own parameter dict between arms, and it is then **read back out of that
dict and printed on every repeat**, so a transcript cannot disagree with the
configuration it was taken under. A printed 0 is LightGBM choosing for itself,
which is its documented default of one thread per physical core and is the
statement the `wA` cell is making.

LightGBM's own Info log does **not** reach this transcript: its messages are
emitted through a C log callback into Python's `print`, and under the Mojo
interop they do not appear in the process's captured output. That is why the
column-wise finding above was taken out of band, in Python, rather than from
inside this file. It is stated as a recorded fact in the header instead of
being probed here, because a probe whose output silently vanishes is worse
than no probe.

One consequence of the auto-choice is worth naming, since it lands inside
LightGBM's timed number and not inside ours. `GetShareStates` is called for
every new Booster, so LightGBM re-runs its column-wise-versus-row-wise test on
every repeat: measured at this shape, 0.29 seconds at one thread and 0.064 at
its default. That is a real cost of the comparator as configured, it is
already inside every LightGBM figure this repository has recorded under
`stock+det`, and it is left in rather than subtracted because `force_col_wise`
is not what stock does.

Usage, from the repository root under the bench environment:

  MOJOTREES_BENCH_DIR=bench mojo run -I src bench/bench_thread_scaling.mojo \
      [rows] [features] [repeats] [arms] [seed]

`arms` is `one` (the partition block, and the default), `two` (the fan-out
block), `all`, or a comma-separated list of names. The two registered blocks
each carry their own `base` and `lgbm` cells at both worker counts, so each is
a self-contained experiment: a number from one block is never compared with a
number from the other, which is the rule a drifting window forces.

The default is the literal block-one list rather than `all`, so that adding an
arm to this file cannot silently change what a bare invocation measured.
"""

from std.collections import Optional
from std.memory import bitcast
from std.os import getenv, setenv
from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.apple_cpu_policy import (
    CpuProfile,
    RowBlockAmortize,
    feature_group_count,
    plan_row_block_count_at,
    plan_feature_group,
)
from mojotrees.binning import BinnedMatrix, fit_bins
from mojotrees.boosting import (
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


@fieldwise_init
struct Arm(Copyable, Movable):
    """One cell of the four-cell design, or one knob arm inside a cell.

    `workers` and the two knobs are exported on **every** mojotrees arm, never
    left alone, because an unset variable that a previous arm set is the
    classic way an A/B runs one arm under the other's label. An empty string
    is exported as empty, which every reader in `apple_cpu_policy` treats as
    unset.
    """

    var name: String
    var is_lgbm: Bool
    var workers: String
    """`MOJOTREES_NUM_WORKERS`; empty is auto. On a LightGBM arm this is the
    pinned `num_threads`, where empty means LightGBM's own default."""
    var blocks: String
    """`MOJOTREES_CPU_ROW_BLOCKS`; empty derives, 1 turns row blocking off."""
    var group: String
    """`MOJOTREES_CPU_FEATURE_GROUP`; empty is the derived width."""
    var tasks_per_core: String
    """`MOJOTREES_CPU_TASKS_PER_CORE`; empty is the shipped 4."""
    var core_pool: String
    """`MOJOTREES_CPU_CORE_POOL`; empty is `all`, `performance` plans against
    the performance cores only."""
    var min_ops: String
    """`MOJOTREES_PARALLEL_MIN_OPS`; empty is the shipped 65,536. Raising it
    is the per-node serial floor: every dispatch whose work estimate is below
    it stays on one core, and because every estimate on the per-node path is
    proportional to the node's rows, raising it is exactly "nodes below this
    size build serially"."""
    var min_task_ops: String
    """`MOJOTREES_PARALLEL_MIN_TASK_OPS`; empty is the shipped 65,536. With
    the core floor off this is the per-node *worker* floor: the task count
    becomes `total_ops / min_task_ops` clamped to `[2, max_auto]`, so a small
    node runs on few workers instead of all ten."""
    var task_floor: String
    """`MOJOTREES_CPU_TASK_FLOOR`; empty is on, `0` reverts the fan-out rule
    to the grain alone. `min_task_ops` only bites with this off, which is why
    they are two fields and are set together."""
    var layout_by_node: String
    """`MOJOTREES_CPU_LAYOUT_BY_NODE`; empty is off, `1` reads a node too
    small to block out of the row-major record array and every other node out
    of the feature-major column array."""

    def lgbm_threads(self) -> Int:
        if self.workers.byte_length() == 0:
            return 0
        try:
            return Int(self.workers)
        except:
            return 0


comptime BLOCK_ONE = (
    "w1_base,w1_noblk,w1_lgbmlike,w1_g16,w1_lgbm,"
    "wA_base,wA_noblk,wA_lgbmlike,wA_g16,wA_lgbm"
)
"""The partition block, and the default arm list.

The default is this list rather than `all` on purpose: `all` would grow
whenever an arm is added, so a bare invocation would stop meaning what an
earlier bare invocation meant, and two transcripts labelled the same way
would not be the same experiment."""

comptime BLOCK_TWO = (
    "w1_base,w1_tpc1,w1_lgbm,"
    "wA_base,wA_tpc1,wA_tpc16,wA_pool,wA_lgbm"
)
"""The fan-out block. Carries its own `base` and `lgbm` cells at both worker
counts, so it is a self-contained experiment and never has to be compared
against a number from the partition block."""


comptime BLOCK_THREE = (
    "w1_base,w1_rowsmall,w1_lgbm,"
    "wA_base,wA_minops,wA_mintask,wA_floor0,wA_rowsmall,wA_lgbm"
)
"""The small-node block. Carries its own `base` and `lgbm` cells at both
worker counts for the same reason block two does, and carries a one-worker
cell for the layout arm because that arm's claim is about memory traffic
rather than about scheduling and so has to be checkable without a fan-out.

Nine arms. The three scheduling arms have no one-worker cell because at one
worker they ARE `w1_base`: `plan_tasks` returns 1 before it reads either
grain, so an arm that only moves a grain cannot move a one-worker number, and
an arm whose name promises a cell it cannot fill is worse than a missing
cell."""


def _all_arms() -> List[Arm]:
    """Every arm this file defines, in the order they run inside a repeat when
    they are selected.

    The one-worker cell runs first and whole, then the auto cell, then each
    engine's opposite number. The order is fixed rather than randomized: a
    fixed order makes repeat `r` of arm `i` and repeat `r` of arm `j` a
    matched pair separated by a known and constant amount of other work,
    which is what the paired reduction below is entitled to assume.

    Selection preserves this order rather than the order the caller names, so
    two runs of the same set are the same experiment however the argument was
    typed.
    """
    var e = String("")
    var one = String("1")
    var arms = List[Arm]()

    # A local so the ten-field constructor reads as a decision rather than a
    # row of commas. Every field is exported on every mojotrees arm, so a
    # knob left at `e` is "explicitly the default" and not "inherited from
    # whichever arm ran last", which is the failure this file is built around.
    def arm(
        name: String,
        workers: String,
        blocks: String = String(""),
        group: String = String(""),
        tasks_per_core: String = String(""),
        core_pool: String = String(""),
        min_ops: String = String(""),
        min_task_ops: String = String(""),
        task_floor: String = String(""),
        layout_by_node: String = String(""),
    ) -> Arm:
        return Arm(
            name, False, workers, blocks, group, tasks_per_core, core_pool,
            min_ops, min_task_ops, task_floor, layout_by_node,
        )

    def lgbm_arm(name: String, workers: String) -> Arm:
        var b = String("")
        return Arm(name, True, workers, b, b, b, b, b, b, b, b)

    # Block one: the partition shape. Row blocks against feature blocks.
    arms.append(arm(String("w1_base"), one))
    arms.append(arm(String("w1_noblk"), one, blocks=one))
    arms.append(arm(String("w1_lgbmlike"), one, blocks=one, group=one))
    arms.append(arm(String("w1_g16"), one, group=String("16")))
    # Block two's one-worker knob sits here, before the LightGBM arm, so that
    # selecting either block leaves the one-worker cell contiguous and the
    # LightGBM arm at its end. The ORDER of this list is the order arms run
    # in, and it is what a paired reduction is entitled to assume is constant.
    arms.append(arm(String("w1_tpc1"), one, tasks_per_core=one))
    # Block three's one-worker cells. `min_ops` and `min_task_ops` are
    # scheduling floors and cannot bite at one worker, so the only block-three
    # arm that has anything to say here is the layout one -- which is not a
    # schedule at all, and whose whole claim is about memory traffic that one
    # core pays as surely as ten. Its one-worker cell is what separates "the
    # layout helps" from "the layout helps the fan-out".
    arms.append(arm(String("w1_rowsmall"), one, layout_by_node=one))
    arms.append(lgbm_arm(String("w1_lgbm"), one))
    arms.append(arm(String("wA_base"), e))
    arms.append(arm(String("wA_noblk"), e, blocks=one))
    arms.append(arm(String("wA_lgbmlike"), e, blocks=one, group=one))
    arms.append(arm(String("wA_g16"), e, group=String("16")))
    # Block two: the fan-out, which is the half of scheduling the partition
    # arms cannot reach. Both of these are pure schedule -- they change how
    # many tasks the same units are handed to and which cores are counted when
    # deciding, and neither can move a summation order -- so every one of them
    # must digest identically to `base`, and the determinism section checks it.
    #
    # `tpc` sweeps `MOJOTREES_CPU_TASKS_PER_CORE`, whose shipped 4 is
    # documented in `apple_cpu_policy` as "a starting point, not a measured
    # optimum". It is the oversubscription factor, and on a machine with four
    # performance cores and six efficiency ones it is the only lever that
    # decides whether a performance core has somewhere to go while an
    # efficiency core is still finishing its share of a statically split
    # range. At 1 it is one task per core and a straggler sets the pace of
    # every barrier; at 16 the fan-out itself may cost more than the imbalance
    # it hides.
    #
    # `pool` is the other question the same asymmetry raises and the blunter
    # one: `MOJOTREES_CPU_CORE_POOL=performance` plans against four cores
    # instead of ten. If the accumulation is memory-bound before ten cores are
    # busy, the six efficiency cores are contributing contention rather than
    # throughput, and this arm is what says so.
    arms.append(arm(String("wA_tpc1"), e, tasks_per_core=one))
    arms.append(arm(String("wA_tpc16"), e, tasks_per_core=String("16")))
    arms.append(arm(String("wA_pool"), e, core_pool=String("performance")))
    # Block three: the small node going parallel. The measured target is a
    # ratio of ratios -- a (row, feature) accumulate costs 6.05x more at a
    # small node than at the root at one worker and 11.65x more at auto, so
    # small nodes lose 1.93x of their relative efficiency the moment the fit
    # is parallel and the root loses none. These four arms are the three ways
    # of attacking that which do not need the kernel rewritten, and they are
    # separate arms because they are separate claims.
    #
    # `minops` is the per-node **serial** floor, and it is the crudest form of
    # the question. `MOJOTREES_PARALLEL_MIN_OPS` is the whole-loop crossover:
    # a dispatch whose work estimate falls below it does not fan out at all.
    # Every estimate on the per-node path is proportional to the node's rows,
    # so raising the crossover to 1,000,000 is exactly "a node below about
    # 10,000 rows at 100 features builds serially" -- the small and tiny
    # classes, and nothing above them. **The registered prediction is that
    # this LOSES**: the small class was measured at 2.385 ns per slot serially
    # against 1.948 at auto, so it does get 1.22x out of ten cores, and
    # throwing that away to save the fan-out has to find 22 percent somewhere.
    # It is run because "small nodes should go serial" is the first thing
    # anybody reaches for and a measured refusal is worth more than an
    # argument.
    #
    # `mintask` is the per-node **worker** floor, which is the same question
    # asked without the cliff. With `MOJOTREES_CPU_TASK_FLOOR=0` the core
    # floor stops forcing one task per core and the task count falls back to
    # the grain, `total_ops / MOJOTREES_PARALLEL_MIN_TASK_OPS`, clamped below
    # at 2. At 262,144 a 6,000-row node runs on 2 workers, a 34,000-row node
    # on about 13, and the root on the full 40. `parallel.env_core_floor`'s
    # own docstring asks for this A/B by name -- "the histogram at small nodes
    # is the one to watch ... run the A/B, and run it early" -- and this is
    # it.
    #
    # `floor0` is `mintask` with the grain left alone, so it isolates what
    # turning the core floor off costs by itself. Without it a win or a loss
    # on `mintask` cannot be attributed to the grain rather than to the floor.
    #
    # `rowsmall` is not a schedule and it is the arm with a quantitative
    # prediction behind it. At 799,110 rows and a 128-byte line, a 6,102-row
    # node reads each feature-major column at a row density of 0.76 percent,
    # so it fetches about 78 MB of cache lines to use 610 KB of bin ids; the
    # same node's row-major records are 100 contiguous bytes per row and cost
    # about 780 KB. The per-fit layout question was already measured and
    # feature-major won it by 1.15x, but that answer is the root's and the
    # large nodes' -- there the column is a unit-stride stream and the record
    # array is a strided walk with nothing gained.
    # `MOJOTREES_CPU_LAYOUT_BY_NODE` asks it per node instead of per fit,
    # keyed on the node's own block
    # count, which is the grower's existing proxy for "big enough to amortize
    # a bin sweep".
    arms.append(arm(String("wA_minops"), e, min_ops=String("1000000")))
    arms.append(
        arm(
            String("wA_mintask"),
            e,
            min_task_ops=String("262144"),
            task_floor=String("0"),
        )
    )
    arms.append(arm(String("wA_floor0"), e, task_floor=String("0")))
    arms.append(arm(String("wA_rowsmall"), e, layout_by_node=one))
    arms.append(lgbm_arm(String("wA_lgbm"), e))
    return arms^


def _select_arms(all_arms: List[Arm], words: String) raises -> List[Arm]:
    """The named arms, in the file's order rather than the caller's.

    `all` is accepted and means every arm defined, `one` and `two` name the
    two registered blocks. An unknown name raises instead of being skipped:
    an arm list with a typo in it silently running nine arms under a label
    that says ten is the kind of thing this repository has thrown results
    away for.
    """
    var want_words = words
    if want_words.byte_length() == 0:
        want_words = String(BLOCK_ONE)
    elif want_words == "one":
        want_words = String(BLOCK_ONE)
    elif want_words == "two":
        want_words = String(BLOCK_TWO)
    elif want_words == "three":
        want_words = String(BLOCK_THREE)
    if want_words == "all":
        return all_arms.copy()

    var wanted = List[String]()
    for part in want_words.split(","):
        var w = String(part)
        if w.byte_length() > 0:
            wanted.append(w)
    for i in range(len(wanted)):
        var found = False
        for j in range(len(all_arms)):
            if all_arms[j].name == wanted[i]:
                found = True
        if not found:
            raise Error(String('unknown arm "', wanted[i], '"'))

    # File order, not caller order.
    var out = List[Arm]()
    for j in range(len(all_arms)):
        for i in range(len(wanted)):
            if all_arms[j].name == wanted[i]:
                out.append(all_arms[j].copy())
                break
    if len(out) == 0:
        raise Error("no arms selected")
    return out^


def _model_digest(booster: Booster, data: BinnedMatrix) -> UInt64:
    """A bitwise digest of this model's predictions on a fixed row stride.

    Bitwise and not approximate. The question is whether two accumulations
    produced the *same* Float64, and a tolerance answers a different question.
    Identical to `bench/bench_serial_kernel.mojo`'s, deliberately, so a digest
    printed by either file is comparable with one printed by the other.
    """
    var step = data.n_rows // 4096
    if step < 1:
        step = 1
    var h = UInt64(0xCBF29CE484222325)
    var r = 0
    while r < data.n_rows:
        var bits = bitcast[DType.uint64, 1](booster.predict_row(data, r))
        h = _splitmix64(h ^ bits)
        r += step
    return h


def _median(values: List[Float64]) -> Float64:
    var s = values.copy()
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    var n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _plateau(values: List[Float64], drop: Int) -> List[Float64]:
    """The repeats past `drop`, which is where both engines plateau."""
    var out = List[Float64]()
    for i in range(len(values)):
        if i >= drop:
            out.append(values[i])
    if len(out) == 0:
        return values.copy()
    return out^


def _spread(values: List[Float64]) -> Float64:
    var lo = values[0]
    var hi = values[0]
    for i in range(len(values)):
        if values[i] < lo:
            lo = values[i]
        if values[i] > hi:
            hi = values[i]
    return hi - lo


def _lo(values: List[Float64]) -> Float64:
    var lo = values[0]
    for i in range(len(values)):
        if values[i] < lo:
            lo = values[i]
    return lo


def _hi(values: List[Float64]) -> Float64:
    var hi = values[0]
    for i in range(len(values)):
        if values[i] > hi:
            hi = values[i]
    return hi


def _index_of(arms: List[Arm], name: String) -> Int:
    for i in range(len(arms)):
        if arms[i].name == name:
            return i
    return -1


def _paired(
    label: String,
    a_name: String,
    b_name: String,
    a: List[Float64],
    b: List[Float64],
    drop: Int,
) -> None:
    """Per-repeat ratio `a[r] / b[r]`, its median, and the win count.

    This is the reduction that survives a drifting window. Repeat `r` of the
    two arms ran inside the same repeat of the same loop, so whatever the
    machine was doing at the time is in both numerators and denominators.
    A median over the ratios is therefore a different and more defensible
    statistic than a ratio of two medians, and the count of repeats in which
    `a` was the slower is a statistic that needs no distribution at all.

    Both are printed alongside the ratio of medians, because when the three
    disagree that disagreement is the finding.
    """
    var ratios = List[Float64]()
    var wins_a = 0
    var n = len(a) if len(a) < len(b) else len(b)
    for r in range(n):
        if r < drop:
            continue
        if b[r] > 0.0:
            ratios.append(a[r] / b[r])
        # Counted over the plateau only, exactly as the ratios are. Counting
        # the head as well would let the repeats where both engines are still
        # climbing vote on a question about the levelled regime.
        if a[r] < b[r]:
            wins_a += 1
    if len(ratios) == 0:
        return
    var line = String(label, " ", a_name, "/", b_name, " paired_ratios:")
    for i in range(len(ratios)):
        line += String(" ", ratios[i])
    print(line)
    var counted = len(a) - drop
    if counted < 0:
        counted = 0
    print(
        label,
        String(a_name, "/", b_name),
        "paired_ratio_median:",
        _median(ratios),
        "paired_ratio_min:",
        _lo(ratios),
        "paired_ratio_max:",
        _hi(ratios),
        "faster_repeats_a_of_counted:",
        wins_a,
        "of",
        counted,
    )


def _lgbm_arm(
    n_rows: Int, n_features: Int, seed: Int, params: BoosterParams
) raises -> PythonObject:
    var bench_dir = getenv("MOJOTREES_BENCH_DIR")
    if bench_dir.byte_length() == 0:
        bench_dir = String("bench")
    Python.add_to_path(bench_dir)
    var module: PythonObject
    try:
        module = Python.import_module("bench_lightgbm")
    except e:
        raise Error(
            String(
                "the lightgbm arm needs bench/bench_lightgbm.py and the bench"
                " environment's LightGBM; run under the bench environment from"
                " the repository root. Underlying error: ",
                String(e),
            )
        )
    # Constructed at LightGBM's own default thread count. The thread count is
    # moved per arm below by writing the parameter, so the value passed here
    # only decides which of the two cells pays the Dataset construction.
    return module.InterleavedArm(
        n_rows,
        n_features,
        String("reg"),
        0,
        seed,
        rounds=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.tree.num_leaves,
        min_data_in_leaf=params.tree.min_data_in_leaf,
        lambda_l2=params.tree.lambda_reg,
        lambda_l1=params.tree.lambda_l1,
        min_child_hess=params.tree.min_child_hess,
        max_depth=params.tree.max_depth,
        max_bin=255,
    )


def _set_lgbm_threads(state: PythonObject, threads: Int) raises:
    """Pin `num_threads`, or hand it back to LightGBM at 0.

    LightGBM's own default for `num_threads` is 0, meaning one thread per
    physical core, so writing 0 is the same statement as leaving it out and
    both arms can be expressed by writing the key.
    """
    state.params["num_threads"] = threads


def _plan_line(
    label: String,
    blocks_req: String,
    group_req: String,
    rows: Int,
    n_bins: Int,
    n_active: Int,
) raises:
    """The dispatch shape one arm's knobs plan at one node size.

    Printed rather than reasoned about, because the arm names are statements
    about block counts and widths and a reader has no other way to check that
    the knobs reached the planner. Recomputed here through the same policy
    functions the fit calls, at the shipped amortization ratio, so a line here
    and the dispatch the kernel makes cannot disagree.
    """
    var requested = 0
    if blocks_req.byte_length() > 0:
        requested = Int(blocks_req)
    var ratio = RowBlockAmortize(8, 1, True)
    var blocks = plan_row_block_count_at(
        ratio, requested, rows, n_bins, n_active, True
    )
    var width: Int
    if group_req.byte_length() > 0:
        width = Int(group_req)
    else:
        width = plan_feature_group(
            CpuProfile.detect(), n_bins, n_active, blocks, True
        )
    var groups = feature_group_count(n_active, width)
    print(
        "plan",
        label,
        "node_rows:",
        rows,
        "row_blocks:",
        blocks,
        "group_width:",
        width,
        "groups:",
        groups,
        "units:",
        blocks * groups,
    )


def main() raises:
    var args = argv()
    var n_rows = 799110
    var n_features = 100
    var repeats = 12
    var seed = 0
    var arm_words = String(BLOCK_ONE)
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        repeats = Int(String(args[3]))
        if repeats < 1:
            raise Error("repeats must be at least 1")
    if len(args) > 4:
        arm_words = String(args[4])
    if len(args) > 5:
        seed = Int(String(args[5]))
    if n_features < 4:
        raise Error("the generator needs at least 4 features")

    var arms = _select_arms(_all_arms(), arm_words)

    # The same generator sequence as bench_serial_kernel.mojo,
    # bench_train_gpu.mojo and bench_lightgbm.py, counter for counter. The
    # LightGBM arm regenerates the dataset on the Python side from the same
    # counters and never receives ours, so a generator that drifted by one
    # multiply would have the two engines fitting different problems.
    var features = List[Float64](capacity=n_rows * n_features)
    var seed_offset = UInt64(seed) * 0x9E3779B97F4A7C15
    for k in range(n_rows * n_features):
        features.append(_uniform(seed_offset + UInt64(k)))
    var noise_base = seed_offset + UInt64(n_rows * n_features)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        var signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
        var u = _uniform(noise_base + UInt64(r))
        target.append(signal + 0.1 * (u - 0.5))

    print(
        "mojotrees cpu thread-scaling bench:",
        n_rows,
        "rows x",
        n_features,
        "features, reg seed",
        seed,
    )
    print("machine:", CpuProfile.detect().describe())
    var t0 = perf_counter_ns()
    var mapper = fit_bins(features, n_rows, n_features, 255)
    var data = mapper.transform(features, n_rows)
    var t1 = perf_counter_ns()
    print("binning_s:", Float64(t1 - t0) / 1e9)
    # Whether the row-major view exists at all, printed rather than assumed.
    # `MOJOTREES_CPU_LAYOUT_BY_NODE` degrades **silently** to feature-major on
    # a matrix with no view, so an arm that names the layout and runs on a
    # matrix that has none is a null wearing a result's label. This line is
    # what separates "the per-node layout rule bought nothing" from "the
    # per-node layout rule never ran".
    print(
        "row_major_view:",
        String("yes") if data.has_row_major() else String("no"),
        "row_major_bytes:",
        data.row_major_bytes(),
    )

    var arm_list = String("")
    for a in range(len(arms)):
        if a > 0:
            arm_list += ","
        arm_list += arms[a].name
    print("arms:", arm_list)
    print("repeats:", repeats)
    if repeats < 3:
        print(
            "warning: fewer than 3 repeats cannot separate a real difference"
            " from machine drift"
        )

    # The shapes each knob plans, at the node sizes a 31-leaf fit on this many
    # rows actually visits. The root, a half, an eighth, a sixty-fourth.
    for a in range(len(arms)):
        if arms[a].is_lgbm:
            continue
        if arms[a].workers != "1":
            continue
        var b = arms[a].blocks
        var g = arms[a].group
        _plan_line(arms[a].name, b, g, n_rows, data.n_bins, n_features)
        _plan_line(arms[a].name, b, g, n_rows // 2, data.n_bins, n_features)
        _plan_line(arms[a].name, b, g, n_rows // 8, data.n_bins, n_features)
        _plan_line(arms[a].name, b, g, n_rows // 64, data.n_bins, n_features)

    var params = BoosterParams.default()
    var lgbm = Optional[PythonObject]()
    var want_lgbm = False
    for a in range(len(arms)):
        if arms[a].is_lgbm:
            want_lgbm = True
    if want_lgbm:
        var state = _lgbm_arm(n_rows, n_features, seed, params)
        print("lightgbm_binning_s:", Float64(py=state.binning_s))
        print("lightgbm_params:", String(py=state.summary()))
        print(
            "lightgbm_multithreading: col-wise at both thread counts,"
            " measured out of band at verbosity 1 on this shape; see the"
            " module docstring"
        )
        lgbm = Optional[PythonObject](state)

    var samples = List[List[Float64]]()
    var digests = List[UInt64]()
    for _ in range(len(arms)):
        samples.append(List[Float64]())
        digests.append(UInt64(0))

    for rep in range(repeats):
        for a in range(len(arms)):
            var seconds: Float64
            var lgbm_threads = -1
            if arms[a].is_lgbm:
                _set_lgbm_threads(lgbm.value(), arms[a].lgbm_threads())
                # Read back rather than assumed, on every repeat, because an
                # arm named after a thread count that it did not run under is
                # the failure this whole file is built to avoid.
                lgbm_threads = Int(py=lgbm.value().resolved_threads())
                var t = perf_counter_ns()
                _ = Int(py=lgbm.value().train())
                seconds = Float64(perf_counter_ns() - t) / 1e9
            else:
                # All five exported on every mojotrees arm. None inherited.
                _ = setenv("MOJOTREES_NUM_WORKERS", arms[a].workers)
                _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", arms[a].blocks)
                _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", arms[a].group)
                _ = setenv(
                    "MOJOTREES_CPU_TASKS_PER_CORE", arms[a].tasks_per_core
                )
                _ = setenv("MOJOTREES_CPU_CORE_POOL", arms[a].core_pool)
                _ = setenv(
                    "MOJOTREES_PARALLEL_MIN_OPS", arms[a].min_ops
                )
                _ = setenv(
                    "MOJOTREES_PARALLEL_MIN_TASK_OPS", arms[a].min_task_ops
                )
                _ = setenv("MOJOTREES_CPU_TASK_FLOOR", arms[a].task_floor)
                _ = setenv(
                    "MOJOTREES_CPU_LAYOUT_BY_NODE", arms[a].layout_by_node
                )
                var t = perf_counter_ns()
                var model = train(data, target, SQUARED_ERROR, params)
                seconds = Float64(perf_counter_ns() - t) / 1e9
                if rep == 0:
                    digests[a] = _model_digest(model, data)
            samples[a].append(seconds)
            if lgbm_threads >= 0:
                print(
                    "run",
                    rep + 1,
                    arms[a].name,
                    "train_s:",
                    seconds,
                    "resolved_num_threads:",
                    lgbm_threads,
                )
            else:
                print("run", rep + 1, arms[a].name, "train_s:", seconds)

    # Everything from here is after the clock has stopped for good.
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_CPU_ROW_BLOCKS", "")
    _ = setenv("MOJOTREES_CPU_FEATURE_GROUP", "")
    _ = setenv("MOJOTREES_CPU_TASKS_PER_CORE", "")
    _ = setenv("MOJOTREES_CPU_CORE_POOL", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_TASK_OPS", "")
    _ = setenv("MOJOTREES_CPU_TASK_FLOOR", "")
    _ = setenv("MOJOTREES_CPU_LAYOUT_BY_NODE", "")

    # The determinism contract, checked in the run that measured the arms
    # rather than in a separate one. Two claims, and they are different:
    #
    # 1. Worker invariance. For every knob setting, the one-worker digest and
    #    the auto digest must be EQUAL. A reduction whose fold order depended
    #    on which task finished first would fail here, and that is a defect
    #    rather than a tolerance.
    # 2. Schedule neutrality. Every arm that changes only *how the same units
    #    are handed out* -- the interleave width, the oversubscription factor,
    #    the core pool -- must digest identically to `base`, because none of
    #    them can change the order in which one feature's bins are summed. The
    #    two blocking arms are exempt and only exempt: `noblk` and `lgbmlike`
    #    change which rows are summed together before the fold, which is a
    #    different Float64 sequence by construction.
    print("--- determinism ---")
    var pairs = List[String]()
    pairs.append(String("base"))
    pairs.append(String("noblk"))
    pairs.append(String("lgbmlike"))
    pairs.append(String("g16"))
    pairs.append(String("tpc1"))
    pairs.append(String("rowsmall"))
    for p in range(len(pairs)):
        var i = _index_of(arms, String("w1_", pairs[p]))
        var j = _index_of(arms, String("wA_", pairs[p]))
        if i < 0 or j < 0:
            continue
        print(
            "worker_invariance",
            pairs[p],
            "w1_digest:",
            digests[i],
            "auto_digest:",
            digests[j],
            "verdict:",
            String("identical") if digests[i] == digests[j] else String(
                "MISMATCH -- the fold order moved with the worker count"
            ),
        )
    # Schedule neutrality, against whichever `base` cell this arm set contains.
    var ref_i = _index_of(arms, String("w1_base"))
    if ref_i < 0:
        ref_i = _index_of(arms, String("wA_base"))
    if ref_i >= 0:
        for a in range(len(arms)):
            if arms[a].is_lgbm:
                continue
            if arms[a].blocks.byte_length() > 0:
                continue  # blocking arms are exempt, and only they are
            print(
                "schedule_neutrality",
                arms[a].name,
                "vs",
                arms[ref_i].name,
                "verdict:",
                String("identical") if digests[a] == digests[ref_i] else String(
                    "MISMATCH -- a schedule-only knob moved a bit"
                ),
            )
    for a in range(len(arms)):
        if not arms[a].is_lgbm:
            print("digest", arms[a].name, digests[a])

    var drop = repeats // 3
    print("--- reductions ---")
    print("plateau_drops_first:", drop)
    for a in range(len(arms)):
        var name = arms[a].name
        var line = String(name, "_samples:")
        for i in range(len(samples[a])):
            line += String(" ", samples[a][i])
        print(line)
        var tail = _plateau(samples[a], drop)
        var med = _median(tail)
        var spread_pct = 0.0
        if med > 0.0:
            spread_pct = 100.0 * _spread(tail) / med
        print(name, "plateau_median_s:", med)
        print(name, "plateau_min_s:", _lo(tail), "plateau_max_s:", _hi(tail))
        print(name, "plateau_spread_pct_of_median:", spread_pct)

    # The four cells, as paired ratios. `scaling` is one engine against
    # itself; `gap` is the two engines against each other at one thread count.
    # The ratio of ratios is `gap_auto / gap_w1` and it is the number this
    # lane owns.
    print("--- paired ---")
    var knobs = List[String]()
    knobs.append(String("base"))
    knobs.append(String("noblk"))
    knobs.append(String("lgbmlike"))
    knobs.append(String("g16"))
    knobs.append(String("tpc1"))
    knobs.append(String("tpc16"))
    knobs.append(String("pool"))
    knobs.append(String("minops"))
    knobs.append(String("mintask"))
    knobs.append(String("floor0"))
    knobs.append(String("rowsmall"))
    for p in range(len(knobs)):
        var i = _index_of(arms, String("w1_", knobs[p]))
        var j = _index_of(arms, String("wA_", knobs[p]))
        if i >= 0 and j >= 0:
            _paired(
                String("scaling"),
                arms[i].name,
                arms[j].name,
                samples[i],
                samples[j],
                drop,
            )
    var l1 = _index_of(arms, String("w1_lgbm"))
    var la = _index_of(arms, String("wA_lgbm"))
    if l1 >= 0 and la >= 0:
        _paired(
            String("scaling"),
            arms[l1].name,
            arms[la].name,
            samples[l1],
            samples[la],
            drop,
        )
    for p in range(len(knobs)):
        var i = _index_of(arms, String("w1_", knobs[p]))
        if i >= 0 and l1 >= 0:
            _paired(
                String("gap_w1"),
                arms[i].name,
                arms[l1].name,
                samples[i],
                samples[l1],
                drop,
            )
        var j = _index_of(arms, String("wA_", knobs[p]))
        if j >= 0 and la >= 0:
            _paired(
                String("gap_auto"),
                arms[j].name,
                arms[la].name,
                samples[j],
                samples[la],
                drop,
            )
    # Each candidate against the shipped default, in its own cell.
    var base_a = _index_of(arms, String("wA_base"))
    var base_1 = _index_of(arms, String("w1_base"))
    for p in range(1, len(knobs)):
        var j = _index_of(arms, String("wA_", knobs[p]))
        if j >= 0 and base_a >= 0:
            _paired(
                String("candidate_auto"),
                arms[base_a].name,
                arms[j].name,
                samples[base_a],
                samples[j],
                drop,
            )
        var i = _index_of(arms, String("w1_", knobs[p]))
        if i >= 0 and base_1 >= 0:
            _paired(
                String("candidate_w1"),
                arms[base_1].name,
                arms[i].name,
                samples[base_1],
                samples[i],
                drop,
            )

    # The verdict, on medians: resolved when two medians differ by more than
    # the wider arm's own plateau spread. Printed as the word, and printed as
    # `indistinguishable` when it is.
    print("--- verdicts ---")
    for p in range(1, len(knobs)):
        var j = _index_of(arms, String("wA_", knobs[p]))
        if j < 0 or base_a < 0:
            continue
        var t0m = _plateau(samples[base_a], drop)
        var t1m = _plateau(samples[j], drop)
        var m0 = _median(t0m)
        var m1 = _median(t1m)
        var s0 = _spread(t0m)
        var s1 = _spread(t1m)
        var widest = s0 if s0 > s1 else s1
        var delta = m0 - m1
        var mag = delta if delta > 0.0 else -delta
        print(
            arms[base_a].name,
            "vs",
            arms[j].name,
            "median_delta_s:",
            delta,
            "widest_plateau_spread_s:",
            widest,
            "verdict:",
            String("resolved") if mag > widest else String(
                "indistinguishable"
            ),
            "ratio:",
            m0 / m1 if m1 > 0.0 else 0.0,
        )
