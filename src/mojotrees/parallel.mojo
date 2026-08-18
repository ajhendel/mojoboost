"""Shared controls for multicore CPU dispatch.

Two environment variables govern every parallel loop in the CPU backend
(histogram accumulation, split scanning, bin fitting, bin transform,
gradient generation, row partitioning, prediction), for reproducible
benchmarking and for tests that must force one path:

- `MOJOTREES_NUM_WORKERS`: `1` forces the serial path; `N > 1` forces the
  parallel path with the work chunked into at most N tasks regardless of
  input size; `0`, unset, or unparsable means auto (task count derived from
  the machine's core count once the work is large enough to amortize
  scheduling).
- `MOJOTREES_PARALLEL_MIN_OPS`: integer override of the auto-mode work
  threshold (`PARALLEL_MIN_OPS`), compared against the caller's op estimate
  (conventionally items * rows touched). This is the whole-loop crossover:
  below it the loop stays serial.
- `MOJOTREES_PARALLEL_MIN_TASK_OPS`: integer override of the *per-task* floor
  (`DEFAULT_MIN_TASK_OPS`), the smallest amount of work a task may be given
  once the loop has cleared the crossover. Its default equals the crossover,
  which is the value this module has always used for both, so setting
  nothing changes nothing. It exists as a separate name because the two are
  separate questions and only the first of them has ever been measured; see
  "Two grains, not one" below.

`binning.mojo` adds one more of the same kind:
`MOJOTREES_BINNING_SELECT_MIN_ROWS` chooses between the two ways a quantile
fit resolves its order statistics (rank selection or a full sort). Both
resolve the same values, so it moves no edge and no bin.

`apple_cpu_policy.mojo` adds four more, all scheduling-only and all
documented there: `MOJOTREES_CPU_TASKS_PER_CORE`, `MOJOTREES_CPU_CORE_POOL`,
`MOJOTREES_CPU_FEATURE_GROUP`, and `MOJOTREES_CPU_COMPACT_MIN_ROWS`. The two
above override them: an explicit worker count bypasses the grain floor and
the core cap alike.

The GPU backend adds its own variables, documented where they are read:
`MOJOTREES_GPU_HIST_STRATEGY` (gpu_tiling.mojo),
`MOJOTREES_GPU_SPLIT_STRATEGY` (train_gpu.mojo; `device` moves per-node
split selection onto the accelerator, `host`, unset, or unrecognized keeps
the host scan), `MOJOTREES_GPU_VERIFY_ROWS` (gpu_active_rows.mojo; `1`
makes every device row partition verify its left count against the
grower's histogram count, one host synchronization per split, off by
default), `MOJOTREES_GPU_TRACE` and `MOJOTREES_GPU_STAGING_SLOTS`
(gpu_runtime.mojo).

Five dispatch shapes are provided. All of them keep every floating-point
summation order independent of the task count, so every result is
bit-identical to the serial path on every machine and at every worker
setting:

- `dispatch_feature_ranges` hands each task a contiguous half-open range of
  independent units (conventionally features). A unit's accumulation runs
  start to finish inside one task, so no sum is ever reassociated, and a
  kernel that wants to interleave N units for instruction-level parallelism
  gets them in the same call rather than in N.
  Histogram accumulation passes *groups* of N features as the unit rather
  than features, which is what guarantees a task holds whole groups: at one
  feature per unit the splitter could hand a task fewer features than the
  width it was told to interleave, and did (50 features over 40 tasks left
  thirty tasks holding one feature each). The unit count is therefore the
  parallelism a wide interleave has left, and `apple_cpu_policy` bounds the
  width against it.
- `dispatch_features` is the one-unit-at-a-time form of the same thing,
  written in terms of it.
- `dispatch_feature_rows` splits by feature *and*, when there are fewer
  features than tasks, by rows within each feature. Elementwise callers
  only: it exists so that a matrix with four features over five million
  rows can use more than four workers, which neither of the two above can
  do. With features to spare it is `dispatch_feature_ranges` verbatim.
- `dispatch_rows` splits a row range into contiguous ascending blocks.
  Callers use it only for elementwise work (gradients, predictions, bin
  lookups, sibling subtraction) and for counting passes, where disjoint
  in-order blocks reproduce the serial result exactly. It is deliberately
  not used for histogram accumulation, which would need a cross-block
  reduction.
- `dispatch_regions` runs several *independent* unit spaces under one
  fan-out. It is `dispatch_feature_ranges` over the concatenation of the
  regions, with the callback told which region each range came from, so a
  caller that would otherwise make three `sync_parallelize` calls in a row
  makes one. See "One fan-out, several regions" below; the independence
  requirement there is a correctness precondition, not advice.

Task count comes from the workload shape rather than from the item count.
One task per item balances perfectly but pays a scheduling event per item,
which dominates on wide inputs, so auto mode caps the fan-out at
`TASKS_PER_CORE` tasks per core and chunks the rest. Which cores are counted,
and how the fan-out reacts to a machine whose cores are not all the same
speed, is `apple_cpu_policy.mojo`'s decision; this module only applies the
grain rule and the split.

Ranges are split evenly rather than by a ceiling-divided chunk: task `w` of
`n` takes `[w * items // n, (w + 1) * items // n)`. Two task counts that
differ by one item never leave a trailing task empty, which a ceiling chunk
does (6 tasks over 10 features leaves the sixth with nothing to do), and the
spread makes the remainder land on different tasks instead of all on the
last. Which task runs which unit changes nothing about the result.

Two grains, not one
-------------------
`plan_tasks` asks two questions and they are not the same question. The first
is whether the loop is worth parallelizing at all, which is a statement about
the *total* work against the cost of the first scheduling event. The second
is how finely the loop may be cut, which is a statement about *one task's*
work against the cost of one more scheduling event. This module used to
answer both with `PARALLEL_MIN_OPS`, and the consequence was a rule that
declared a loop worth parallelizing and then ran it on one task: for any
total between one grain and two grains, `total_ops // grain` is 1, and a
one-task fan-out is the serial path with a threshold check attached. That is
strictly worse than staying serial, and it needed no measurement to see. The
task count is now floored at `MIN_TASKS_ABOVE_GRAIN` once the crossover has
been cleared, so clearing the crossover means something.

The second consequence was subtler and is the reason a caller must not
deflate its own estimate to control the fan-out. A stage whose per-item work
is cheaper than a histogram op scales its estimate down, which is correct and
is the documented convention. But under one shared grain that scaling also
moves the go/no-go decision, so a stage that divided its count by enough to
stop over-fanning at one size ended up declared not worth parallelizing at a
size sixteen times larger. Deflating the estimate is the wrong instrument for
the fan-out: the per-task floor is the right one, and it is now separately
named and separately overridable. State the honest work; let the two grains
decide.

Neither grain has been measured against the other, and neither has been
measured against a kernel. What *has* been measured, once, is the thing both
of them are trading against: see "What a fan-out costs, measured" below, and
note that it makes the per-task floor the more suspect of the two, since the
cost of a fan-out rises with its width. `DEFAULT_MIN_TASK_OPS`
equals `PARALLEL_MIN_OPS`, which reproduces the fan-out this module has
always chosen everywhere except the degenerate one-task window above, and
`MOJOTREES_PARALLEL_MIN_TASK_OPS` is there so a sweep can lower it without a
rebuild. Lowering it is the single knob that decides how much of an
asymmetric machine a mid-sized loop reaches, and bench/bench_profile.mojo on
an idle machine is what would settle it.

The core floor, and why the grain alone leaves cores idle
---------------------------------------------------------
`MIN_TASKS_ABOVE_GRAIN = 2` was the admission that the per-task grain and the
crossover cannot both be `PARALLEL_MIN_OPS`: for any total between one grain
and two, the grain permits no legal split, so a floor had to be bolted on.
Two was the smallest floor that is actually parallel. It is not the right
floor, and the reason is arithmetic on the numbers this module already holds.

A loop that clears the crossover has been declared worth a fan-out. A
`sync_parallelize` is one wake of the pool and one barrier at the end, and
that cost is paid in full the moment the loop is declared parallel at all --
it does not shrink because the work was handed to three tasks instead of ten.
So having paid it, a task count below the core count leaves cores idle for
the whole of the barrier it already bought. The split scan is the worked
example: `split_scan_ops(50, 255, two_sided)` is 216,750, and
`216,750 // 65,536` is 3, so a fifty-feature node fans three ways at every
node size in a fit, on any machine, leaving seven of ten cores idle behind a
barrier that was paid for regardless. Nothing about the node's size enters
that number, which is what makes it a defect in the rule rather than a
mis-tuning of it.

The floor is therefore `dispatch_cores`, not 2: once the crossover is
cleared, use the machine that was woken, and go past one task per core only
when the grain independently says each of those tasks still holds a full
grain of work. `MIN_TASKS_ABOVE_GRAIN` survives as the floor under the floor,
for a machine that reports one core.

What bounds this from above -- the reason it is one task per core and not
`max_auto_tasks` -- was that nothing had measured the marginal cost of a
task, so the smallest defensible step past "not enough tasks to fill the
machine" was "exactly enough". Something has now: the sweep in "What a
fan-out costs, measured" prices an empty fan-out at every width this module
can choose, and it rises monotonically with the width. That does not repeal
the floor, which is an argument about idle cores rather than about task cost,
but it does mean the floor is now the first thing to A/B rather than the
last. What bounds it from below is the one measurement this
module does hold, and it is worth writing down because it is easy to lose:
bench/bench_threshold.mojo forces its parallel arm with
`MOJOTREES_PARALLEL_MIN_OPS=1` alone. Under the single-grain rule that read
*both* questions from that variable, so the arm that established the
crossover was a `max_auto_tasks`-way fan-out -- at the crossover row, 65,536
ops over 40 tasks, about 1,638 ops each -- and it beat the serial path by
1.2-1.6x on Apple M4, AMD Zen4 and Neoverse-N2. A task of 1,638 ops paid for
its scheduling event on all three. The floor here gives a loop sitting
exactly at the crossover `PARALLEL_MIN_OPS / dispatch_cores` ops per task,
which at ten cores is 6,554, four times larger than the task size that was
measured to win. Two caveats on leaning on that: the shape measured was
histogram accumulation over independent features, not a row-block split, and
since the two grains were separated that harness no longer reproduces the arm
it recorded (it now forces a two-task fan-out), so the number cannot be
re-taken without also setting `MOJOTREES_PARALLEL_MIN_TASK_OPS=1`.

This floor cannot make any loop parallel that was serial before it: the
crossover is untouched and is tested first. It cannot lower any task count
either, since it only ever raises `by_grain`. What it can do is over-fan a
loop that has just cleared the crossover, and that is the risk it carries.

What a fan-out costs, measured
------------------------------
Every paragraph above this one was written against an unknown constant, and
several of them say so. `bench/bench_dispatch_cost.mojo` measures it: R
`sync_parallelize` calls whose body is one padded store, divided by R, at a
sweep of task counts, with a no-fan-out control that prices the body. It is
the CPU counterpart of `bench_launch_cost.mojo` and it exists for the same
reason -- a per-event cost nobody has measured turns every argument about
scheduling into arithmetic over a variable.

**One run, 2026-08-16, Apple M4 (10 physical / 4 performance), load average
5.05 on a shared box, minimum of five trials of 2000 dispatches each:**

    tasks   1      2      4      8     10     16     25     32     40
    us   0.06   4.84  39.44  51.94 130.50 244.06 276.89 396.23 293.04

with the control (the same body, called 40 times, no `sync_parallelize`) at
0.02 us. Read it for its two orders of magnitude and its shape, not for its
digits: the box was loaded, a barrier-synchronized fan-out finishes when its
slowest task is scheduled, and an idle machine would give smaller numbers.
The 40-task row coming in under the 32-task row is that noise showing.

Three things survive the noise, and all three are mechanism rather than
tuning:

- **The pool is already persistent.** A one-task `sync_parallelize` costs 60
  nanoseconds. Creating a thread costs tens of microseconds on any operating
  system, so no thread is being created here; the runtime holds its workers
  across calls and the one-task case never leaves the calling thread. A
  hand-rolled persistent pool in this module would therefore replace a
  persistent pool with a persistent pool. That is the whole of the argument
  against building one, and it is why this module still has exactly one
  primitive under it.
- **What is paid per dispatch is the wake, and it scales with the width.**
  From 60 nanoseconds at one task to ~5 microseconds at two and ~130 at ten,
  with an empty body and therefore no work to imbalance. Every microsecond in
  that row is fan-out. A wider fan-out is a dearer fan-out, monotonically
  across the measured range, which is the opposite of what a cost model that
  treats the fan-out as a fixed price would predict.
- **Therefore the number of dispatches is a first-class quantity**, not an
  accounting detail. `phase_profile`'s `HOST_*_DISPATCHES` constants are the
  count, and `dispatch_regions` below is the instrument for lowering it.

What this measurement does **not** settle, and must not be used to claim: it
does not price any real kernel, it does not say whether the core floor above
is a win or a loss, and it cannot be turned into a speedup for anything. It
prices one event. `MOJOTREES_CPU_TASK_FLOOR=0` against the default is still
the A/B that answers the floor, and this number is the reason that A/B is
now worth running first rather than last: the floor raised the *width* of
exactly the dispatches whose work is smallest (the split scan 3 to 10, the
small-node histogram 4 to 10, the medium-node partition 2 to 10), and the
column above is monotone increasing in width.

One fan-out, several regions
----------------------------
`dispatch_regions` exists because of the row above. Two consecutive
dispatches of ten tasks pay the ten-task wake twice; one dispatch over the
union of their units pays it once and hands each task a longer run. The
saving is exactly one fan-out at the merged width, which is a count a caller
can state and a reader can check, and it is the only thing this shape claims.

The precondition is **independence**, and it is a correctness precondition
rather than a performance one. Two regions may be fused only when no unit of
either reads storage any unit of the other writes. A pipeline must not be
fused: the row-blocked histogram accumulates per `(block, group)` unit and
then folds the per-block partials per output slot, and the fold reads every
block, so those two dispatches are a dependency and the barrier between them
is load-bearing. Fusing them would not be slower, it would be wrong.

What is safely fusible is a set of *siblings*: the two children of a split
searched for their best feature, the several leaves of a depth-wise level
accumulated at once, a zeroing pass over features nobody accumulates beside
an accumulation over the features somebody does. Each of those is one output
range per unit and no cross-unit read, which is the same contract
`dispatch_feature_ranges` already states for one region.

Exactness is unchanged and for the same reason. A unit is never split across
tasks, a task walks its regions in ascending region order, and every unit
runs exactly once. So a fused dispatch executes each region's units in the
same grouping discipline the region's own `dispatch_feature_ranges` would
have, and any caller that was bit-identical across task counts before is
bit-identical across fusions. What *does* change is the work estimate: a
fused call passes the sum of the regions' estimates, so a pair that sat below
the crossover separately can clear it together. That is the correct decision
-- the crossover is a statement about the total work behind one scheduling
event, and after fusion there is one event -- but it is a change in the
answer and is called out here rather than discovered.

Who carries the snapshot
------------------------
`DispatchSettings.resolve()` is called once per fit, in each CPU trainer in
`boosting.mojo`, next to where the fit's `GrowScratch` is built. From there it
travels as a `settings` argument, defaulted to `DispatchSettings.unresolved()`
so that a call site that has not been wired keeps the live reads it always
had. Wired today:

- `boosting.fill_grad_hess` and the score update (`_add_tree_scores`,
  `_add_by_leaf`, `_add_by_traversal`), which is every per-round dispatch the
  CPU trainers make.
- Every dispatch in `histogram.mojo` except the frozen replica builder: the
  accumulation planner, the excluded-feature zeroing, the gradient gather,
  both accumulation ladders, and the sibling subtraction.
- The split scan in `split.mojo`.

Not wired, and this is the gap that matters: `tree.mojo` is what calls the
histogram builders and the split scan once per node, and it does not hold a
snapshot to pass. Until it does, those entry points take the sentinel and the
per-node reads are still made. The receiving half is here so that filling it
in is one field on `GrowScratch` and five argument passes.

Nesting. These dispatches are not reentrant-aware: a `sync_parallelize`
inside a task would oversubscribe the machine with no scheduler to arbitrate
it. Every caller in this package dispatches from the single-threaded part of
its stage, and a kernel that runs inside a task uses the serial helper
instead (`Histogram.reset` rather than a dispatching zero pass, for
example). Anything new that runs inside a task must do the same.
"""

from max.algorithm import sync_parallelize
from std.os import getenv

from .apple_cpu_policy import (
    DEFAULT_TASKS_PER_CORE,
    ResolvedCpuPolicy,
    cpu_profile,
    env_tasks_per_core,
)

# Serial-vs-parallel crossover measured at 25k-50k ops on Apple M4, AMD
# Zen4, and Neoverse-N2 (bench/bench_threshold.mojo); 1 << 16 sits above
# all three with 1.2-1.6x parallel speedup at exactly this size.
comptime PARALLEL_MIN_OPS = 1 << 16

# Smallest work a single task may be given once the crossover above has been
# cleared. Equal to the crossover, which is the value this module applied to
# both questions before they were separated, so the default fan-out is
# unchanged everywhere the old rule produced more than one task. Override
# with `MOJOTREES_PARALLEL_MIN_TASK_OPS`. Nothing has measured it as a
# quantity in its own right.
comptime DEFAULT_MIN_TASK_OPS = PARALLEL_MIN_OPS

# Floor under the core floor. A loop that has cleared the crossover is fanned
# out over at least `dispatch_cores` tasks (see "The core floor" in the module
# docstring); this is what that floor falls back to on a machine that reports
# one core, and it is the reason a fan-out is never one task. One is not an
# option: a one-task fan-out runs the same work on the same core as the serial
# path and pays a scheduling event for the privilege, so a rule that answers 1
# here has decided nothing.
comptime MIN_TASKS_ABOVE_GRAIN = 2

# Denominator of the elementwise per-row cost weight (see
# `elementwise_row_ops`).
comptime ELEMENTWISE_ROW_COST_DEN = 4

# Auto-mode fan-out per core. The value, the core pool it multiplies, and the
# reasoning behind both now live in `apple_cpu_policy.mojo`; this name stays
# because benchmarks and tests print and bound it. Sweep it with
# bench/bench_profile.mojo (or `MOJOTREES_CPU_TASKS_PER_CORE`) on an idle
# machine before treating it as tuned.
comptime TASKS_PER_CORE = DEFAULT_TASKS_PER_CORE


def _env_int(name: String, default: Int) -> Int:
    var s = getenv(name)
    if s.byte_length() == 0:
        return default
    try:
        var n = Int(s)
        if n < 0:
            return default
        return n
    except:
        return default


def env_num_workers() -> Int:
    """0 = auto, 1 = serial, N > 1 = force N-way chunked parallelism."""
    return _env_int("MOJOTREES_NUM_WORKERS", 0)


def env_parallel_min_ops() -> Int:
    return _env_int("MOJOTREES_PARALLEL_MIN_OPS", PARALLEL_MIN_OPS)


def env_parallel_min_task_ops() -> Int:
    """The per-task floor. 0 or unparsable means the default, which equals
    the whole-loop crossover."""
    var n = _env_int("MOJOTREES_PARALLEL_MIN_TASK_OPS", DEFAULT_MIN_TASK_OPS)
    return n if n > 0 else DEFAULT_MIN_TASK_OPS


def env_core_floor() -> Bool:
    """Whether the core floor in `_cap_tasks` is applied. On by default.

    `MOJOTREES_CPU_TASK_FLOOR=0` reverts the fan-out rule to the one that
    shipped before the floor existed, where the task count came from the
    grain alone and bottomed out at `MIN_TASKS_ABOVE_GRAIN`.

    **This exists because the floor is unmeasured.** It was derived from the
    observation that a `sync_parallelize` pays its wake and its barrier in
    full the moment the loop is declared parallel, so a task count below the
    core count leaves cores idle behind a cost already bought. That argument
    is sound and it is still an argument. It changes the task count for every
    caller whose work sits between one grain and the machine ceiling --
    the split scan 3 to 10, the gradient fill 3 to 10 at a million rows, the
    row partition at medium nodes 2 to 10, the histogram at small nodes 4 to
    10 -- and none of those has been timed.

    The histogram at small nodes is the one to watch: it was measured at
    1.20x with four tasks, which is poor enough that the cause could be
    imbalance, which more tasks fix, or barrier cost, which more tasks make
    worse. Arithmetic cannot separate those and this flag is what lets a
    measurement do it.

    A second measurement now leans on the answer without settling it.
    `bench/bench_dispatch_cost.mojo` prices an *empty* fan-out at every width
    (see "What a fan-out costs, measured" in the module docstring) and finds
    it rising monotonically with the width -- with no body, and therefore no
    imbalance to explain any of it. That says the cost this flag turns off is
    real and grows with exactly the quantity the flag raises. It does not say
    the floor is a net loss, because it prices none of the work the extra
    cores go on to do. It says: run the A/B, and run it early.

    It is deliberately NOT a `DispatchSettings` field on the unresolved path
    only: both paths honor it, so an A/B does not accidentally compare a
    snapshot arm against a live arm.
    """
    var s = getenv("MOJOTREES_CPU_TASK_FLOOR")
    return s != "0"


def elementwise_row_ops(n_rows: Int) -> Int:
    """Work estimate, in histogram-op equivalents, for a pass that touches
    `n_rows` rows and does a handful of arithmetic on sequential arrays.

    Gradient and hessian generation, prediction accumulation, and bin lookup
    are all this shape: two or three sequential loads, a few flops, two
    sequential stores, no indirection and no scattered write. A histogram op,
    the unit `plan_tasks` compares against its threshold, is a scattered
    read-modify-write into three arrays, so an elementwise row is worth a
    fraction of one and the estimate says so.

    The fraction is `1 / ELEMENTWISE_ROW_COST_DEN` and it is not measured.
    What is measured bounds it, from one side only: with an *unscaled* count,
    100k rows of gradient generation asked for one task per core and
    bench/bench_profile.mojo timed that fan-out well below the serial path, so
    100k elementwise rows sit below the crossover and the denominator is
    therefore greater than about 1.5. Any denominator of 2 or more keeps that
    finding intact, and 4 keeps it intact with a factor of 2.6 to spare. What
    the denominator does decide, and the only thing it decides, is the row
    count at which the stage starts to fan out: at 4 that is 262,144 rows,
    where the smallest legal fan-out gives each of two tasks 131,072 rows,
    which is thirteen times the 10,000-row task that lost. Nothing here claims
    a speedup at any size.

    It is a named function rather than a divisor written at the call site
    because a call site that writes `n // 16` has silently encoded a
    serial-to-parallel boundary at a million rows, which is inside the shapes
    this library is asked to train on, and has done it in a place where
    nobody reading the caller would look for it.

    Callers whose per-row work is dearer than this, an `exp` per row for the
    Poisson and logistic objectives for instance, are under-counted by it.
    That is conservative in the direction of staying serial and is left alone
    until someone measures the difference between the objectives.
    """
    if n_rows <= 0:
        return 0
    var ops = n_rows // ELEMENTWISE_ROW_COST_DEN
    return ops if ops > 0 else 1


@fieldwise_init
struct DispatchSettings(Copyable, Movable):
    """Every value the dispatch rule reads from outside itself, read once.

    A dispatch asks six questions of the environment and one of the machine:
    `MOJOTREES_NUM_WORKERS`, `MOJOTREES_PARALLEL_MIN_OPS`,
    `MOJOTREES_PARALLEL_MIN_TASK_OPS`, `MOJOTREES_CPU_TASKS_PER_CORE`,
    `MOJOTREES_CPU_CORE_POOL`, `MOJOTREES_CPU_FEATURE_GROUP`,
    `MOJOTREES_CPU_COMPACT_MIN_ROWS`, and the core counts. The tree grower
    asks them once per dispatch and dispatches several times per node, so a
    fit of a hundred trees at thirty-one leaves asks them on the order of a
    hundred thousand times, and every answer is the same. Resolve once, carry
    the value for the length of the fit, pass it to the `_with` forms below.

    Snapshot semantics, and no cache to invalidate. The value does not
    observe a `setenv` that happens after `resolve()`, and there is no hidden
    global that a later reader might get stale. The functions that do not take
    a settings value still read the environment on every call exactly as they
    always have, so a test that flips `MOJOTREES_NUM_WORKERS` mid-process and
    then calls `plan_tasks` or any `dispatch_*` sees the flip. Only code that
    has explicitly resolved a snapshot holds one, and that code invalidates by
    resolving again.
    """

    var policy: ResolvedCpuPolicy
    var num_workers: Int
    var min_ops: Int
    var min_task_ops: Int

    var core_floor: Bool
    """Whether the core floor in `_cap_tasks` applies, from
    `MOJOTREES_CPU_TASK_FLOOR`. Carried in the snapshot rather than read at
    the dispatch so that the resolved and live paths answer the same question
    the same way; see `env_core_floor` for why the flag exists at all."""

    var resolved: Bool
    """Whether this value is a snapshot or the sentinel.

    False only for `unresolved()`, which is what a defaulted threading
    parameter carries at a call site nobody has wired yet. `plan_tasks_with`
    and every `*_with` dispatch below test it first and fall through to the
    live rule, so an unresolved settings value is indistinguishable from not
    having passed one -- same answer, same `getenv` calls, same order.

    Its purpose is staging, not policy. A parameter that had no default would
    have to be filled in at every call site in the package in one commit,
    across files that three different lanes own; with a default, the
    receiving half lands first and each call site is wired when its owner can
    wire it. A hot call site still holding the sentinel is not finished."""

    @staticmethod
    def resolve() raises -> DispatchSettings:
        """One detection of the machine and one read of each variable.

        Raises with `ResolvedCpuPolicy.resolve`, for an off-ladder
        `MOJOTREES_CPU_FEATURE_GROUP`. Taking the snapshot is the one place
        that refusal can be reported once per fit instead of once per node."""
        return DispatchSettings(
            ResolvedCpuPolicy.resolve(),
            env_num_workers(),
            env_parallel_min_ops(),
            env_parallel_min_task_ops(),
            env_core_floor(),
            True,
        )

    @staticmethod
    def unresolved() -> DispatchSettings:
        """The sentinel. No machine detection and no `getenv`, so building one
        costs a few integer stores; it is what every defaulted `settings`
        parameter in this package constructs, once per call, on a path that
        then goes on to read the environment live exactly as it always did."""
        return DispatchSettings(
            ResolvedCpuPolicy.unresolved(), 0, 0, 0, True, False
        )

    def describe(self) -> String:
        if not self.resolved:
            return String("unresolved (live reads)")
        return String(
            self.policy.describe(),
            " workers=",
            self.num_workers,
            " min_ops=",
            self.min_ops,
            " min_task_ops=",
            self.min_task_ops,
            " core_floor=",
            self.core_floor,
        )


@always_inline
def _cap_tasks(
    total_ops: Int, min_task_ops: Int, dispatch_cores: Int, max_auto: Int
) -> Int:
    """Task count for a loop that has already cleared the crossover.

    The one copy of the fan-out rule, so the resolved and the unresolved
    entry points cannot drift. Three bounds, in this order:

    - the grain, `total_ops // min_task_ops`, which is how many tasks may hold
      a full per-task grain of work;
    - the core floor, `dispatch_cores`, raising that answer when the grain
      would leave cores idle behind a barrier the loop has already been
      declared worth paying (see "The core floor" in the module docstring),
      and never below `MIN_TASKS_ABOVE_GRAIN` so a one-core machine still gets
      something that is actually parallel;
    - the machine's ceiling, `max_auto`, which binds last and binds
      absolutely, so a policy that has capped the fan-out is never overridden
      by the floor.

    Monotone in `total_ops` and monotone in the machine, and it never returns
    1: the caller has already decided this loop is worth parallelizing and one
    task does not parallelize it. It is a scheduling decision only -- every
    dispatch shape in this module is documented to give the same values at
    every task count -- so this can be retuned without moving an output.
    """
    var by_grain = max_auto
    if min_task_ops > 0:
        by_grain = total_ops // min_task_ops
    # `dispatch_cores` arrives already reduced to MIN_TASKS_ABOVE_GRAIN when
    # the caller resolved `MOJOTREES_CPU_TASK_FLOOR=0`, so the floor is off
    # without this function reading anything. `_cap_tasks` stays pure, which
    # is what lets the snapshot path call it without touching the
    # environment.
    var floor = dispatch_cores
    if floor < MIN_TASKS_ABOVE_GRAIN:
        floor = MIN_TASKS_ABOVE_GRAIN
    if by_grain < floor:
        by_grain = floor
    return by_grain if by_grain < max_auto else max_auto


def _effective_cores(dispatch_cores: Int, floor_on: Bool) -> Int:
    """`dispatch_cores` when the core floor is on, and the pre-floor value
    when it is off. One place, so the live and snapshot paths cannot drift."""
    return dispatch_cores if floor_on else MIN_TASKS_ABOVE_GRAIN


@always_inline
def _forced_chunks(
    workers: Int, total_ops: Int, per_worker: Int, min_task_ops: Int
) -> Int:
    """Chunks an explicit worker count is cut into. The one copy of the rule,
    so the live and snapshot paths cannot drift, and pure, so the snapshot
    path calling it touches no environment.

    `workers * per_worker`, which is the multiplication auto mode already
    applies to the core count, clamped down by the per-task grain so a small
    phase is not cut into pieces smaller than a scheduling event is worth,
    and clamped **up** to `workers` so this can only ever raise a chunk count.

    Never applied to auto mode, which already multiplies by the same factor
    in `CpuProfile.max_auto_tasks`. `plan_tasks` clamps the result to the item
    count afterwards, as it does for every other answer this module produces.

    WHY THIS IS UNCONDITIONAL, and the measurement that made it so
    --------------------------------------------------------------

    It shipped on 2026-08-18 behind `MOJOTREES_CPU_OVERSUBSCRIBE`, default
    off, whose docstring named the measurement that would delete it in either
    direction. That measurement ran the same day. Batch prediction, real data,
    51,630 rows by 90 features, 100 trees, four configurations interleaved in
    ONE process against one fitted model, medians of five:

        arm          1 worker   forced 10   forced 10   auto
                                10 chunks   40 chunks   (shipped)
        leaf-wise    105.99 ms    34.07       28.97      28.91
        depth-wise    40.77 ms    13.77       12.06      11.83

    Every prediction was bit-identical across all four, which is the gate this
    change is held to and not a tolerance.

    Two things follow and both are recorded because the second is the one a
    later reader needs. The switched arm is faster, 1.18x and 1.14x, so the
    forced path's equal split was really costing something and the fix is
    kept. And the switched arm lands on TOP of auto, 28.97 against 28.91 and
    12.06 against 11.83, which says the multiplication reproduces the shipped
    geometry exactly rather than inventing a third one.

    **What this does NOT fix, so nobody re-opens it as if it might.** Auto
    mode already cut 40 chunks before this change, so no user on a default
    fit gains anything here; the gain is confined to callers who set
    `MOJOTREES_NUM_WORKERS` to their core count and were silently getting the
    worst geometry. And 40 chunks still converts only about 3.5x of ten cores.
    That remaining ceiling is NOT chunk geometry, it reproduces in a
    standalone probe with no mojotrees code in it, and it is recorded as an
    external limit in `docs/design/DECLINED_OPTIMIZATIONS.md` rather than as
    an open lane.
    """
    var per = per_worker if per_worker > 0 else DEFAULT_TASKS_PER_CORE
    if per < 1:
        per = 1
    var chunks = workers * per
    if chunks < workers:
        # Non-positive product: the multiplication overflowed or `workers` is
        # not a sane count. Fall back to the unswitched answer.
        return workers
    if min_task_ops > 0:
        var by_grain = total_ops // min_task_ops
        if chunks > by_grain:
            chunks = by_grain
    if chunks < workers:
        chunks = workers
    return chunks


def plan_tasks_with(
    settings: DispatchSettings, n_items: Int, total_ops: Int
) -> Int:
    """`plan_tasks` against an already-resolved snapshot.

    The same rule on the same numbers, with nothing read from the
    environment and nothing detected about the machine. Every caller on the
    per-node path should be reaching this form; `plan_tasks` remains for
    callers that have no snapshot to hand and for tests that change a
    variable between calls.

    The unresolved sentinel falls through to `plan_tasks`, so a defaulted
    `settings` parameter reads the environment exactly as the call site did
    before the parameter existed.
    """
    if not settings.resolved:
        return plan_tasks(n_items, total_ops)
    if n_items <= 1:
        return 1
    if settings.num_workers == 1:
        return 1
    var n_tasks = settings.num_workers
    if settings.num_workers == 0:
        if total_ops < settings.min_ops:
            return 1
        n_tasks = _cap_tasks(
            total_ops,
            settings.min_task_ops,
            _effective_cores(
                settings.policy.dispatch_cores(), settings.core_floor
            ),
            settings.policy.max_auto_tasks(),
        )
    else:
        # The forced path. `policy.tasks_per_core` is the same number
        # `env_tasks_per_core()` returns on the live path, read once at
        # `resolve()` instead of here.
        n_tasks = _forced_chunks(
            settings.num_workers,
            total_ops,
            settings.policy.tasks_per_core,
            settings.min_task_ops,
        )
    if n_tasks < 1:
        n_tasks = 1
    if n_tasks > n_items:
        n_tasks = n_items
    return n_tasks


def plan_tasks(n_items: Int, total_ops: Int) -> Int:
    """How many parallel tasks to split `n_items` units of work into.

    Returns 1 for the serial path. The result is a scheduling decision only:
    every caller produces the same answer at every task count, so this can be
    tuned freely without changing any output.

    `total_ops` is the caller's work estimate in *histogram-op equivalents*:
    one accumulate of a gradient, a hessian, and a count into a scattered bin.
    Stages whose per-row work is cheaper or dearer than that scale their
    estimate accordingly (see the callers), because a single count of "rows"
    or "features" says nothing about how long a task will run, and the whole
    point of the threshold is to compare work against scheduling overhead.

    Two limits apply in auto mode, and they are two limits rather than one
    written twice (see "Two grains, not one" in the module docstring). Below
    the whole-loop crossover the loop stays serial. Above it the task count is
    capped so that no task holds less than the per-task floor: fanning 100k
    cheap ops across 40 workers costs far more in scheduling than the 2.5k ops
    each one would run.

    Having cleared the crossover the answer is at least one task per core, and
    so is never 1: a one-task fan-out is the serial path plus a scheduling
    event, and a three-task fan-out on ten cores is seven idle cores behind a
    barrier that has already been paid for. That floor is "The core floor" in
    the module docstring, and it is the reason the grain alone -- which for
    the split scan answers 3 at every node size in a fit -- is not the whole
    rule. An explicit `MOJOTREES_NUM_WORKERS` bypasses all of it, so tests can
    force the parallel path at any size. It used to ALSO return the worker
    count verbatim, making the chunk count equal the worker count, which is a
    perfectly even split and the worst shape for a machine whose cores are not
    all the same speed. Since 2026-08-18 a forced count is multiplied by
    `TASKS_PER_CORE` exactly as auto mode multiplies the core count; see
    `_forced_chunks` for the rule and for the measurement that made it
    unconditional.

    The per-core ceiling comes from `apple_cpu_policy`, which decides how many
    cores to count on a machine whose cores are not all the same speed. Its
    default is every physical core times `TASKS_PER_CORE`, which is what this
    line has always computed.

    This form reads the environment on every call, which is what lets a test
    change a variable between two calls and see the change. Anything on the
    per-node path should hold a `DispatchSettings` and call `plan_tasks_with`
    instead; the two compute the same answer from the same numbers.
    """
    if n_items <= 1:
        return 1
    var workers = env_num_workers()
    if workers == 1:
        return 1
    var n_tasks = workers
    if workers == 0:
        var grain = env_parallel_min_ops()
        if total_ops < grain:
            return 1
        # One detection, two questions asked of it. `cpu_profile()` is
        # `CpuProfile.detect()` and re-reads the machine every call, so binding
        # it here is what keeps the live path at the one detection it has
        # always made rather than two.
        var profile = cpu_profile()
        n_tasks = _cap_tasks(
            total_ops,
            env_parallel_min_task_ops(),
            _effective_cores(profile.dispatch_cores(), env_core_floor()),
            profile.max_auto_tasks(),
        )
    else:
        # The forced path. Unconditional since 2026-08-18; see `_forced_chunks`
        # for the measurement that retired the switch that used to guard it.
        n_tasks = _forced_chunks(
            workers,
            total_ops,
            env_tasks_per_core(),
            env_parallel_min_task_ops(),
        )
    if n_tasks < 1:
        n_tasks = 1
    if n_tasks > n_items:
        n_tasks = n_items
    return n_tasks


def dispatch_feature_ranges[FuncType: def (Int, Int) -> None](
    func: FuncType, n_features: Int, total_ops: Int
) raises:
    """Run `func(start, end)` over contiguous ranges covering
    [0, n_features), under the env contract above.

    `func` must be safe to run concurrently for disjoint ranges: each feature
    writes only its own output range, so ranges of them do too. `total_ops`
    is the work estimate compared against the auto-mode threshold,
    conventionally n_features * rows_touched plus whatever else the kernel
    does per feature (see `apple_cpu_policy.derive_accumulation_plan`, which
    counts the zeroing pass a histogram build also runs).

    The serial path is a single `func(0, n_features)` call, which is the same
    shape a task gets, so there is one body to test rather than two. Handing
    a kernel a range rather than one feature at a time is what lets it
    interleave two features in one inner loop; it changes nothing about which
    feature accumulates what, only how many are in flight.
    """
    _run_feature_ranges(func, n_features, plan_tasks(n_features, total_ops))


def dispatch_feature_ranges_with[FuncType: def (Int, Int) -> None](
    settings: DispatchSettings,
    func: FuncType,
    n_features: Int,
    total_ops: Int,
) raises:
    """`dispatch_feature_ranges` against an already-resolved snapshot."""
    _run_feature_ranges(
        func, n_features, plan_tasks_with(settings, n_features, total_ops)
    )


def _run_feature_ranges[FuncType: def (Int, Int) -> None](
    func: FuncType, n_features: Int, n_tasks: Int
) raises:
    """The split itself, with the task count already chosen. One copy, so the
    resolved and unresolved entry points hand out identical ranges."""
    if n_tasks <= 1:
        if n_features > 0:
            func(0, n_features)
        return

    # Even split: task w takes [w * n // tasks, (w + 1) * n // tasks). With
    # n_tasks <= n_features (plan_tasks clamps it) no task comes out empty,
    # and the remainder is spread instead of landing entirely on the last.
    def do_range(w: Int) {imm}:
        func(
            (w * n_features) // n_tasks,
            ((w + 1) * n_features) // n_tasks,
        )

    sync_parallelize(do_range, n_tasks)


def dispatch_features[FuncType: def (Int) -> None](
    func: FuncType, n_features: Int, total_ops: Int
) raises:
    """Run `func(f)` for f in [0, n_features) under the env contract above.

    The one-feature-at-a-time form of `dispatch_feature_ranges`, and written
    in terms of it so there is a single split rule to reason about. Use this
    unless the kernel has something to gain from seeing several features at
    once.
    """

    def do_range(start: Int, end: Int) {imm}:
        for f in range(start, end):
            func(f)

    dispatch_feature_ranges(do_range, n_features, total_ops)


def dispatch_features_with[FuncType: def (Int) -> None](
    settings: DispatchSettings,
    func: FuncType,
    n_features: Int,
    total_ops: Int,
) raises:
    """`dispatch_features` against an already-resolved snapshot."""

    def do_range(start: Int, end: Int) {imm}:
        for f in range(start, end):
            func(f)

    dispatch_feature_ranges_with(settings, do_range, n_features, total_ops)


def region_units(sizes: List[Int]) -> Int:
    """Units a `dispatch_regions` call covers: the sum of the region sizes,
    with a non-positive size counted as zero.

    Public because a caller has to hand `dispatch_regions` a work estimate for
    the same union, and computing the two from different arithmetic is how a
    fan-out ends up sized against a unit space it does not have.
    """
    var total = 0
    for i in range(len(sizes)):
        if sizes[i] > 0:
            total += sizes[i]
    return total


def dispatch_regions[FuncType: def (Int, Int, Int) -> None](
    func: FuncType, sizes: List[Int], total_ops: Int
) raises:
    """Run `func(region, start, end)` over contiguous ranges covering every
    region's units, under one fan-out.

    **THIS HAS NO CALLER, DELIBERATELY, AND THAT IS DECLARED HERE RATHER THAN
    LEFT TO BE DISCOVERED.** It is a receiving half. This repository found
    three built-tested-never-wired mechanisms in a single night --
    `DispatchSettings`, `gpu_gradient_stream.HostGradientStage`, and eight
    functions in `apple_cpu_policy` -- each of which passed its tests because
    the tests called it directly while no production call site existed, and
    each of which had its benefit claimed in a commit message. This one says
    so instead.

    The named cross-lane edit that wires it is `split.find_best_split_pair`:
    hoist the per-node state `scan_feature` closes over into a context struct,
    and replace the two per-child `dispatch_features_with` calls with one
    `dispatch_regions_with` over both children, leaving both serial ascending
    folds unchanged. That is bit-identical by construction -- same per-feature
    scan, same fold order, same strict `>` tie-break.

    Why it is worth wiring, as a derived bound rather than a measurement: the
    split search fans out once per node at width 10 regardless of node size,
    which is 6,001 fan-outs per fit behind the smallest per-node work in the
    round. Fusing a split's two children halves that count, for a derived 44
    percent of the split-search phase, which is itself at most 27 percent of
    the parallel round.

    **If that edit is not sequenced, delete this rather than leave it.** An
    unwired receiving half that nobody has committed to wiring is exactly the
    pathology described above.

    `sizes[r]` is how many units region `r` holds; a region of zero or fewer
    units is skipped and still occupies its index, so a caller may pass a
    fixed-length vector with an inactive region zeroed rather than renumbering
    its regions. `total_ops` is the work estimate for the **union**, in the
    usual histogram-op equivalents, and `region_units` is the matching unit
    count.

    **The regions must be independent.** No unit of any region may read
    storage that a unit of another region writes. This is the same contract
    `dispatch_feature_ranges` states within one region, extended across
    regions, and it is what makes fusing two dispatches into one legal rather
    than merely cheaper. A producer and its consumer -- the row-blocked
    histogram's accumulate and its fold, a count pass and the scatter that
    reads its prefix sums -- are a dependency, the barrier between them is
    load-bearing, and fusing them is a correctness bug and not a tuning
    regression. See "One fan-out, several regions" in the module docstring.

    What it is worth is one fan-out at the merged width, which the caller can
    count. What it cannot do is change a result: a unit is never split across
    tasks, every unit runs exactly once, and a task walks its regions in
    ascending region order, so each region's units are grouped exactly as its
    own `dispatch_feature_ranges` would have grouped them. What it does
    change is the go/no-go decision, since the estimate is now the union's;
    that is deliberate and is documented in the module docstring.

    The serial path calls `func(r, 0, sizes[r])` for each non-empty region in
    ascending order, which is the same shape a task gets, so there is one body
    to test rather than two.
    """
    _run_regions(func, sizes, plan_tasks(region_units(sizes), total_ops))


def dispatch_regions_with[FuncType: def (Int, Int, Int) -> None](
    settings: DispatchSettings,
    func: FuncType,
    sizes: List[Int],
    total_ops: Int,
) raises:
    """`dispatch_regions` against an already-resolved snapshot."""
    _run_regions(
        func, sizes, plan_tasks_with(settings, region_units(sizes), total_ops)
    )


def _run_regions[FuncType: def (Int, Int, Int) -> None](
    func: FuncType, sizes: List[Int], n_tasks: Int
) raises:
    """The split itself, with the task count already chosen. One copy, so the
    resolved and unresolved entry points hand out identical ranges."""
    var n_regions = len(sizes)
    # Prefix offsets over the concatenated unit space: `offsets[r]` is the
    # flat id of region r's first unit and `offsets[n_regions]` is the total.
    # Built once here rather than rescanned inside every task, and held in a
    # local whose lifetime spans the whole `sync_parallelize`, which is
    # synchronous.
    var offsets = List[Int](capacity=n_regions + 1)
    offsets.append(0)
    var running = 0
    for r in range(n_regions):
        if sizes[r] > 0:
            running += sizes[r]
        offsets.append(running)
    var n_units = running
    if n_units <= 0:
        return
    if n_tasks <= 1:
        for r in range(n_regions):
            if sizes[r] > 0:
                func(r, 0, sizes[r])
        return

    var off_p = offsets.unsafe_ptr()

    # The same even split `_run_feature_ranges` uses, over the union: task w
    # takes flat [w * n // tasks, (w + 1) * n // tasks) and cuts it at every
    # region boundary it crosses. `plan_tasks` has clamped n_tasks to n_units,
    # so no task comes out with an empty flat range; a task *can* still emit
    # no call for a given region, which is the point.
    def do_range(w: Int) {imm}:
        var lo = (w * n_units) // n_tasks
        var hi = ((w + 1) * n_units) // n_tasks
        for r in range(n_regions):
            var r0 = off_p.unsafe_load(r)
            var r1 = off_p.unsafe_load(r + 1)
            if r1 <= lo:
                continue
            if r0 >= hi:
                break
            var s = (lo - r0) if lo > r0 else 0
            var e = (hi - r0) if hi < r1 else (r1 - r0)
            if s < e:
                func(r, s, e)

    sync_parallelize(do_range, n_tasks)


def dispatch_feature_rows[FuncType: def (Int, Int, Int) -> None](
    func: FuncType, n_features: Int, n_rows: Int, total_ops: Int
) raises:
    """Run `func(f, row_start, row_end)` over tiles covering
    [0, n_features) x [0, n_rows), under the env contract above.

    For an elementwise kernel: one where the value written at (f, r) depends
    on that cell alone. `dispatch_features` splits such a kernel by feature
    and nothing else, so a matrix with four features cannot use more than
    four workers however many rows it has. Four features by five million rows
    is a real shape (a few engineered signals over a long history), and on
    that shape feature-only splitting leaves most of the machine idle.

    So the split falls back to rows only when it has to. With at least as
    many features as tasks, this *is* `dispatch_feature_ranges`, called
    rather than reimplemented: one task per contiguous run of features,
    exactly the schedule that shape has always had, and no new tasks to pay
    for. Below that, each feature's rows are cut into enough blocks to reach
    the task count, and the tiles are dispatched as one flat list.

    Tiling cannot move a result. Every tile writes only its own
    `(f, [row_start, row_end))` cells, reads only the matching input cells,
    and computes each from that cell alone, so the output is the same bytes
    at every task count and on every machine. That is a property of the
    kernel, not of the schedule, which is why this is restricted to
    elementwise callers: a kernel that reduces across rows (a column
    minimum, a rank, a histogram) needs a combine step and must not use this.

    `total_ops` is the usual work estimate in histogram-op equivalents. It is
    compared against the auto-mode threshold once, over the whole matrix, so
    a shape stays serial here on exactly the same grounds it would elsewhere.
    """
    var n_tasks = plan_tasks(n_features * n_rows, total_ops)
    if n_tasks <= 1:
        for f in range(n_features):
            if n_rows > 0:
                func(f, 0, n_rows)
        return

    if n_features >= n_tasks:
        # Enough features to keep every task busy: the established schedule,
        # reached through the established call so there is one split rule.
        def do_range(start: Int, end: Int) {imm}:
            for f in range(start, end):
                func(f, 0, n_rows)

        dispatch_feature_ranges(do_range, n_features, total_ops)
        return

    # Blocks per feature, rounded up, so `n_features * blocks >= n_tasks` and
    # no worker is left without a tile. Clamped to `n_rows` because a block
    # cannot be shorter than one row.
    var per_feature = (n_tasks + n_features - 1) // n_features
    if per_feature > n_rows:
        per_feature = n_rows
    if per_feature < 1:
        return
    var chunk = (n_rows + per_feature - 1) // per_feature
    # Recount from the chunk, as `plan_row_blocks` does: ceiling division can
    # leave a trailing block empty, and an empty tile is a task that costs
    # scheduling and does nothing.
    var n_blocks = (n_rows + chunk - 1) // chunk

    def do_tile(t: Int) {imm}:
        var f = t // n_blocks
        var b = t - f * n_blocks
        var r0 = b * chunk
        var r1 = r0 + chunk
        if r1 > n_rows:
            r1 = n_rows
        func(f, r0, r1)

    sync_parallelize(do_tile, n_features * n_blocks)


@fieldwise_init
struct RowBlocks(Copyable, Movable):
    """A split of [0, n_rows) into `n_blocks` contiguous ascending blocks.

    Made explicit, rather than hidden inside a dispatch call, because a
    two-pass parallel algorithm has to size its per-block scratch before it
    runs the first pass: the row partitioner counts per block, prefix-sums
    the counts, then scatters. `n_blocks == 1` is the serial plan.
    """

    var n_blocks: Int
    var chunk: Int
    var n_rows: Int

    @always_inline
    def start(self, b: Int) -> Int:
        return b * self.chunk

    @always_inline
    def end(self, b: Int) -> Int:
        var e = (b + 1) * self.chunk
        return self.n_rows if e > self.n_rows else e


def _blocks_for(n_rows: Int, n_tasks: Int) -> RowBlocks:
    """The block geometry, with the task count already chosen."""
    if n_rows <= 0:
        return RowBlocks(0, 1, 0)
    if n_tasks <= 1:
        return RowBlocks(1, n_rows, n_rows)
    var chunk = (n_rows + n_tasks - 1) // n_tasks
    # Ceiling division can leave trailing blocks empty (10 rows over 4 tasks
    # gives chunk 3, and block 3 would be empty); recount from the chunk so
    # every block has work and `n_blocks` is exact.
    return RowBlocks((n_rows + chunk - 1) // chunk, chunk, n_rows)


def plan_row_blocks(n_rows: Int, total_ops: Int) -> RowBlocks:
    """Choose the block split for a row range under the env contract."""
    if n_rows <= 0:
        return RowBlocks(0, 1, 0)
    return _blocks_for(n_rows, plan_tasks(n_rows, total_ops))


def plan_row_blocks_with(
    settings: DispatchSettings, n_rows: Int, total_ops: Int
) -> RowBlocks:
    """`plan_row_blocks` against an already-resolved snapshot."""
    if n_rows <= 0:
        return RowBlocks(0, 1, 0)
    return _blocks_for(n_rows, plan_tasks_with(settings, n_rows, total_ops))


def run_row_blocks[FuncType: def (Int) -> None](
    blocks: RowBlocks, func: FuncType
) raises:
    """Run `func(b)` for each block id in `blocks`, in parallel when the plan
    has more than one block. `func` reads its own range via
    `blocks.start(b)` / `blocks.end(b)`."""
    if blocks.n_blocks <= 0:
        return
    if blocks.n_blocks == 1:
        func(0)
        return
    sync_parallelize(func, blocks.n_blocks)


def dispatch_rows[FuncType: def (Int, Int) -> None](
    func: FuncType, n_rows: Int, total_ops: Int
) raises:
    """Run `func(start, end)` over contiguous blocks covering [0, n_rows).

    Blocks are disjoint and each one walks its rows in ascending order, so
    elementwise callers get bit-identical results at every task count. The
    serial path is a single `func(0, n_rows)` call, which is the same shape a
    block gets, so there is only one body to test.
    """
    var blocks = plan_row_blocks(n_rows, total_ops)

    def do_block(b: Int) {imm}:
        func(blocks.start(b), blocks.end(b))

    run_row_blocks(blocks, do_block)


def dispatch_rows_with[FuncType: def (Int, Int) -> None](
    settings: DispatchSettings, func: FuncType, n_rows: Int, total_ops: Int
) raises:
    """`dispatch_rows` against an already-resolved snapshot."""
    var blocks = plan_row_blocks_with(settings, n_rows, total_ops)

    def do_block(b: Int) {imm}:
        func(blocks.start(b), blocks.end(b))

    run_row_blocks(blocks, do_block)
