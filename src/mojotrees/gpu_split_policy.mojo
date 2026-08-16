"""Conservative workload policy for GPU split selection.

This module chooses between the host histogram scan and the resident-device
scan.  It is deliberately pure: reported hardware facts, workload facts and
eligibility in; a structured decision out.  It imports no GPU runtime and can
therefore be tested on a CPU-only machine.  Its one package import is
`growth_policy`, which is itself importless, so that the growth-policy codes
this module now branches on cannot drift from the ones the growers use.

Eligibility and profitability are separate.  A path that cannot preserve the
requested tree semantics, or whose resident frontier does not fit, is never
selected.  A path that is eligible is selected automatically only under a
hardware profile backed by an end-to-end A/B measurement and only above a
conservative threshold.  Unknown devices stay on the host scan; explicit
``device`` remains the trainer's hard requirement and bypasses this policy.

Profitability is also a function of the growth policy, which is what version 2
of this module adds.  The same shape, the same hardware and the same histogram
can be a loss for the device search under leaf-wise growth and a large win
under depth-wise growth, because depth-wise batches a whole level into one
search while the host scan pays a per-node download and a per-node host
synchronization that no batching removes.  One threshold applied to both
policies was therefore right for one of them and wrong for the other.  There
are now two thresholds, each carrying the measurement that installed it.
"""

from .growth_policy import GROW_DEPTHWISE, GROW_LEAFWISE, grow_policy_name


comptime SPLIT_POLICY_HOST = 1
comptime SPLIT_POLICY_DEVICE_RESIDENT = 2

comptime SPLIT_REASON_UNSUPPORTED = 1
comptime SPLIT_REASON_RESIDENT_MEMORY = 2
comptime SPLIT_REASON_UNKNOWN_HARDWARE = 3
comptime SPLIT_REASON_BELOW_CROSSOVER = 4
comptime SPLIT_REASON_VALIDATED_WORKLOAD = 5
# Reasons this module never returns, because they describe a decision that
# was taken before the workload was ever weighed. The trainer builds a
# decision carrying one of these when a caller asked for a path by name or
# through `MOJOTREES_GPU_SPLIT_STRATEGY`, so that everything a user or a
# benchmark reads about the resolved path comes back in one shape whether or
# not the policy was consulted.
comptime SPLIT_REASON_EXPLICIT_REQUEST = 6
comptime SPLIT_REASON_ENVIRONMENT_REQUEST = 7

# The growth policy a decision was made for, when the caller did not say.
#
# This is not a third growth policy. It is the honest record of a caller that
# constructed or requested a decision without threading `TreeParams.grow_policy`
# into it, which is every caller in the tree on the day this constant was
# added: `train_gpu.split_search_decision_for` has a builder and a `TreeParams`
# and does not yet pass the policy through. An unspecified decision is weighed
# against the leaf-wise threshold, because leaf-wise is the default growth and
# the higher of the two thresholds, so treating an unknown caller as leaf-wise
# can only decline a device path, never select one that was not measured.
#
# It is deliberately not spelled `GROW_LEAFWISE`. A decision that says
# `grow_policy=leafwise` should mean somebody passed leaf-wise, and a decision
# that says `grow_policy=unspecified` should tell a reader of a trace line
# exactly why a depth-wise fit did not get the depth-wise floor.
comptime SPLIT_GROW_UNSPECIFIED = -1

# Normalization makes unlike shapes comparable without pretending rows alone
# determine the cost.  The measured workload used 255 bins and 31 leaves.
comptime SPLIT_REFERENCE_BINS = 255
comptime SPLIT_REFERENCE_LEAVES = 31

# Version 1: one threshold, measured under leaf-wise growth, applied to all
# growth policies.
# Version 2: the leaf-wise threshold unchanged, plus a separate and much lower
# depth-wise floor, added 2026-08-15 from the sweep II addendum below.
#
# Bump this when a threshold moves, when a rule is added, narrowed or
# withdrawn, or when `normalized_split_work` changes what it measures. Do not
# bump it for a comment. `device_policy.POLICY_VERSION` carries the same
# contract for the CPU/GPU device choice and is a separate number: the two
# policies are versioned independently because they are installed from
# different measurements.
comptime SPLIT_POLICY_VERSION = 2

# --- Rule 1: leaf-wise growth, Apple M4 ------------------------------------
#
# Evidence installed 2026-08-14:
#
#   Apple M4 / Metal, 250000 x 100, 255 bins, 31 leaves, 100 rounds
#   host scan       3.217 s
#   resident device 3.154 s
#   crossover bracket: device lost at 50000 x 100 and won narrowly at
#   250000 x 100.
#
# The observed winning point is 25 million cells.  AUTO requires twice that
# normalized work because the win at the measured point was only ~2%, smaller
# than ordinary thermal noise.  This threshold is intentionally conservative:
# explicit device selection remains available for measurement and tuning.
#
# A knife edge worth knowing about, and deliberately not filed off. The
# headline benchmark shape (1,000,000 rows, 50 features, 255 bins, 31 leaves)
# normalizes to exactly 50,000,000.0, so it clears this threshold by
# floating-point equality and nothing else: the gate below is `work <
# threshold`, so equal means device. One fewer row, one fewer feature, or any
# `feature_fraction` under 1 puts the same run on the host scan instead. The
# threshold is not moved here, because where it sits is a measured crossover
# and moving it from an argument is exactly what this module refuses to do.
#
# What the edge decides is not only which scan runs. The host scan pays, per
# node, a full `3 * n_features * n_bins` histogram download and a host
# synchronization, and then dequantizes those cells into a Float64
# `Histogram` (`GpuHistogramBuilder.histogram_from_host`), which the hybrid
# scheduler's calibrated model prices at about 10 ns per cell. The
# device-resident scan pays none of that: the histogram is scanned where it
# lives and a 136-byte record crosses instead. So a single row, a single
# feature, or a `feature_fraction` of 0.99 is the difference between paying
# a per-node download, block, and conversion and paying none of them.
# `SplitSearchDecision.uses_device` is the predicate for it.
#
# What is done instead is to make the decision, and this proximity, legible:
# `SplitSearchDecision.margin` and `on_crossover_boundary` report it,
# `describe` prints it, and `tests/test_gpu_split_launch_overhead.mojo` pins
# it so that a later change to the comparison or the formula cannot move the
# benchmark's path in silence.
comptime M4_MIN_NORMALIZED_WORK = 50_000_000.0
comptime M4_EVIDENCE_ID = String("apple-m4-resident-split-2026-08-14-v1")
comptime M4_EVIDENCE_SOURCE = String(
    "in-module record of 2026-08-14; no benchmark file was committed for it,"
    " which is a gap in this rule's provenance rather than in its numbers"
)
comptime M4_MEASURED_ON = String(
    "Apple M4, Metal; 250,000 x 100 dense, 255 bins, 31 leaves, 100 rounds,"
    " leaf-wise growth"
)

# --- Rule 2: depth-wise growth, Apple M4 -----------------------------------
#
# Evidence installed 2026-08-15, from the addendum to
# `bench/results/sweep2_2026-08-15/RESULTS.md` ("Addendum: the path taken,
# which the sweep above failed to record"). Apple M4, 10 cores, 16 GB, Mojo
# 1.0.0 (ed45d567), repository commit 7443673, quiet machine, five repeats per
# arm, median reported, 250,000 rows x 50 dense features, 255 bins, 31 leaves,
# 100 rounds, squared error, single output:
#
#   arm at 250,000 x 50   automatic (host scan)   device search forced
#   GPU leaf-wise                    1.967 s               2.268 s
#   GPU depth-wise                   1.909 s               1.214 s
#
# That shape normalizes to exactly 12,500,000, a quarter of rule 1's
# threshold, so both arms took the host scan automatically and the forced
# column is what the gate was costing. Leaf-wise is 15 percent worse on the
# device search there, which is exactly what rule 1 exists to prevent, so rule
# 1 is not touched. Depth-wise is 37 percent better, 0.70 seconds at this
# shape.
#
# Why the same gate is right for one policy and wrong for the other. Both
# policies split the same number of nodes for the same leaf budget, so the
# host scan downloads the same number of histograms either way. What differs
# is the number of *searches*: depth-wise commits a whole level at once, so
# the device search is launched about once per depth (roughly 5 per tree at 31
# leaves) instead of about once per split (roughly 30). Batching a level does
# nothing for the host scan, whose per-node download, host synchronization and
# Float64 dequantization survive any amount of batching above them, so
# depth-wise moves the crossover down and leaf-wise does not.
#
# WHAT THIS RULE IS NOT. It is one point. The device search won at 12,500,000
# normalized work; where it *starts* winning is unmeasured and is somewhere at
# or below that number. This is a floor, not a fitted threshold, and it was
# not derived by scaling rule 1 by the ratio of launch counts or by anything
# else resembling a model. Nothing below 250,000 rows has been run depth-wise
# with the two paths interleaved, and the 50,000 x 100 point that bracketed
# rule 1 from below was never repeated under depth-wise growth. If the real
# depth-wise crossover is at 2,000,000 normalized work, this rule leaves that
# entire range on the host scan, and finding out costs one interleaved sweep.
#
# Scope, and why it is drawn this narrowly. The floor is the measured point in
# rows and in features as well as in normalized work, in the same shape and for
# the same reason as `device_policy.crossover_rules`: 12,500 rows by 1,000
# features is also 12,500,000 normalized work and is a completely different
# ratio of per-launch cost to per-launch work, and this record says nothing
# about it. A shape under either floor falls back to rule 1, so the depth-wise
# rule can only ever select the device search where rule 1 would not; it never
# declines one rule 1 would have taken.
#
# The measured point sits exactly on its own threshold, the way the headline
# leaf-wise benchmark sits exactly on rule 1's. `normalized_split_work(250_000,
# 50, 255, 31)` is 12,500,000.0 and the gate is `work < threshold`, so the
# measured shape takes the device path by floating-point equality and one fewer
# row puts it back on the host scan. `on_crossover_boundary` reports it.
#
# What would falsify this floor, in which case narrow or withdraw it rather
# than patch it:
#
#   - an interleaved forced-host against forced-device depth-wise pair at
#     250,000 x 50 on an M4, in one window on an idle machine, where the
#     device search is not faster;
#   - the same at a leaf budget far from 31, where the launch-count argument
#     above predicts the gap should change and nothing has measured it;
#   - the same at a bin count far from 255, for the same reason;
#   - a depth-wise fit whose device-search tree differs from its host-scan
#     tree beyond the known Float32 near-tie behavior of the device scan,
#     which would make this a correctness change and not a threshold at all.
comptime M4_DEPTHWISE_MIN_NORMALIZED_WORK = 12_500_000.0
comptime M4_DEPTHWISE_MIN_ROWS = 250_000
comptime M4_DEPTHWISE_MIN_FEATURES = 50
comptime M4_DEPTHWISE_EVIDENCE_ID = String(
    "apple-m4-depthwise-level-batched-split-2026-08-15-v1"
)
comptime M4_DEPTHWISE_EVIDENCE_SOURCE = String(
    "bench/results/sweep2_2026-08-15/RESULTS.md 'Addendum: the path taken,"
    " which the sweep above failed to record'"
)
comptime M4_DEPTHWISE_MEASURED_ON = String(
    "Apple M4, 10 cores, Metal, Mojo 1.0.0 (ed45d567), commit 7443673;"
    " 250,000 x 50 dense, 255 bins, 31 leaves, 100 rounds, squared error,"
    " single output, depth-wise growth; five repeats, median"
)


def split_reason_name(reason: Int) -> String:
    if reason == SPLIT_REASON_UNSUPPORTED:
        return String("unsupported-parameters")
    if reason == SPLIT_REASON_RESIDENT_MEMORY:
        return String("resident-memory")
    if reason == SPLIT_REASON_UNKNOWN_HARDWARE:
        return String("unknown-hardware")
    if reason == SPLIT_REASON_BELOW_CROSSOVER:
        return String("below-crossover")
    if reason == SPLIT_REASON_VALIDATED_WORKLOAD:
        return String("validated-workload")
    if reason == SPLIT_REASON_EXPLICIT_REQUEST:
        return String("explicit-request")
    if reason == SPLIT_REASON_ENVIRONMENT_REQUEST:
        return String("environment-request")
    return String("unknown")


def split_grow_policy_name(grow_policy: Int) -> String:
    """The growth policy a decision was weighed for, including "unspecified".

    Delegates to `growth_policy.grow_policy_name` for the real policies so a
    trace line and a parameter dump spell them the same way.
    """
    if grow_policy == SPLIT_GROW_UNSPECIFIED:
        return String("unspecified")
    return grow_policy_name(grow_policy)


def split_evidence_citation(evidence_id: String) -> String:
    """Where a rule's numbers live and what device they were taken on.

    The form `device_policy.CrossoverEvidence.cite` uses, and for the same
    reason: a decision that says "the device search, on evidence" is worth
    what the reader can go and check, so the source and the machine travel
    with the identifier. Returns "none" for a decision that cited no rule,
    which is every refusal and every explicitly requested path.
    """
    if evidence_id == M4_EVIDENCE_ID:
        return String(M4_EVIDENCE_SOURCE, " on ", M4_MEASURED_ON)
    if evidence_id == M4_DEPTHWISE_EVIDENCE_ID:
        return String(
            M4_DEPTHWISE_EVIDENCE_SOURCE, " on ", M4_DEPTHWISE_MEASURED_ON
        )
    return String("none")


def _is_metal(api: String) -> Bool:
    return api.find("metal") >= 0 or api.find("Metal") >= 0


def _is_observed_m4(api: String, arch: String) -> Bool:
    """Whether the report is the exact M4 profile measured above.

    Modular currently reports ``api=metal, arch_name=4-metal4`` on the
    development M4.  Human-readable M4 spellings are accepted as well.  A
    ten-core Metal device is *not* enough: another generation may share that
    core count while having different synchronization and atomic costs.
    """
    if not _is_metal(api):
        return False
    return (
        arch == "4-metal4"
        or arch.find("Apple M4") >= 0
        or arch.find("apple m4") >= 0
    )


def normalized_split_work(
    n_rows: Int,
    active_features: Int,
    n_bins: Int,
    num_leaves: Int,
) -> Float64:
    """Root-cell work scaled by bins and the leaf budget.

    This is a ranking feature, not a predicted time.  Float64 multiplication
    avoids overflowing ``Int`` for a structurally valid but enormous matrix.
    Active features are used rather than physical columns so feature
    subsampling does not accidentally select the device for work it will not
    perform.

    ONE MEASURE FOR BOTH GROWTH POLICIES, DELIBERATELY. The `num_leaves / 31`
    factor is a leaf-wise notion: under leaf-wise growth the leaf budget is
    also roughly the number of split searches, and under depth-wise growth it
    is not, because a level is searched at once and the number of searches per
    tree is closer to the depth. So there is a real argument that depth-wise
    should be ranked by a different formula, and it is not taken, for two
    reasons.

    The first is that with a single measured point per growth policy, the
    measure and the threshold are not separately identified. Any monotone
    rescaling of the measure is exactly absorbed by rescaling the threshold and
    reproduces the same decision at the measured shape; what it changes is only
    how the rule extrapolates to shapes nobody ran. Choosing that extrapolation
    from an argument about launch counts is installing a curve from reasoning,
    which is the thing this module exists to refuse. A leaf-budget sweep under
    both policies would identify it, and that sweep has not been run.

    The second is that the launch-count argument does not by itself say which
    way the boundary moves. Depth-wise reduces the number of searches, but the
    host path's per-node histogram download, host synchronization and Float64
    dequantization are per node under both policies, and node count tracks the
    leaf budget under both. The quantity that changes with the leaf budget is
    therefore not only the device path's cost, and the sign of the net effect
    at a leaf budget far from 31 is a measurement, not a derivation.

    So the measure is shared and only the threshold differs. When the leaf
    sweep exists, this docstring is where the case for a per-policy measure
    should be reopened; changing it for one policy without the other would make
    the two thresholds incomparable and is worse than either alternative.
    """
    if n_rows <= 0 or active_features <= 0 or n_bins <= 0 or num_leaves <= 0:
        return 0.0
    return (
        Float64(n_rows)
        * Float64(active_features)
        * Float64(n_bins)
        / Float64(SPLIT_REFERENCE_BINS)
        * Float64(num_leaves)
        / Float64(SPLIT_REFERENCE_LEAVES)
    )


def depthwise_floor_applies(
    grow_policy: Int, n_rows: Int, active_features: Int
) -> Bool:
    """Whether rule 2's lower floor covers this (policy, shape) pair.

    True only for depth-wise growth at or above the rows and features that
    were measured. Everything else, including leaf-wise growth, an
    unspecified growth policy, and a depth-wise shape narrower or shorter
    than the measured one, is weighed against rule 1 instead. Public because
    the question "which rule would decide this shape" is worth being able to
    ask without building a decision or opening a device.
    """
    return (
        grow_policy == GROW_DEPTHWISE
        and n_rows >= M4_DEPTHWISE_MIN_ROWS
        and active_features >= M4_DEPTHWISE_MIN_FEATURES
    )


def split_threshold_for(
    grow_policy: Int, n_rows: Int, active_features: Int
) -> Float64:
    """The normalized-work threshold this (policy, shape) pair is weighed
    against.

    12,500,000 where rule 2 applies and 50,000,000 everywhere else. Because
    rule 2's floor is strictly the lower of the two, a shape that selects the
    device search under leaf-wise growth selects it under depth-wise growth as
    well; the policy awareness only ever adds device selection.
    """
    if depthwise_floor_applies(grow_policy, n_rows, active_features):
        return M4_DEPTHWISE_MIN_NORMALIZED_WORK
    return M4_MIN_NORMALIZED_WORK


def split_evidence_for(
    grow_policy: Int, n_rows: Int, active_features: Int
) -> String:
    """The evidence identifier for the rule that will decide this shape.

    Paired with `split_threshold_for` through `depthwise_floor_applies`, so a
    threshold and the record it came from cannot be reported from different
    rules.
    """
    if depthwise_floor_applies(grow_policy, n_rows, active_features):
        return M4_DEPTHWISE_EVIDENCE_ID
    return M4_EVIDENCE_ID


struct SplitSearchDecision(Copyable, Movable):
    var selected: Int
    var reason: Int
    var normalized_work: Float64
    var threshold: Float64
    var evidence_id: String
    var grow_policy: Int
    """The growth policy this decision was weighed for, or
    `SPLIT_GROW_UNSPECIFIED` when the constructing caller did not say. It is
    reported rather than inferred: two rules exist and a reader of a trace
    line should not have to guess which fit produced it."""

    def __init__(
        out self,
        selected: Int,
        reason: Int,
        normalized_work: Float64,
        threshold: Float64 = 0.0,
        evidence_id: String = String("none"),
        grow_policy: Int = SPLIT_GROW_UNSPECIFIED,
    ):
        self.selected = selected
        self.reason = reason
        self.normalized_work = normalized_work
        self.threshold = threshold
        self.evidence_id = evidence_id
        self.grow_policy = grow_policy

    def uses_device(self) -> Bool:
        return self.selected == SPLIT_POLICY_DEVICE_RESIDENT

    def weighed_workload(self) -> Bool:
        """Whether this decision came from comparing work to a threshold.

        False for the eligibility refusals, for unknown hardware, and for a
        path a caller named outright: in all of those the threshold and the
        margin below say nothing.
        """
        return (
            self.reason == SPLIT_REASON_BELOW_CROSSOVER
            or self.reason == SPLIT_REASON_VALIDATED_WORKLOAD
        )

    def used_depthwise_floor(self) -> Bool:
        """Whether the lower depth-wise threshold is what decided this.

        The evidence identifier is the rule's identity, so this is the same
        question as "did this decision cite rule 2". A depth-wise fit at a
        shape below the measured rows or features answers False, because it
        was weighed against the leaf-wise threshold, and that is a fact worth
        being able to read directly off the decision rather than inferring
        from the printed number.
        """
        return (
            self.weighed_workload()
            and self.evidence_id == M4_DEPTHWISE_EVIDENCE_ID
        )

    def margin(self) -> Float64:
        """Normalized work minus the threshold it was compared against.

        Zero means the workload sits exactly on the crossover and the
        selected path was decided by a single strict comparison of two
        Float64 values that are equal. Positive means device, negative means
        host, and the size says how far the shape would have to move to
        change the answer. Zero for a decision that weighed no workload.

        The threshold subtracted is the one the growth policy selected, so a
        margin is only comparable against another margin from the same rule.
        `used_depthwise_floor` says which one it was.
        """
        if not self.weighed_workload():
            return 0.0
        return self.normalized_work - self.threshold

    def on_crossover_boundary(self) -> Bool:
        """Whether the workload lands exactly on the threshold.

        This is a fact worth surfacing rather than a condition worth acting
        on. The gate is `work < threshold`, so a workload at exactly the
        threshold takes the device path by floating-point equality alone,
        and a shape one row, one feature, or one `feature_fraction` step
        smaller takes the host path instead. The headline 1,000,000 x 50
        benchmark at 255 bins and 31 leaves is such a workload under rule 1,
        and the 250,000 x 50 shape rule 2 was measured at is such a workload
        under rule 2. Reporting it is not a claim that either side is faster;
        it is a claim that the measurement is standing on an edge and should
        be read knowing that.
        """
        return self.weighed_workload() and (
            self.normalized_work == self.threshold
        )

    def cite(self) -> String:
        """The source and machine behind the rule this decision used."""
        return split_evidence_citation(self.evidence_id)

    def describe(self) -> String:
        """One line naming the path, the reason, the numbers and the rule.

        `grow_policy` and `evidence` together identify which of the two
        thresholds was applied: a depth-wise fit that shows the leaf-wise
        evidence identifier was outside rule 2's measured shape and fell back
        to rule 1. `cite` expands the identifier into its source and machine
        and is deliberately not inlined here, because this line is printed
        once per tree under `MOJOTREES_GPU_SPLIT_TRACE`.
        """
        var edge = (
            " boundary=exact-threshold" if self.on_crossover_boundary()
            else String("")
        )
        return String(
            "split_strategy=",
            "device-resident" if self.uses_device() else "host",
            " reason=",
            split_reason_name(self.reason),
            " grow_policy=",
            split_grow_policy_name(self.grow_policy),
            " normalized_work=",
            self.normalized_work,
            " threshold=",
            self.threshold,
            " evidence=",
            self.evidence_id,
            edge,
        )


def decide_split_search(
    api: String,
    arch: String,
    n_rows: Int,
    active_features: Int,
    n_bins: Int,
    num_leaves: Int,
    semantics_supported: Bool,
    resident_frontier_fits: Bool,
    grow_policy: Int = SPLIT_GROW_UNSPECIFIED,
) -> SplitSearchDecision:
    """Choose the automatic split-search path and retain the reason.

    The rule, as arithmetic, after eligibility and hardware have passed:

        leaf-wise, or a caller that did not name a growth policy
            device iff work >= 50,000,000

        depth-wise
            device iff (n_rows >= 250,000 and active_features >= 50
                        and work >= 12,500,000)
                    or work >= 50,000,000

    where `work` is `normalized_split_work`. The depth-wise condition is a
    superset of the leaf-wise one, so making the gate policy-aware moves
    shapes onto the device search and never off it.

    `grow_policy` is last and defaults to `SPLIT_GROW_UNSPECIFIED` so that
    callers written against version 1 keep exactly the behavior they had.
    That default is also the current state of the tree: nothing threads
    `TreeParams.grow_policy` in yet, so rule 2 is built, tested and
    unreached until a caller passes it. `MOJOTREES_GPU_SPLIT_STRATEGY` is
    resolved before this function is called and is unaffected in both
    directions; an explicitly named path never consults either rule.
    """
    var work = normalized_split_work(
        n_rows, active_features, n_bins, num_leaves
    )
    if not semantics_supported:
        return SplitSearchDecision(
            SPLIT_POLICY_HOST,
            SPLIT_REASON_UNSUPPORTED,
            work,
            grow_policy=grow_policy,
        )
    if not resident_frontier_fits:
        return SplitSearchDecision(
            SPLIT_POLICY_HOST,
            SPLIT_REASON_RESIDENT_MEMORY,
            work,
            grow_policy=grow_policy,
        )
    if not _is_observed_m4(api, arch):
        return SplitSearchDecision(
            SPLIT_POLICY_HOST,
            SPLIT_REASON_UNKNOWN_HARDWARE,
            work,
            grow_policy=grow_policy,
        )
    var threshold = split_threshold_for(grow_policy, n_rows, active_features)
    var evidence = split_evidence_for(grow_policy, n_rows, active_features)
    if work < threshold:
        return SplitSearchDecision(
            SPLIT_POLICY_HOST,
            SPLIT_REASON_BELOW_CROSSOVER,
            work,
            threshold,
            evidence,
            grow_policy,
        )
    return SplitSearchDecision(
        SPLIT_POLICY_DEVICE_RESIDENT,
        SPLIT_REASON_VALIDATED_WORKLOAD,
        work,
        threshold,
        evidence,
        grow_policy,
    )
