"""Conservative workload policy for GPU split selection.

This module chooses between the host histogram scan and the resident-device
scan.  It is deliberately pure: reported hardware facts, workload facts and
eligibility in; a structured decision out.  It imports no GPU runtime and can
therefore be tested on a CPU-only machine.

Eligibility and profitability are separate.  A path that cannot preserve the
requested tree semantics, or whose resident frontier does not fit, is never
selected.  A path that is eligible is selected automatically only under a
hardware profile backed by an end-to-end A/B measurement and only above a
conservative threshold.  Unknown devices stay on the host scan; explicit
``device`` remains the trainer's hard requirement and bypasses this policy.
"""


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

# Normalization makes unlike shapes comparable without pretending rows alone
# determine the cost.  The measured workload used 255 bins and 31 leaves.
comptime SPLIT_REFERENCE_BINS = 255
comptime SPLIT_REFERENCE_LEAVES = 31

# Evidence installed today:
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


struct SplitSearchDecision(Copyable, Movable):
    var selected: Int
    var reason: Int
    var normalized_work: Float64
    var threshold: Float64
    var evidence_id: String

    def __init__(
        out self,
        selected: Int,
        reason: Int,
        normalized_work: Float64,
        threshold: Float64 = 0.0,
        evidence_id: String = String("none"),
    ):
        self.selected = selected
        self.reason = reason
        self.normalized_work = normalized_work
        self.threshold = threshold
        self.evidence_id = evidence_id

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

    def margin(self) -> Float64:
        """Normalized work minus the threshold it was compared against.

        Zero means the workload sits exactly on the crossover and the
        selected path was decided by a single strict comparison of two
        Float64 values that are equal. Positive means device, negative means
        host, and the size says how far the shape would have to move to
        change the answer. Zero for a decision that weighed no workload.
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
        benchmark at 255 bins and 31 leaves is such a workload. Reporting it
        is not a claim that either side is faster; it is a claim that the
        measurement is standing on an edge and should be read knowing that.
        """
        return self.weighed_workload() and (
            self.normalized_work == self.threshold
        )

    def describe(self) -> String:
        var edge = (
            " boundary=exact-threshold" if self.on_crossover_boundary()
            else String("")
        )
        return String(
            "split_strategy=",
            "device-resident" if self.uses_device() else "host",
            " reason=",
            split_reason_name(self.reason),
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
) -> SplitSearchDecision:
    """Choose the automatic split-search path and retain the reason."""
    var work = normalized_split_work(
        n_rows, active_features, n_bins, num_leaves
    )
    if not semantics_supported:
        return SplitSearchDecision(
            SPLIT_POLICY_HOST, SPLIT_REASON_UNSUPPORTED, work
        )
    if not resident_frontier_fits:
        return SplitSearchDecision(
            SPLIT_POLICY_HOST, SPLIT_REASON_RESIDENT_MEMORY, work
        )
    if not _is_observed_m4(api, arch):
        return SplitSearchDecision(
            SPLIT_POLICY_HOST, SPLIT_REASON_UNKNOWN_HARDWARE, work
        )
    if work < M4_MIN_NORMALIZED_WORK:
        return SplitSearchDecision(
            SPLIT_POLICY_HOST,
            SPLIT_REASON_BELOW_CROSSOVER,
            work,
            M4_MIN_NORMALIZED_WORK,
            M4_EVIDENCE_ID,
        )
    return SplitSearchDecision(
        SPLIT_POLICY_DEVICE_RESIDENT,
        SPLIT_REASON_VALIDATED_WORKLOAD,
        work,
        M4_MIN_NORMALIZED_WORK,
        M4_EVIDENCE_ID,
    )
