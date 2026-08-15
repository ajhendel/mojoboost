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

    def describe(self) -> String:
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
