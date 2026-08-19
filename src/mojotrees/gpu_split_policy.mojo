"""Eligibility policy for GPU split selection.

This module chooses between the host histogram scan and the resident-device
scan.  It is deliberately pure: reported hardware facts, workload facts and
eligibility in; a structured decision out.  It imports no GPU runtime and can
therefore be tested on a CPU-only machine.  Its one package import is
`growth_policy`, which is itself importless, so that the growth-policy codes
this module reports cannot drift from the ones the growers use.

**IT IS ELIGIBILITY ONLY, AS OF 2026-08-16.**  It used to weigh profitability
too, against a measured crossover, and version 2 of it carried two thresholds
because the crossover was believed to differ by growth policy.  The sweep
those thresholds were owed was run and there is no crossover at any shape from
5.0M to 70.0M normalized work: the device search wins everywhere and wins by
MORE at the smaller shapes.  Both thresholds are deleted and none replaces
them.  The block above `split_reason_name` carries the numbers and the
argument.

What remains is three eligibility tests, and eligibility is a different kind
of question: it asks whether the device search can serve a configuration at
all, which has the same answer at every size.  A path that cannot preserve the
requested tree semantics, or whose resident frontier does not fit, is never
selected.  Unknown devices stay on the host scan -- not because they are too
small, but because nothing in this repository has ever run the device split
search on one.  Explicit ``device`` remains the trainer's hard requirement and
bypasses this policy.
"""

from .growth_policy import grow_policy_name


comptime SPLIT_POLICY_HOST = 1
comptime SPLIT_POLICY_DEVICE_RESIDENT = 2

comptime SPLIT_REASON_UNSUPPORTED = 1
comptime SPLIT_REASON_RESIDENT_MEMORY = 2
comptime SPLIT_REASON_UNKNOWN_HARDWARE = 3
# RETIRED 2026-08-16 and reserved rather than reused. `decide_split_search`
# no longer returns it: it named the profitability comparison, and there is
# no profitability comparison. Its number is not recycled, and
# `split_reason_name` still renders it, so a trace line or a serialized
# decision from before that date reads back as what it was instead of as
# something else.
comptime SPLIT_REASON_BELOW_CROSSOVER = 4
comptime SPLIT_REASON_VALIDATED_WORKLOAD = 5
"""Eligible, and on hardware this has been measured on.

The name says "workload" and no workload is weighed any more; it is kept
rather than renamed because it is a public constant and renaming it would
churn every reader to say something the docstring can say instead."""
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
# added. It selected which of two thresholds a fit was weighed against until
# 2026-08-16; both thresholds are now deleted, so it selects nothing and is
# reported only.
#
# It is deliberately not spelled `GROW_LEAFWISE`. A decision that says
# `grow_policy=leafwise` should mean somebody passed leaf-wise, and a decision
# that says `grow_policy=unspecified` should say so.
comptime SPLIT_GROW_UNSPECIFIED = -1

# Normalization makes unlike shapes comparable without pretending rows alone
# determine the cost.  The measured workload used 255 bins and 31 leaves.
comptime SPLIT_REFERENCE_BINS = 255
comptime SPLIT_REFERENCE_LEAVES = 31

# Version 1: one threshold, measured under leaf-wise growth, applied to all
# growth policies.
# Version 2: the leaf-wise threshold unchanged, plus a separate and much lower
# depth-wise floor, added 2026-08-15 from the sweep II addendum.
# Version 3: BOTH THRESHOLDS WITHDRAWN, 2026-08-16, on a four-shape
# interleaved sweep that found no crossover to install. This module now
# decides eligibility only.
#
# Bump this when a threshold moves, when a rule is added, narrowed or
# withdrawn, or when `normalized_split_work` changes what it measures. Do not
# bump it for a comment. `device_policy.POLICY_VERSION` carries the same
# contract for the CPU/GPU device choice and is a separate number: the two
# policies are versioned independently because they are installed from
# different measurements.
comptime SPLIT_POLICY_VERSION = 3

# --- There is no crossover, and this is where one used to be ---------------
#
# Two thresholds stood here until 2026-08-16. Rule 1 sent a leaf-wise fit to
# the host scan below 50,000,000 normalized work; rule 2 lowered that to
# 12,500,000 for a depth-wise fit at or above 250,000 rows and 50 features.
# Both were installed from a single measured point with a ~2 percent margin,
# and rule 1's own `M4_EVIDENCE_SOURCE` said out loud that no benchmark file
# was ever committed for it.
#
# THE SWEEP THEY WERE OWED WAS RUN AND THERE IS NO CROSSOVER. Four shapes,
# `gpu-host` against `gpu-device`, interleaved in one process, five repeats
# each, the box verified quiet at every shape boundary:
#
#   shape            normalized work   host scan   device plane   ratio
#   100,000 x 50           5.0M          1.695        0.918       1.85x
#   250,000 x 100         25.0M          2.494        1.862       1.34x
#   463,715 x 90          41.7M          3.041        2.309       1.32x
#   700,000 x 100         70.0M          4.041        3.128       1.29x
#
# Every shape resolved and every range was disjoint: at each one the device
# plane's slowest repeat beat the host scan's fastest. The range spans both
# retired thresholds, so neither is a boundary the sweep failed to bracket.
#
# THE PREMISE WAS INVERTED, WHICH IS THE FINDING RATHER THAN THE MARGIN. A
# profitability gate assumes the device plane is not worth its fixed cost
# below some size. Its advantage is LARGEST at the smallest shape and shrinks
# as the work grows -- 1.85x down to 1.29x -- which is what you see when the
# thing being avoided is per NODE rather than per row. The host scan pays,
# per node, a full `3 * n_features * n_bins` download, a host block, and a
# Float64 dequantization of those cells; fewer rows means those dominate
# more, not less. So the gate was protecting the losing arm hardest at
# exactly the sizes where it lost worst, and no measured value replaces it:
# there is nothing to install, and no evidence file to write, because there
# is no crossover to record.
#
# WHAT SURVIVES, AND WHY IT IS NOT A THRESHOLD. Three gates remain below and
# all three are ELIGIBILITY rather than profitability -- they answer "can the
# device search serve this at all", which is a question about capability and
# has the same answer at every size:
#
#   - the semantic refusal (`TreeParams.extra`, `feature_fraction_bylevel`),
#   - the resident-frontier memory answer,
#   - the validated-device table (`split_device_is_validated`).
#
# The third deserves a sentence, because it looks like the hardware scope of
# a deleted threshold and is not. The sweep was run on one machine, so sending
# unmeasured hardware to the device arm on the strength of an M4 measurement
# would be installing a performance claim about a machine nobody owns -- the
# exact move this module exists to refuse. It stays until somebody measures a
# second device.
#
# UPDATED 2026-08-19. When that was written, CUDA and HIP both read "not run"
# in `docs/GPU_VALIDATION.md` and one sentence covered both. It no longer
# does: an RTX 5090 ran this code on 2026-08-18 and passed 66 GPU assertions,
# while AMD remains at zero. The gate's ANSWER is unchanged for both, because
# correctness is not a crossover and neither has a sweep, but the two are no
# longer in the same state and the reason a reader is given now says which is
# which. That is what the validated-device table below exists for.
#
# SCOPE, because it is easy to overstate. This compares `SPLIT_POLICY_HOST`
# against `SPLIT_POLICY_DEVICE_RESIDENT`, and both of them are GPU fits. It
# says nothing about the CPU/GPU crossover, which is `AUTO_GPU_MIN_ROWS` in
# device_policy.mojo and is untouched. The claim is "within a GPU fit, the
# device split plane always wins", not "the GPU always wins".



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


def _is_metal(api: String) -> Bool:
    return api.find("metal") >= 0 or api.find("Metal") >= 0


# ---------------------------------------------------------------------------
# THE VALIDATED-DEVICE TABLE
#
# WHY THIS IS A TABLE AND NOT A BOOLEAN. Until 2026-08-19 this was a single
# predicate named `_is_observed_m4`, and its shape encoded an assumption that
# had stopped being true: that there is one validated device, that it is an
# M4, and that everything else is one undifferentiated "unknown". A second
# device has now executed this code -- an NVIDIA RTX 5090 on 2026-08-18,
# 66 GPU assertions passing across five suites -- and the predicate had no
# way to say anything about it. It answered False and the fit took the host
# scan, which is still the correct ROUTE and was the wrong SILENCE: a reader
# could not tell "nobody has ever run this backend" from "this backend runs,
# and the specific measurement that would promote it has not been taken".
#
# Those are different states and they have different next actions, so they
# are now different rows.
#
# WHAT A ROW MEANS. A device is in the validated table when the
# device-resident split search has been MEASURED against the host scan on it
# and won. Nothing weaker qualifies. Correctness is not enough, and the
# NVIDIA row below is exactly that case: the arithmetic is verified, the
# crossover is not, so it stays out and its fits take the host scan.
#
# The gate is deliberately conservative in the same direction it always was.
# Sending an unmeasured device to the device plane would install a
# performance claim about a machine nobody here owns, and the host scan does
# not produce the same model as the device scan on any backend -- the two
# differ in precision and in when they dequantize -- so the wrong answer here
# silently changes a model, not just a speed.
#
# HOW TO ADD A DEVICE. Two edits, both below, and one prerequisite:
#
#   0. Run the interleaved `gpu-host` against `gpu-device` sweep at several
#      shapes on the new part, with disjoint ranges, and commit the record.
#   1. Add a branch to `validated_split_device` returning a new row code.
#   2. Add the matching branches to `split_device_name` and
#      `split_device_evidence`. `split_device_evidence` must cite a committed
#      file; "it seemed faster" is not a row.
#
# Nothing else in the module reads hardware. `decide_split_search` asks this
# table one question and that is the whole hardware gate.
# ---------------------------------------------------------------------------

comptime SPLIT_DEVICE_UNVALIDATED = -1
comptime SPLIT_DEVICE_APPLE_M4 = 0


def validated_split_device(api: String, arch: String) -> Int:
    """The validated-table row for this device, or `SPLIT_DEVICE_UNVALIDATED`.

    Modular reports ``api=metal, arch_name=4-metal4`` on the development M4.
    Human-readable M4 spellings are accepted as well.  A ten-core Metal device
    is *not* enough on its own: another generation may share that core count
    while having different synchronization and atomic costs, so the generation
    has to be named.
    """
    if _is_metal(api) and (
        arch == "4-metal4"
        or arch.find("Apple M4") >= 0
        or arch.find("apple m4") >= 0
    ):
        return SPLIT_DEVICE_APPLE_M4
    return SPLIT_DEVICE_UNVALIDATED


def split_device_name(row: Int) -> String:
    """The part a validated row stands for, spelled as the record spells it."""
    if row == SPLIT_DEVICE_APPLE_M4:
        return String("Apple M4, 10 core, Metal")
    return String("unvalidated")


def split_device_evidence(row: Int) -> String:
    """The committed record that put a row in the table.

    A row without a citable measurement is not a row.  This is the field that
    makes that rule checkable by reading rather than by trusting.
    """
    if row == SPLIT_DEVICE_APPLE_M4:
        return String(
            "interleaved gpu-host against gpu-device sweep, 2026-08-16, four"
            " shapes from 5.0M to 70.0M normalized work, five repeats each,"
            " every range disjoint, device plane ahead 1.29x to 1.85x"
        )
    return String("none")


def split_device_is_validated(api: String, arch: String) -> Bool:
    """Whether the device-resident split search has been measured here."""
    return validated_split_device(api, arch) != SPLIT_DEVICE_UNVALIDATED


def split_device_note(api: String, arch: String) -> String:
    """What is known about a device, including devices that are not in the
    table.

    `decide_split_search` returns `SPLIT_REASON_UNKNOWN_HARDWARE` for
    everything outside the table, which is accurate and undifferentiated.
    This is the string that differentiates it, for a warning, a trace line or
    a benchmark header.  Three states exist and a reader needs to tell them
    apart:

      - validated: measured, and this fit takes the device plane;
      - run but unmeasured: the backend executes and its crossover has not
        been taken, so the fit takes the host scan and the missing work is a
        benchmark;
      - never run: no hardware of this kind has executed this code, so the
        missing work is an entire validation pass.

    NVIDIA moved from the third state to the second on 2026-08-18 and this
    function exists because nothing could express that.
    """
    var row = validated_split_device(api, arch)
    if row != SPLIT_DEVICE_UNVALIDATED:
        return String("validated on ") + split_device_name(row) + String(
            "; evidence: "
        ) + split_device_evidence(row)
    if _is_metal(api):
        return String(
            "Metal, but not the validated M4 generation. The device plane is"
            " measured 1.29x to 1.85x ahead on the M4 and that number is an"
            " M4 number; this part is unmeasured and takes the host scan"
        )
    if api.find("cuda") >= 0 or api.find("CUDA") >= 0:
        return String(
            "CUDA has executed this code: an RTX 5090 passed 66 GPU"
            " assertions on 2026-08-18 (docs/GPU_VALIDATION.md). What is"
            " missing is the interleaved host-against-device split sweep, not"
            " a first run, and it is blocked behind the training hang"
            " recorded in docs/UPSTREAM_MAX_CUDA_HANG.md. Until that sweep"
            " exists this fit takes the host scan"
        )
    if api.find("hip") >= 0 or api.find("rocm") >= 0:
        return String(
            "HIP has executed this code: an MI300X passed 22 assertions on"
            " 2026-08-19 including a full training fit"
            " (docs/GPU_VALIDATION.md). What is missing is the interleaved"
            " host-against-device split sweep, which is a benchmark and not a"
            " first run. Until it exists this fit takes the host scan. Check"
            " /opt/rocm/.info/version first: MAX targets ROCm 6 and reports no"
            " accelerator on ROCm 7, where the GPU path compiles out silently"
        )
    return String(
        "this backend is not in the validated table and has not been seen"
        " here at all. This fit takes the host scan"
    )


def normalized_split_work(
    n_rows: Int,
    active_features: Int,
    n_bins: Int,
    num_leaves: Int,
) -> Float64:
    """Root-cell work scaled by bins and the leaf budget.

    **REPORTED, NEVER DECIDED ON, as of 2026-08-16.** This was the measure a
    threshold was compared against. There is no threshold: the crossover the
    comparison assumed was measured at four shapes spanning 5.0M to 70.0M of
    exactly this quantity and does not exist, so nothing branches on the value
    any more. It survives because it is a good one-number summary of a fit's
    shape for a trace line, a benchmark header and a bug report, and because
    deleting it would make those three less legible for no gain.

    Read a printed value accordingly: it says how big a fit is, not which path
    it took. `SplitSearchDecision.reason` says which path and why.

    A ranking feature, not a predicted time. Float64 multiplication avoids
    overflowing `Int` for a structurally valid but enormous matrix. Active
    features are used rather than physical columns, which mattered when this
    selected a path and now only keeps the reported number honest about the
    work a feature-subsampled fit will actually do.

    ONE MEASURE FOR BOTH GROWTH POLICIES. The `num_leaves / 31` factor is a
    leaf-wise notion and there was a real argument that depth-wise should be
    ranked by a different formula. That argument is now moot rather than
    settled: it was an argument about where a boundary should sit, and there
    is no boundary. If a future measurement finds a regime the device search
    loses -- which this sweep looked for across a fourteen-fold range of this
    quantity and did not find -- the argument reopens here, and it reopens
    needing a leaf-budget sweep under both policies, which has still never
    been run.
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
    """Which split search a fit takes, and why.

    `threshold` and `evidence_id` were here until 2026-08-16 and are gone
    with the thresholds they described. Nothing replaced them, because a
    decision that compares no number against no threshold has no margin to
    report and no measurement to cite -- see the block above `split_reason_
    name` for the sweep that established there was nothing to compare.
    `normalized_work` stays and is REPORTED ONLY: it is a useful one-number
    summary of a fit's shape and it decides nothing.
    """

    var selected: Int
    var reason: Int
    var normalized_work: Float64
    var grow_policy: Int
    """The growth policy this decision was made for, or
    `SPLIT_GROW_UNSPECIFIED` when the constructing caller did not say. It is
    reported rather than inferred, and it no longer selects a rule: the two
    rules it used to choose between are both retired."""

    def __init__(
        out self,
        selected: Int,
        reason: Int,
        normalized_work: Float64,
        grow_policy: Int = SPLIT_GROW_UNSPECIFIED,
    ):
        self.selected = selected
        self.reason = reason
        self.normalized_work = normalized_work
        self.grow_policy = grow_policy

    def uses_device(self) -> Bool:
        return self.selected == SPLIT_POLICY_DEVICE_RESIDENT

    def describe(self) -> String:
        """One line naming the path, the reason and the shape.

        Printed once per tree under `MOJOTREES_GPU_SPLIT_TRACE`. It used to
        carry a threshold, an evidence identifier and a boundary marker; all
        three named a crossover that was measured on 2026-08-16 and does not
        exist, so printing them would be reporting a comparison nobody made.
        `normalized_work` survives as a shape summary and not as a decision.
        """
        return String(
            "split_strategy=",
            "device-resident" if self.uses_device() else "host",
            " reason=",
            split_reason_name(self.reason),
            " grow_policy=",
            split_grow_policy_name(self.grow_policy),
            " normalized_work=",
            self.normalized_work,
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

    **THIS DOCSTRING DESCRIBED A RULE THE BODY NO LONGER HAS. Corrected
    2026-08-17.** It stated two work thresholds, "device iff
    work >= 50,000,000" for leaf-wise and a lower depth-wise floor of
    "n_rows >= 250,000 and active_features >= 50 and work >= 12,500,000", and
    it stated that nothing threads `TreeParams.grow_policy` in yet so the
    depth-wise rule was "built, tested and unreached until a caller passes
    it". Neither survives a read of the code. Both thresholds were withdrawn on
    2026-08-16 by the four-shape interleaved sweep recorded above
    `split_reason_name`, and `SPLIT_POLICY_VERSION` was bumped to 3 for exactly
    that; and `train_gpu.split_search_decision_for` passes
    `params.grow_policy` on every AUTO call. A reader asking "does depth-wise
    route to the device today" got the wrong answer from the text above, which
    is why it is quoted here rather than merely replaced.

    The rule at head, in the order the body asks it. There is no arithmetic
    left in it.

        HOST, `SPLIT_REASON_UNSUPPORTED`        if not semantics_supported
        HOST, `SPLIT_REASON_RESIDENT_MEMORY`    if not resident_frontier_fits
        HOST, `SPLIT_REASON_UNKNOWN_HARDWARE`   if not in the validated table
        DEVICE_RESIDENT, `SPLIT_REASON_VALIDATED_WORKLOAD`   otherwise

    So every eligible shape on measured hardware takes the device search,
    under every growth policy, at every size. `n_rows`, `active_features`,
    `n_bins` and `num_leaves` are still parameters because
    `normalized_split_work` is still computed and REPORTED on the decision; no
    branch reads it.

    `grow_policy` is likewise carried and not read. It is last and defaults to
    `SPLIT_GROW_UNSPECIFIED` so that callers written against version 1 keep the
    behavior they had, and it selected which of the two thresholds a fit was
    weighed against until those thresholds were deleted. It now travels onto
    the returned `SplitSearchDecision` for the record and decides nothing, so a
    caller that omits it loses a field of provenance and cannot change a route.
    `MOJOTREES_GPU_SPLIT_STRATEGY` is resolved before this function is called
    and is unaffected either way; an explicitly named path never reaches here
    at all.
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
    if not split_device_is_validated(api, arch):
        return SplitSearchDecision(
            SPLIT_POLICY_HOST,
            SPLIT_REASON_UNKNOWN_HARDWARE,
            work,
            grow_policy=grow_policy,
        )
    # Eligible, and on the one device this has been measured on. There is no
    # fourth test: the profitability comparison that used to stand here was
    # measured out of existence on 2026-08-16 (see the block above
    # `split_reason_name`), so every shape that gets this far takes the
    # device search. `n_rows`, `active_features`, `n_bins` and `num_leaves`
    # are still parameters because `work` is still reported.
    return SplitSearchDecision(
        SPLIT_POLICY_DEVICE_RESIDENT,
        SPLIT_REASON_VALIDATED_WORKLOAD,
        work,
        grow_policy,
    )
