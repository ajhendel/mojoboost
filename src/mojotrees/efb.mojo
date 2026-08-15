"""Exclusive feature bundling (EFB).

Sparse data usually has many features that are almost never non-zero at the
same row: one-hot columns of the same source variable, bag-of-words counts,
indicator columns. EFB packs such mutually exclusive features into a single
*bundle* that occupies one column of the binned matrix, so histogram
construction costs O(#bundles) column scans instead of O(#features) while
losing nothing: within a bundle every original feature keeps its own
contiguous range of bin ids, so an original (feature, bin) pair is always
recoverable from the bundled bin.

This module holds the EFB core -- conflict measurement, plan construction,
encoding, decoding, and the bundled matrix -- and the dense CPU integration
that makes it reachable.

Where it is live, and how it avoids the model entirely
------------------------------------------------------
`enable_bundle` is off by default (`DEFAULT_ENABLE_BUNDLE`) and, when it is
set, it applies to the **dense CPU trainers in boosting.mojo** and to nothing
else. `BoosterParams.bundling` carries the switch, `prepare_bundling` fits the
plan once per training call, and `tree.grow_tree` takes the resulting
`BundledMatrix`.

The integration rests on one decision: **a bundle is a histogram layout and
never leaves the grower**. Concretely, inside `grow_tree`:

- histograms are accumulated over the *bundled* matrix, which is the whole
  point: O(#bundles) column scans instead of O(#features);
- each accumulated histogram is expanded straight back into the per-feature
  shape (`expand_bundled_histogram`), so split search, leaf values, and
  sibling subtraction read exactly what they have always read and need no
  knowledge of bundling at all;
- row partitioning reads the *original* matrix, because the split it routes
  by is expressed in original terms.

So a `Tree` grown with bundling is indistinguishable from one grown without
it: same feature ids, same threshold bins, same missing routing, same
categorical sets. `split.find_best_split`, `Model`, `Booster.predict_*`,
`serialize.mojo`, `importance.mojo`, and `contrib.mojo` therefore need **no
change at all**, no plan travels with the model, and there is no v4 format
bump. That is a deliberate trade: the plan is rebuilt from the training matrix
on every training call and is never used to score, which costs one extra dense
matrix, one pass over it, and one O(#features x #bins) expansion per node --
the same order as the sibling subtraction the grower already does -- and buys
a fit that no downstream consumer can tell apart.

The sparse path is a different question. `bundle_csc` and the sparse plan
below still exist for it, and a sparse integration that bundles the *stored*
matrix would face exactly the model-plumbing problem
`handoffs/task13_efb.md` sections 4, 6, and 10 describe (a `FeatureBundling`
on `Model`, a v4 section, the reader in `python/mojotrees/inspection.py`),
because the sparse predictors route rows by column. Nothing here does that
and nothing here enables it.

What the dense path does not do
-------------------------------
- `max_conflict_rate` above 0.0 is still refused (`check_bundling_params`).
  Lossy bundling drops training values, and the loss lands in the trees where
  no metric reports it; the plan counts collisions exactly
  (`FeatureBundling.is_lossless`), so this is a policy decision waiting on a
  benchmark, not a missing capability.
- Nothing enables bundling automatically. `EfbParams.min_reduction` and the
  `use_bundling` verdict decide only whether an *already requested* bundling
  is worth applying; when the verdict is negative the trainer falls back to
  the unbundled matrix, which is the same fit.
- The sparse, GPU, distributed, ranking, custom-objective, and custom-metric
  trainers do not honor the switch. They are not in this lane's ownership;
  `check_bundling_honored` is the one-line refusal they need so that an
  ignored setting is reported rather than silently dropped.

What "exclusive" means here
---------------------------
A feature's *default bin* is the bin holding the value 0.0
(`sparse.default_bins`); a row is **non-default** for feature f when its bin
differs from `default_bin[f]`. Two features conflict at a row when both are
non-default there. That is the only definition used, and it settles the three
cases the sparse layout distinguishes (see sparse.mojo):

- An **absent** entry is the value 0.0, so it is at the default bin and never
  conflicts. Absent rows are what makes bundling pay.
- An **explicitly stored zero**, or any stored value that happens to bin to
  the default bin, is treated identically to an absent entry: it is not a
  conflict, and `bundle_csc` drops it, because the bundle column recovers it
  from the bundle's default bin. Storing such an entry or leaving it out
  therefore produces a bit-identical plan and a bit-identical bundled matrix.
- A **missing** value (a stored `NaN`, binned into the feature's reserved
  missing bin) is *not* at the default bin, so it is non-default and it does
  conflict. Missingness is a real value here, not an absence. See
  "Missing values" below for why that makes bundling them opt-in.

Bin layout
----------
A bundle with a single member is **identity encoded**: its bundle bin *is*
the member's local bin, its default bin is the member's default bin, and its
category table and reserved missing bin carry over unchanged. A plan in which
every bundle is a singleton therefore reproduces the source matrix exactly,
apart from dropping stored entries that sat at their default bin.

A bundle with two or more members reserves bundle bin 0
(`EFB_SHARED_BIN`) for "every member is at its default", and gives member k
the half-open range `[slot_offset[k], slot_offset[k] + local_bins[k] - 1)`,
laid out in member order starting at 1. A member's local bin b encodes as

    b == default          ->  EFB_SHARED_BIN
    b <  default          ->  slot_offset[k] + b
    b >  default          ->  slot_offset[k] + b - 1

so the member contributes `local_bins[k] - 1` bundle bins and the bundle uses
`1 + sum(local_bins - 1)` in total. That total is capped at
`EfbParams.max_bundle_bins` (256 at the most), because the binned matrix
stores bins as `UInt8`. Note the same formula gives a singleton's bin count,
`local_bins[0]`, so the cap is uniform.

Collisions
----------
With `max_conflict_rate > 0` a bundle may hold rows where several members are
non-default, and one column cannot represent that. The rule is fixed and
deterministic: **the member appearing earliest in the bundle's member order
wins the row**, and the losing entries are dropped, i.e. they read back as
their feature's default bin. `collisions[b]` counts exactly those dropped
entries, so a plan states its own approximation error rather than hiding it.
`max_conflict_rate = 0.0`, the default, admits no collisions at all and the
bundling is then exactly lossless.

Splits stay per-feature
-----------------------
Bundling is a storage and histogram-traversal optimization, never a change of
hypothesis space. A threshold across two members' ranges would compare bin
ids of unrelated features and is meaningless. Split search must scan each
member's own range, and `unbundle_histogram` reconstructs a member's local
histogram from the bundle's: the member's own range is copied back, and its
default bin is recovered by subtracting that range from the node total, since
rows where the member is default sit either in bundle bin 0 or inside some
*other* member's range. That subtraction is the same trade the sparse
accumulator already makes for its zero bin.

Missing values
--------------
A member's reserved missing bin encodes like any other non-default bin, and
`missing_bins` reports the bundle bins that hold missing values. But a
multi-member bundle can then hold several of them, and the binned matrix
carries one `missing_bin` per column, while the split search learns one
`default_left` per node. On the **sparse** plan (`fit_bundles`), which is
built for a consumer that routes rows by bundle column, that is why
`EfbParams.bundle_missing` defaults to False and a feature reserving a
missing bin is left a singleton.

The **dense** plan (`fit_bundles_dense`) is free of that restriction and
ignores `bundle_missing`, because the dense integration never routes a row
by a bundle column: `grow_tree` partitions on the original matrix and reads
the original `BinnedMatrix.missing_bin` table, and the split search reads an
expanded per-feature histogram in which the missing bin sits at its original
index. A missing value is simply one more non-default bin in the bundle,
recovered exactly. The dense plan therefore records `slot_missing = -1` on
every slot, which says "this plan makes no claim about routing missing
values", and a dense caller must not read `missing_bins` off it.

Categorical features
--------------------
Categorical features are never bundled with anything: each becomes a
singleton bundle. Their bins index a fitted category table, and bin 0 is the
reserved unknown/missing bin that must route right (see categorical.mojo);
neither survives being offset into a shared range, and a set split spanning
two features' categories is not a thing the tree can express. Identity
encoding for singletons is what keeps their tables valid unchanged.

Determinism
-----------
Fitting is serial and order-fixed on purpose. Features are visited in
descending non-default count with ties broken toward the smaller feature
index (via the stable `_argsort`, the same trick `categorical.mojo` uses), a
candidate joins the first bundle in creation order that accepts it, and
bundles come out in creation order. The same matrix and parameters therefore
always give a bit-identical plan, on any machine and under any worker count.
The plan is small (O(n_features)) and fitting it is O(nnz log nnz) at worst,
so there is no reason to trade that determinism for parallelism.

Differences from LightGBM
-------------------------
- LightGBM caps a group's conflict count against a budget derived from
  `max_conflict_rate` and tracks it approximately across a group; the budget
  here is the same but the conflict count is exact, computed against the
  bundle's accumulated set of non-default rows.
- LightGBM's bundles may mix a feature's zero bin with its neighbours'
  because its "most frequent bin" is folded away; here the shared bin is
  defined by the default bin (the bin of 0.0), which is what the sparse path
  already means by absent.
- LightGBM will bundle categorical features into multi-value groups; this
  implementation does not, for the reason above.
"""

from .binning import BinMapper, BinnedMatrix
from .metrics import _argsort
from .categorical import CategoricalSpec
from .sparse import SparseBinnedMatrix

# The bundle bin meaning "every member of this bundle is at its default".
# Only multi-member bundles reserve it; a singleton is identity encoded.
comptime EFB_SHARED_BIN = 0

# "No such feature", "no such bin", "not bundled".
comptime EFB_NONE = -1

# Binned matrices store bins as UInt8, so no bundle can exceed 256 bins.
comptime EFB_MAX_BINS = 256

# Memory accounting: a stored entry costs one Int row index and one UInt8
# bin, and each column costs one Int offset plus one UInt8 default bin.
comptime _BYTES_PER_INDEX = 8
comptime _BYTES_PER_BIN = 1

# LightGBM defaults `enable_bundle` to true. mojotrees defaults it to false
# and, until the model plumbing named in the module docstring exists, accepts
# nothing else.
comptime DEFAULT_ENABLE_BUNDLE = False


def check_bundling_supported(enable_bundle: Bool, cpu: Bool = True) raises:
    """Accept a bundling request the dense CPU trainers can actually honor.

    `enable_bundle=false` is LightGBM's own default spelled out and is always
    accepted. `enable_bundle=true` is accepted for a CPU run, which is where
    `boosting.mojo` applies it; asking for it on another device is refused by
    name rather than ignored, because a silently unbundled fit is a correct
    model, just not the one that was asked for, and nothing in the metrics
    would show it.
    """
    if not enable_bundle:
        return
    if cpu:
        return
    raise Error(
        "'enable_bundle' is implemented for the dense CPU trainers only; the"
        " GPU trainer builds its histograms from the dense BinnedMatrix"
        " directly and never sees a bundled column. Set device=cpu, or leave"
        " enable_bundle at its default of false"
    )


def check_bundling_params(max_conflict_rate: Float64) raises:
    """Refuse a conflict rate that asks for lossy bundling.

    `max_conflict_rate` is the one bundling knob whose cost is invisible: a
    collision drops a value from the training matrix, so the model is an
    approximation of the unbundled one and no metric says so. The plan counts
    those drops exactly (`FeatureBundling.collisions`,
    `FeatureBundling.is_lossless`), so this is a policy decision waiting on a
    benchmark rather than a missing capability, and until that benchmark
    exists only LightGBM's own default of 0.0 is accepted.
    """
    if max_conflict_rate == 0.0:
        return
    raise Error(
        "'max_conflict_rate' above 0.0 trades exactness for columns: a"
        " collision drops a training value, and the loss lands in the trees"
        " where no metric reports it. Bundling itself is available"
        " ('enable_bundle'); only the lossy mode is withheld pending a"
        " benchmark. See handoffs/connect_09_algorithms.md"
    )




@fieldwise_init
struct EfbParams(Copyable, Movable):
    """Bundling knobs.

    - `max_conflict_rate`: fraction of rows a bundle may hold collisions on,
      LightGBM's parameter of that name. 0.0 (the default) makes bundling
      exactly lossless.
    - `max_bundle_bins`: bins one bundle may use, capped at `EFB_MAX_BINS`
      by the `UInt8` bin storage.
    - `max_bundle_size`: members per bundle; 0 means unlimited.
    - `max_nondefault_rate`: a feature non-default on a larger fraction of
      rows than this is too dense to bundle usefully and is left a singleton.
      LightGBM applies the same filter at 0.95.
    - `min_reduction`: fraction of the dense histogram footprint the plan
      must remove before `use_bundling` is True (see `FeatureBundling`).
    - `bundle_missing`: allow features reserving a missing bin to join a
      multi-member bundle. Off by default; see the module docstring.
    """

    var max_conflict_rate: Float64
    var max_bundle_bins: Int
    var max_bundle_size: Int
    var max_nondefault_rate: Float64
    var min_reduction: Float64
    var bundle_missing: Bool

    @staticmethod
    def default() -> EfbParams:
        return EfbParams(0.0, EFB_MAX_BINS, 0, 0.95, 0.0, False)

    def check(self) raises:
        if self.max_conflict_rate < 0.0 or self.max_conflict_rate > 1.0:
            raise Error("max_conflict_rate must be in [0, 1]")
        if self.max_bundle_bins < 2 or self.max_bundle_bins > EFB_MAX_BINS:
            raise Error("max_bundle_bins must be in [2, 256]")
        if self.max_bundle_size < 0:
            raise Error("max_bundle_size must be non-negative")
        if self.max_nondefault_rate <= 0.0 or self.max_nondefault_rate > 1.0:
            raise Error("max_nondefault_rate must be in (0, 1]")
        if self.min_reduction < 0.0 or self.min_reduction >= 1.0:
            raise Error("min_reduction must be in [0, 1)")


struct EfbSettings(Copyable, Movable):
    """LightGBM's `enable_bundle` plus the knobs it governs, as one field a
    trainer can carry.

    `enabled` is the master switch and defaults to False
    (`DEFAULT_ENABLE_BUNDLE`), unlike LightGBM, which defaults it to true; see
    the module docstring for why nothing here turns bundling on by itself.
    `params` is the plan-construction policy and is inert while `enabled` is
    False, which is what keeps a default configuration on exactly the path it
    took before bundling existed.

    Held apart from `EfbParams` rather than added to it so that every
    positional `EfbParams(...)` caller keeps working unchanged.
    """

    var enabled: Bool
    var params: EfbParams

    def __init__(
        out self,
        enabled: Bool = DEFAULT_ENABLE_BUNDLE,
        var params: EfbParams = EfbParams.default(),
    ):
        self.enabled = enabled
        self.params = params^

    @staticmethod
    def disabled() -> EfbSettings:
        """Bundling off, which is the default everywhere."""
        return EfbSettings(False, EfbParams.default())

    def check(self) raises:
        """Range-check the knobs, and refuse the lossy mode.

        Runs whether or not `enabled` is set, so a bad value in a parameter
        string is named before any data is read rather than at the first
        training call that happens to turn bundling on.
        """
        self.params.check()
        check_bundling_params(self.params.max_conflict_rate)


def check_bundling_honored(settings: EfbSettings, trainer: String) raises:
    """Refuse an active bundling request in a trainer that does not apply it.

    `boosting.mojo`'s dense trainers honor `BoosterParams.bundling`; the
    sparse, GPU, distributed, ranking, custom-objective, and custom-metric
    trainers do not. Each of those should call this on entry so that a caller
    who set the switch is told which trainer dropped it, rather than getting
    an unbundled fit that looks exactly like a bundled one.
    """
    if not settings.enabled:
        return
    raise Error(
        "'enable_bundle' is applied by the dense CPU trainers in"
        " boosting.mojo (train, train_more, train_with_valid,"
        " train_multiclass*); ",
        trainer,
        " builds its histograms another way and would ignore it. Leave"
        " enable_bundle at its default of false for this trainer",
    )


@fieldwise_init
struct LocalHistogram(Copyable, Movable):
    """One original feature's per-bin statistics, recovered from a bundle's
    histogram by `unbundle_histogram`. Indexed by the feature's own local bin
    ids, so split search reads it exactly as it reads an unbundled column."""

    var grad: List[Float64]
    var hess: List[Float64]
    var count: List[Int]


struct FeatureBundling(Copyable, Movable):
    """A fitted bundling plan: which features share a column, and how their
    bins map into it.

    Bundle b owns member slots `[bundle_start[b], bundle_start[b + 1])`.
    Slot k describes one original feature: `members[k]` is its index,
    `slot_bins[k]` its local bin count, `slot_default[k]` its local default
    bin, `slot_missing[k]` its local missing bin (or -1), and
    `slot_offset[k]` where its non-default bins begin inside the bundle
    (0 for the identity-encoded singleton case).

    `bundle_of[f]` and `slot_of[f]` invert that for a feature. `bundle_bins[b]`
    is how many bins bundle b uses and `collisions[b]` how many stored entries
    the bundling drops in it (see the module docstring).

    The plan is a function of the training matrix and the parameters alone,
    so it can be applied unchanged to any later matrix binned by the same
    mapper: `bundle_csc` reads only the plan.
    """

    var members: List[Int]
    var bundle_start: List[Int]
    var slot_offset: List[Int]
    var slot_bins: List[Int]
    var slot_default: List[Int]
    var slot_missing: List[Int]
    var bundle_of: List[Int]
    var slot_of: List[Int]
    var bundle_bins: List[Int]
    var collisions: List[Int]
    var n_features: Int
    var n_rows: Int
    var n_bins: Int
    var source_entries: Int
    var bundled_entries: Int
    var use_bundling: Bool

    def __init__(
        out self,
        var members: List[Int],
        var bundle_start: List[Int],
        var slot_offset: List[Int],
        var slot_bins: List[Int],
        var slot_default: List[Int],
        var slot_missing: List[Int],
        var bundle_of: List[Int],
        var slot_of: List[Int],
        var bundle_bins: List[Int],
        var collisions: List[Int],
        n_features: Int,
        n_rows: Int,
        n_bins: Int,
        source_entries: Int,
        bundled_entries: Int,
        use_bundling: Bool,
    ):
        self.members = members^
        self.bundle_start = bundle_start^
        self.slot_offset = slot_offset^
        self.slot_bins = slot_bins^
        self.slot_default = slot_default^
        self.slot_missing = slot_missing^
        self.bundle_of = bundle_of^
        self.slot_of = slot_of^
        self.bundle_bins = bundle_bins^
        self.collisions = collisions^
        self.n_features = n_features
        self.n_rows = n_rows
        self.n_bins = n_bins
        self.source_entries = source_entries
        self.bundled_entries = bundled_entries
        self.use_bundling = use_bundling

    @staticmethod
    def none() -> FeatureBundling:
        """The empty plan: no bundles, no features, bundling off.

        This is the "not bundled" value, so a consumer needs no `Optional`:
        `active()` is False for it and every caller tests that one predicate.
        It is deliberately not a valid plan in `validate`'s sense (it covers
        no features), and nothing ever validates it, because nothing ever
        encodes or decodes through it.
        """
        return FeatureBundling(
            List[Int](),
            List[Int](),
            List[Int](),
            List[Int](),
            List[Int](),
            List[Int](),
            List[Int](),
            List[Int](),
            List[Int](),
            List[Int](),
            0,
            0,
            0,
            0,
            0,
            False,
        )

    def active(self) -> Bool:
        """Whether this plan describes a layout at all. False for `none()`,
        which is the value a caller that is not bundling passes."""
        return len(self.bundle_bins) > 0

    def n_bundles(self) -> Int:
        return len(self.bundle_bins)

    def bundle_size(self, bundle: Int) -> Int:
        return self.bundle_start[bundle + 1] - self.bundle_start[bundle]

    def member_at(self, bundle: Int, k: Int) -> Int:
        """The k-th member feature of `bundle`, in member order."""
        return self.members[self.bundle_start[bundle] + k]

    def is_bundled(self, feature: Int) -> Bool:
        """Whether `feature` actually shares its column with another."""
        return self.bundle_size(self.bundle_of[feature]) > 1

    def max_bundle_bins(self) -> Int:
        """Bins the widest bundle uses; the bundled matrix's `n_bins`."""
        var out = 0
        for b in range(len(self.bundle_bins)):
            if self.bundle_bins[b] > out:
                out = self.bundle_bins[b]
        return out

    def total_collisions(self) -> Int:
        var out = 0
        for b in range(len(self.collisions)):
            out += self.collisions[b]
        return out

    def is_lossless(self) -> Bool:
        """Whether the bundling drops no stored entry, so every original
        (feature, bin) is recoverable for every row."""
        return self.total_collisions() == 0

    def encode(self, feature: Int, local_bin: Int) raises -> Int:
        """The bundle bin a local bin of `feature` maps to.

        A multi-member bundle sends the member's default bin to
        `EFB_SHARED_BIN`; a singleton is identity encoded.
        """
        if feature < 0 or feature >= self.n_features:
            raise Error("feature index out of range")
        var s = self.slot_of[feature]
        if s < 0:
            raise Error("feature is not in the plan")
        if local_bin < 0 or local_bin >= self.slot_bins[s]:
            raise Error("local bin out of range for this feature")
        if self.bundle_size(self.bundle_of[feature]) == 1:
            return local_bin
        var d = self.slot_default[s]
        if local_bin == d:
            return EFB_SHARED_BIN
        if local_bin < d:
            return self.slot_offset[s] + local_bin
        return self.slot_offset[s] + local_bin - 1

    def slot_containing(self, bundle: Int, bundle_bin: Int) raises -> Int:
        """The member slot owning `bundle_bin`, or `EFB_NONE` for a
        multi-member bundle's shared bin."""
        if bundle < 0 or bundle >= self.n_bundles():
            raise Error("bundle index out of range")
        if bundle_bin < 0 or bundle_bin >= self.bundle_bins[bundle]:
            raise Error("bundle bin out of range")
        var lo = self.bundle_start[bundle]
        var hi = self.bundle_start[bundle + 1]
        if hi - lo == 1:
            return lo
        if bundle_bin == EFB_SHARED_BIN:
            return EFB_NONE
        for k in range(lo, hi):
            var start = self.slot_offset[k]
            if bundle_bin >= start and bundle_bin < start + self.slot_bins[
                k
            ] - 1:
                return k
        # Unreachable for a validated plan: the members' ranges tile
        # [1, bundle_bins) exactly.
        raise Error("bundle bin belongs to no member")

    def decode_feature(self, bundle: Int, bundle_bin: Int) raises -> Int:
        """The original feature a bundle bin came from, or `EFB_NONE` for a
        multi-member bundle's shared bin, which belongs to every member at
        once."""
        var s = self.slot_containing(bundle, bundle_bin)
        if s == EFB_NONE:
            return EFB_NONE
        return self.members[s]

    def charged_feature(
        self, feature: Int, bin: Int, is_categorical: Bool = False
    ) raises -> Int:
        """The dataset feature a split found in the search space is charged
        to, for a per-feature cost keyed by dataset feature id.

        `cegb.CegbConfig`'s penalty vectors and `cegb.CegbLedger` are indexed
        that way, and the split search does not always see that id:

        - Without bundling, and for every categorical feature (categorical
          splits are searched as category partitions over one column), the
          search space *is* the dataset, so this is the identity.
        - With bundling on, a scanned "feature" is a bundle and a threshold
          bin belongs to one member of it. `decode_feature` recovers the
          member, and the member is what actually gets read at prediction
          time, so the member is what a per-feature cost is charged to.
          Charging the bundle would make one sparse feature's first use pay
          for every feature bundled with it.

        A multi-member bundle's shared bin (`EFB_SHARED_BIN`) belongs to
        every member at once, so a split whose threshold is that bin cannot be
        attributed to one feature; that is refused rather than charged to an
        arbitrary member. A validated plan never puts a threshold there -- the
        shared bin is where every member's default value lands -- so this is a
        guard on an inconsistent plan, not a case a caller has to handle.

        This lives here rather than in `cegb.mojo` because it is a pure
        bundling query and because `cegb.mojo` deliberately imports nothing
        from this package: `tree_parameters_extra` imports `cegb`, and this
        module reaches `tree_parameters_extra` through `binning`, so the
        import would close a cycle.
        """
        if not self.use_bundling:
            return feature
        if is_categorical:
            raise Error(
                "a categorical split inside a feature bundle cannot be"
                " charged to one dataset feature; categorical features are"
                " not bundled, so this plan and this split disagree"
            )
        if feature < 0 or feature >= self.n_bundles():
            raise Error(
                "bundle ",
                feature,
                " is out of range for ",
                self.n_bundles(),
                " bundles",
            )
        if self.bundle_size(feature) == 1:
            return self.member_at(feature, 0)
        var member = self.decode_feature(feature, bin)
        if member == EFB_NONE:
            raise Error(
                "a split on bundle ",
                feature,
                " at its shared bin belongs to every member of the bundle, so"
                " it cannot be charged to one feature",
            )
        return member

    def decode_bin(self, bundle: Int, bundle_bin: Int) raises -> Int:
        """The original local bin a bundle bin came from, or `EFB_NONE` for
        the shared bin."""
        var s = self.slot_containing(bundle, bundle_bin)
        if s == EFB_NONE:
            return EFB_NONE
        if self.bundle_size(bundle) == 1:
            return bundle_bin
        var local = bundle_bin - self.slot_offset[s]
        if local >= self.slot_default[s]:
            local += 1
        return local

    def missing_bins(self, bundle: Int) raises -> List[Int]:
        """Bundle bins of `bundle` that hold missing values, ascending by
        member order. Empty when no member reserves a missing bin."""
        if bundle < 0 or bundle >= self.n_bundles():
            raise Error("bundle index out of range")
        var out = List[Int]()
        for k in range(
            self.bundle_start[bundle], self.bundle_start[bundle + 1]
        ):
            if self.slot_missing[k] >= 0:
                out.append(self.encode(self.members[k], self.slot_missing[k]))
        return out^

    def bundle_default_bin(self, bundle: Int) -> Int:
        """The bundle's own default bin: the shared bin for a multi-member
        bundle, the member's own default bin for a singleton."""
        var lo = self.bundle_start[bundle]
        if self.bundle_start[bundle + 1] - lo == 1:
            return self.slot_default[lo]
        return EFB_SHARED_BIN

    def binned_bytes_unbundled(self) -> Int:
        """Bytes the source binned CSC matrix occupies."""
        return (
            self.source_entries * (_BYTES_PER_INDEX + _BYTES_PER_BIN)
            + (self.n_features + 1) * _BYTES_PER_INDEX
            + self.n_features * _BYTES_PER_BIN
        )

    def binned_bytes_bundled(self) -> Int:
        """Bytes the bundled binned CSC matrix occupies."""
        var n = self.n_bundles()
        return (
            self.bundled_entries * (_BYTES_PER_INDEX + _BYTES_PER_BIN)
            + (n + 1) * _BYTES_PER_INDEX
            + n * _BYTES_PER_BIN
        )

    def histogram_slots_unbundled(self) -> Int:
        """(feature, bin) slots a dense histogram over the source matrix
        needs. The histogram is rectangular, so every feature costs the full
        `n_bins` whether it uses them or not."""
        return self.n_features * self.n_bins

    def histogram_slots_bundled(self) -> Int:
        """(bundle, bin) slots a dense histogram over the bundled matrix
        needs, padded the same rectangular way."""
        return self.n_bundles() * self.max_bundle_bins()

    def validate(self) raises:
        """Structural check of the plan: every feature placed exactly once,
        every member range inside its bundle, and the ranges tiling the
        bundle's bins without gaps or overlap."""
        var n = self.n_bundles()
        if len(self.bundle_start) != n + 1:
            raise Error("bundle_start must have length n_bundles + 1")
        if self.bundle_start[0] != 0:
            raise Error("bundle_start must start at 0")
        if self.bundle_start[n] != len(self.members):
            raise Error("bundle_start must end at the member count")
        if len(self.slot_offset) != len(self.members):
            raise Error("slot arrays must be per member")
        if (
            len(self.slot_bins) != len(self.members)
            or len(self.slot_default) != len(self.members)
            or len(self.slot_missing) != len(self.members)
        ):
            raise Error("slot arrays must be per member")
        if (
            len(self.bundle_of) != self.n_features
            or len(self.slot_of) != self.n_features
        ):
            raise Error("bundle_of and slot_of must be per feature")
        if len(self.collisions) != n:
            raise Error("collisions must be per bundle")

        var seen = List[Bool](capacity=self.n_features)
        seen.resize(self.n_features, False)
        for b in range(n):
            var lo = self.bundle_start[b]
            var hi = self.bundle_start[b + 1]
            if hi <= lo:
                raise Error("a bundle must have at least one member")
            if self.bundle_bins[b] < 1 or self.bundle_bins[b] > EFB_MAX_BINS:
                raise Error("bundle bin count out of range")
            var expected = 1 if hi - lo > 1 else 0
            var total = 1
            for k in range(lo, hi):
                var f = self.members[k]
                if f < 0 or f >= self.n_features:
                    raise Error("member feature index out of range")
                if seen[f]:
                    raise Error("feature appears in more than one bundle")
                seen[f] = True
                if self.bundle_of[f] != b or self.slot_of[f] != k:
                    raise Error("bundle_of/slot_of disagree with the members")
                if self.slot_bins[k] < 1:
                    raise Error("a member must have at least one bin")
                if (
                    self.slot_default[k] < 0
                    or self.slot_default[k] >= self.slot_bins[k]
                ):
                    raise Error("member default bin out of range")
                if self.slot_missing[k] >= self.slot_bins[k]:
                    raise Error("member missing bin out of range")
                if self.slot_missing[k] >= 0 and (
                    self.slot_missing[k] == self.slot_default[k]
                ):
                    raise Error("a missing bin cannot be the default bin")
                if hi - lo == 1:
                    if self.slot_offset[k] != 0:
                        raise Error("a singleton bundle is identity encoded")
                    total = self.slot_bins[k]
                else:
                    if self.slot_offset[k] != expected:
                        raise Error("member ranges must tile the bundle")
                    expected += self.slot_bins[k] - 1
                    total = expected
            if total != self.bundle_bins[b]:
                raise Error("bundle bin count disagrees with its members")
            if self.collisions[b] < 0:
                raise Error("collision count must be non-negative")
        for f in range(self.n_features):
            if not seen[f]:
                raise Error("feature is in no bundle")


def feature_bin_count(mapper: BinMapper, feature: Int) raises -> Int:
    """How many bins `feature` can take under `mapper`.

    Read from the fitted mapper rather than counted off the data, because a
    bin a training column happened not to use is still a bin an unseen row
    can land in, and a bundle's ranges have to leave room for it.
    """
    if feature < 0 or feature >= mapper.n_features:
        raise Error("feature index out of range")
    if mapper.cats.is_cat(feature):
        # Kept categories map to bins 1..k, plus the reserved unknown bin 0.
        return mapper.cats.n_categories(feature) + 1
    var n = (
        mapper.edge_offsets[feature + 1] - mapper.edge_offsets[feature] + 1
    )
    var mb = mapper.missing_bin[feature]
    if mb >= 0 and mb + 1 > n:
        n = mb + 1
    return n


def nondefault_rows(data: SparseBinnedMatrix, feature: Int) raises -> List[Int]:
    """Ascending rows where `feature` is not at its default bin.

    Stored entries that sit at the default bin are skipped, which is what
    makes an explicitly stored zero and an absent entry indistinguishable
    here.
    """
    if feature < 0 or feature >= data.n_features:
        raise Error("feature index out of range")
    var d = data.default_bin[feature]
    var out = List[Int]()
    for i in range(data.col_offsets[feature], data.col_offsets[feature + 1]):
        if data.bin[i] != d:
            out.append(data.row_index[i])
    return out^


def conflict_count(a: List[Int], b: List[Int]) -> Int:
    """Rows present in both ascending row lists. O(len(a) + len(b))."""
    var i = 0
    var j = 0
    var out = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def pairwise_conflict(
    data: SparseBinnedMatrix, f: Int, g: Int
) raises -> Int:
    """Rows where features `f` and `g` are both non-default."""
    return conflict_count(nondefault_rows(data, f), nondefault_rows(data, g))


def _count_in(rows: List[Int], sorted_union: List[Int]) -> Int:
    """How many of `rows` are already in `sorted_union`, by binary search, so
    the cost follows the candidate rather than the accumulated bundle."""
    var out = 0
    for i in range(len(rows)):
        var lo = 0
        var hi = len(sorted_union)
        while lo < hi:
            var mid = (lo + hi) // 2
            if sorted_union[mid] < rows[i]:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(sorted_union) and sorted_union[lo] == rows[i]:
            out += 1
    return out


def _merge_rows(a: List[Int], b: List[Int]) -> List[Int]:
    """Ascending union of two ascending row lists."""
    var out = List[Int](capacity=len(a) + len(b))
    var i = 0
    var j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    while i < len(a):
        out.append(a[i])
        i += 1
    while j < len(b):
        out.append(b[j])
        j += 1
    return out^


def _fit_bundles_core(
    var rows: List[List[Int]],
    counts: List[Int],
    local_bins: List[Int],
    defaults: List[Int],
    missings: List[Int],
    eligible: List[Bool],
    n_features: Int,
    n_rows: Int,
    n_bins: Int,
    source_entries: Int,
    params: EfbParams,
) raises -> FeatureBundling:
    """The greedy itself, shared by the sparse and the dense entry points.

    Every caller describes each feature the same way -- its non-default rows,
    how many there are, how many local bins it has, its default and missing
    bins, and whether it may be bundled at all -- and this decides the
    layout. Keeping one copy is what makes a sparse plan and a dense plan
    packed by identical rules, so a future sparse integration cannot drift
    from the dense one.

    `counts[f]` rather than `len(rows[f])` is the non-default count, because a
    caller may decline to materialize the row list of a feature it has
    already ruled ineligible: on a dense matrix an ineligible column's list
    can be nearly `n_rows` long and is never read, since an ineligible
    feature becomes a closed singleton and is never a merge candidate. For a
    caller that materializes everything the two are equal and this changes
    nothing.

    Greedy, serial, and order-fixed (see "Determinism" in the module
    docstring). Every feature ends in exactly one bundle: a feature that
    cannot or should not be bundled becomes a singleton rather than being
    dropped, so the plan always covers the whole matrix and a caller can
    apply it without carrying a second, unbundled matrix alongside.

    The returned plan reports `use_bundling`; it does not act on it.
    """
    var max_conflicts = Int(Float64(n_rows) * params.max_conflict_rate)

    var neg_counts = List[Float64](capacity=n_features)
    for f in range(n_features):
        neg_counts.append(-Float64(counts[f]))

    # Stable ascending sort of the negated counts is a descending sort of the
    # counts with ties broken toward the smaller feature index.
    var order = _argsort(neg_counts)

    var b_members = List[List[Int]]()
    var b_union = List[List[Int]]()
    var b_entries = List[Int]()
    var b_bins = List[Int]()
    var b_conflict = List[Int]()
    var b_open = List[Bool]()

    for oi in range(len(order)):
        var f = order[oi]
        var width = local_bins[f] - 1
        if not eligible[f]:
            var only = List[Int](capacity=1)
            only.append(f)
            b_members.append(only^)
            b_union.append(rows[f].copy())
            b_entries.append(counts[f])
            b_bins.append(width)
            b_conflict.append(0)
            b_open.append(False)
            continue
        var placed = False
        for b in range(len(b_members)):
            if not b_open[b]:
                continue
            if (
                params.max_bundle_size > 0
                and len(b_members[b]) >= params.max_bundle_size
            ):
                continue
            if 1 + b_bins[b] + width > params.max_bundle_bins:
                continue
            var conflict = _count_in(rows[f], b_union[b])
            if b_conflict[b] + conflict > max_conflicts:
                continue
            b_members[b].append(f)
            b_union[b] = _merge_rows(b_union[b], rows[f])
            b_entries[b] = len(b_union[b])
            b_bins[b] += width
            b_conflict[b] += conflict
            placed = True
            break
        if not placed:
            var only = List[Int](capacity=1)
            only.append(f)
            b_members.append(only^)
            b_union.append(rows[f].copy())
            b_entries.append(counts[f])
            b_bins.append(width)
            b_conflict.append(0)
            b_open.append(True)

    var n_bundles = len(b_members)
    var members = List[Int](capacity=n_features)
    var bundle_start = List[Int](capacity=n_bundles + 1)
    var slot_offset = List[Int](capacity=n_features)
    var slot_bins = List[Int](capacity=n_features)
    var slot_default = List[Int](capacity=n_features)
    var slot_missing = List[Int](capacity=n_features)
    var bundle_of = List[Int](capacity=n_features)
    bundle_of.resize(n_features, EFB_NONE)
    var slot_of = List[Int](capacity=n_features)
    slot_of.resize(n_features, EFB_NONE)
    var bundle_bins = List[Int](capacity=n_bundles)
    var collisions = List[Int](capacity=n_bundles)
    var bundled_entries = 0

    bundle_start.append(0)
    for b in range(n_bundles):
        var size = len(b_members[b])
        var offset = 1
        for k in range(size):
            var f = b_members[b][k]
            bundle_of[f] = b
            slot_of[f] = len(members)
            members.append(f)
            slot_bins.append(local_bins[f])
            slot_default.append(defaults[f])
            slot_missing.append(missings[f])
            if size == 1:
                slot_offset.append(0)
            else:
                slot_offset.append(offset)
                offset += local_bins[f] - 1
        bundle_start.append(len(members))
        bundle_bins.append(1 + b_bins[b])
        collisions.append(b_conflict[b])
        bundled_entries += b_entries[b]

    var plan = FeatureBundling(
        members^,
        bundle_start^,
        slot_offset^,
        slot_bins^,
        slot_default^,
        slot_missing^,
        bundle_of^,
        slot_of^,
        bundle_bins^,
        collisions^,
        n_features,
        n_rows,
        n_bins,
        source_entries,
        bundled_entries,
        False,
    )
    plan.validate()
    # The fallback decision. Bundling has to buy back both the column count
    # it saves in traversal and the rectangular histogram it widens: a plan
    # that packs a few narrow features into one very wide bundle can cost
    # more histogram slots than it saves columns.
    var before = plan.histogram_slots_unbundled()
    var after = plan.histogram_slots_bundled()
    plan.use_bundling = (
        n_bundles < n_features
        and Float64(after) <= Float64(before) * (1.0 - params.min_reduction)
    )
    return plan^


def fit_bundles(
    mapper: BinMapper,
    data: SparseBinnedMatrix,
    params: EfbParams = EfbParams.default(),
) raises -> FeatureBundling:
    """Build a bundling plan for a binned sparse matrix.

    Bin widths come from the fitted mapper rather than from the data, because
    a bin a training column happened not to use is still a bin an unseen row
    can land in, and a sparse plan is built for a consumer that encodes rows
    through it at scoring time. The dense entry point below has a different
    answer for that, and says why.
    """
    params.check()
    if mapper.n_features != data.n_features:
        raise Error("mapper and matrix must agree on n_features")
    if data.n_rows < 1 or data.n_features < 1:
        raise Error("matrix must have positive dimensions")

    var n_features = data.n_features
    var n_rows = data.n_rows
    var dense_cut = Int(Float64(n_rows) * params.max_nondefault_rate)

    var rows = List[List[Int]](capacity=n_features)
    var counts = List[Int](capacity=n_features)
    var local_bins = List[Int](capacity=n_features)
    var defaults = List[Int](capacity=n_features)
    var missings = List[Int](capacity=n_features)
    var eligible = List[Bool](capacity=n_features)

    for f in range(n_features):
        var nd = nondefault_rows(data, f)
        var n_local = feature_bin_count(mapper, f)
        var d = Int(data.default_bin[f])
        if d >= n_local:
            raise Error(
                "matrix default bin is outside the mapper's bin count; the"
                " matrix was binned by a different mapper"
            )
        var mb = data.missing_bin[f]
        if mb >= n_local:
            raise Error("matrix missing bin is outside the mapper's bin count")
        # A feature is bundleable when it is numerical, not too dense, not
        # holding missing values the consumer cannot yet route, and narrow
        # enough to fit a bundle at all.
        var ok = not mapper.cats.is_cat(f)
        if mb >= 0 and not params.bundle_missing:
            ok = False
        if len(nd) > dense_cut:
            ok = False
        if n_local > params.max_bundle_bins:
            ok = False
        counts.append(len(nd))
        rows.append(nd^)
        local_bins.append(n_local)
        defaults.append(d)
        missings.append(mb)
        eligible.append(ok)

    return _fit_bundles_core(
        rows^,
        counts,
        local_bins,
        defaults,
        missings,
        eligible,
        n_features,
        n_rows,
        data.n_bins,
        data.nnz(),
        params,
    )


# ---------------------------------------------------------------------------
# The dense CPU path
# ---------------------------------------------------------------------------
#
# A dense `BinnedMatrix` has no notion of an absent entry, so "the bin holding
# 0.0" is not the right definition of a feature's resting value here and the
# mapper that would compute it is a layer above the trainer. The dense path
# uses **the column's most frequent bin** instead. On the data EFB is for --
# one-hot columns, indicator columns, bag-of-words counts, anything quantile
# binned whose mass sits on one value -- that bin *is* the bin of 0.0, so the
# two definitions agree exactly; where they differ, the most frequent bin is
# the better choice, because it is by construction the bin whose rows are free
# to share the bundle, which is the quantity that decides whether bundling
# pays. LightGBM folds away a feature's most frequent bin for the same reason.
#
# Widths are counted off the training matrix rather than read from a mapper,
# which is safe here and only here: the dense plan never encodes a row that
# was not part of the fit. It is used to lay out histograms during growth and
# is discarded when the last tree is grown, so there is no unseen row for a
# too-narrow range to lose. See `dense_bin_counts` for what that costs and
# what it does not.


def dense_bin_counts(data: BinnedMatrix) raises -> List[Int]:
    """How many local bins each column of a dense binned matrix uses.

    One more than the highest bin the column actually holds, raised to cover
    the feature's reserved missing bin and, for a categorical feature, its
    whole fitted category table, so a category that no training row carries
    still gets its slot.

    Counted off the data, not off a mapper. That is sound for the dense
    integration and unsound in general: it is correct because the plan is used
    only to lay out histograms over *this* matrix, and a bin no training row
    occupies contributes nothing to any histogram. It would be wrong for a
    plan that has to encode an unseen row, which is why `fit_bundles` reads
    the mapper instead.

    Nothing downstream loses a candidate by it. `expand_bundled_histogram`
    writes each member back into the full `n_bins`-wide slice the split search
    reads, zeroing the bins past `n_local` -- which is what an unbundled
    histogram holds for them too, since no training row occupies them. The
    scan therefore sees the same candidates it always saw.
    """
    if data.n_features < 1 or data.n_rows < 1:
        raise Error("matrix must have positive dimensions")
    var out = List[Int](capacity=data.n_features)
    for f in range(data.n_features):
        var top = 0
        var base = f * data.n_rows
        for r in range(data.n_rows):
            var b = Int(data.bins[base + r])
            if b > top:
                top = b
        if data.missing_bin[f] > top:
            top = data.missing_bin[f]
        if data.cats.is_cat(f) and data.cats.n_categories(f) > top:
            top = data.cats.n_categories(f)
        out.append(top + 1)
    return out^


def dense_default_bins(data: BinnedMatrix) raises -> List[Int]:
    """Each column's most frequent bin, ties broken toward the smaller bin id.

    This is the dense path's "at rest" bin, the one a bundle's shared bin
    stands in for; see the section comment above for why it is the most
    frequent bin and not the bin of 0.0. Ties are broken deterministically so
    that the same matrix always yields the same plan.
    """
    if data.n_features < 1 or data.n_rows < 1:
        raise Error("matrix must have positive dimensions")
    var out = List[Int](capacity=data.n_features)
    var tally = List[Int](capacity=EFB_MAX_BINS)
    tally.resize(EFB_MAX_BINS, 0)
    for f in range(data.n_features):
        for b in range(EFB_MAX_BINS):
            tally[b] = 0
        var base = f * data.n_rows
        for r in range(data.n_rows):
            tally[Int(data.bins[base + r])] += 1
        var best = 0
        for b in range(1, EFB_MAX_BINS):
            if tally[b] > tally[best]:
                best = b
        out.append(best)
    return out^


def nondefault_rows_dense(
    data: BinnedMatrix, feature: Int, default_bin: Int
) raises -> List[Int]:
    """Ascending rows where a dense column is not at `default_bin`.

    The dense counterpart of `nondefault_rows`. A missing value is one of
    these: it sits in its own reserved bin, which is a value the bundle has to
    keep, not an absence.
    """
    if feature < 0 or feature >= data.n_features:
        raise Error("feature index out of range")
    var out = List[Int]()
    var base = feature * data.n_rows
    for r in range(data.n_rows):
        if Int(data.bins[base + r]) != default_bin:
            out.append(r)
    return out^


def count_nondefault_dense(
    data: BinnedMatrix, feature: Int, default_bin: Int
) raises -> Int:
    """How many rows of a dense column are not at `default_bin`.

    Counted without materializing the row list, so a column that turns out to
    be too dense to bundle costs a pass rather than an allocation the size of
    the column. On a wide dense matrix that is the difference between a plan
    fit and an out-of-memory one.
    """
    if feature < 0 or feature >= data.n_features:
        raise Error("feature index out of range")
    var out = 0
    var base = feature * data.n_rows
    for r in range(data.n_rows):
        if Int(data.bins[base + r]) != default_bin:
            out += 1
    return out


def fit_bundles_dense(
    data: BinnedMatrix, params: EfbParams = EfbParams.default()
) raises -> FeatureBundling:
    """Build a bundling plan for a dense binned matrix.

    Same greedy, same determinism, and the same `use_bundling` verdict as the
    sparse entry point; the differences are the three the section comment
    above states, plus one more:

    - `EfbParams.bundle_missing` is ignored. A feature reserving a missing bin
      bundles like any other, because the dense integration never routes a row
      by a bundle column (see the module docstring). `slot_missing` is left at
      -1 on every slot to say so.

    The plan reports `use_bundling`; it does not act on it. `prepare_bundling`
    is what turns the verdict into a decision.
    """
    params.check()
    if data.n_rows < 1 or data.n_features < 1:
        raise Error("matrix must have positive dimensions")

    var n_features = data.n_features
    var n_rows = data.n_rows
    var dense_cut = Int(Float64(n_rows) * params.max_nondefault_rate)
    var defaults = dense_default_bins(data)
    var local_bins = dense_bin_counts(data)

    var rows = List[List[Int]](capacity=n_features)
    var counts = List[Int](capacity=n_features)
    var missings = List[Int](capacity=n_features)
    var eligible = List[Bool](capacity=n_features)
    var source_entries = 0

    for f in range(n_features):
        var n_nd = count_nondefault_dense(data, f, defaults[f])
        source_entries += n_nd
        # A feature is bundleable when it is numerical, not too dense, and
        # narrow enough to fit a bundle at all. Missingness is not a bar here.
        var ok = not data.cats.is_cat(f)
        if n_nd > dense_cut:
            ok = False
        if local_bins[f] > params.max_bundle_bins:
            ok = False
        # Only an eligible feature's rows are materialized: an ineligible one
        # becomes a closed singleton, which the greedy never reads rows from,
        # and on a dense matrix that list is the expensive thing.
        if ok:
            rows.append(nondefault_rows_dense(data, f, defaults[f]))
        else:
            rows.append(List[Int]())
        counts.append(n_nd)
        missings.append(EFB_NONE)
        eligible.append(ok)

    return _fit_bundles_core(
        rows^,
        counts,
        local_bins,
        defaults,
        missings,
        eligible,
        n_features,
        n_rows,
        data.n_bins,
        source_entries,
        params,
    )


def bundle_dense(
    data: BinnedMatrix, plan: FeatureBundling
) raises -> BinnedMatrix:
    """Apply a dense plan, producing the bundled binned matrix.

    One column per bundle, column-major like its source. A row at a member's
    default bin is left at the bundle's default bin, which is what recovers
    it; on a collision the earliest member in the bundle's member order keeps
    the row, exactly as `bundle_csc` resolves it, so the two applications of
    one plan agree.

    A singleton bundle is identity encoded, so its category table and its
    reserved missing bin carry over unchanged and a plan of all singletons
    reproduces the source matrix bin for bin. A multi-member bundle is
    numerical and carries no reserved missing bin, because the dense
    integration routes missing values from the original matrix (see the module
    docstring); `missing_bin` is -1 on it.
    """
    if plan.n_features != data.n_features:
        raise Error("plan and matrix must agree on n_features")
    if plan.n_rows != data.n_rows:
        raise Error("plan and matrix must agree on n_rows")

    var n_rows = data.n_rows
    var n_bundles = plan.n_bundles()
    var n_bins = plan.max_bundle_bins()
    var bins = List[UInt8](capacity=n_rows * n_bundles)
    bins.resize(n_rows * n_bundles, 0)
    var missing_bin = List[Int](capacity=n_bundles)
    var cat_flags = List[Bool](capacity=n_bundles)
    var cat_codes = List[Int]()
    var cat_offsets = List[Int](capacity=n_bundles + 1)
    cat_offsets.append(0)

    for b in range(n_bundles):
        var lo = plan.bundle_start[b]
        var hi = plan.bundle_start[b + 1]
        var size = hi - lo
        var out_base = b * n_rows
        if size == 1:
            # Identity encoded: copy the column through untouched.
            var f = plan.members[lo]
            var in_base = f * n_rows
            for r in range(n_rows):
                bins[out_base + r] = data.bins[in_base + r]
            cat_flags.append(data.cats.is_cat(f))
            if data.cats.is_cat(f):
                for i in range(
                    data.cats.offsets[f], data.cats.offsets[f + 1]
                ):
                    cat_codes.append(data.cats.codes[i])
            missing_bin.append(data.missing_bin[f])
        else:
            # Every row starts at the shared bin, which is what "every member
            # is at its default" reads back as.
            for r in range(n_rows):
                bins[out_base + r] = UInt8(EFB_SHARED_BIN)
            for k in range(lo, hi):
                var f = plan.members[k]
                var d = plan.slot_default[k]
                var in_base = f * n_rows
                for r in range(n_rows):
                    var v = Int(data.bins[in_base + r])
                    if v == d:
                        continue
                    # The shared bin is "unclaimed": no member ever encodes to
                    # it, so an occupied slot means an earlier member already
                    # took this row and wins the collision.
                    if Int(bins[out_base + r]) != EFB_SHARED_BIN:
                        continue
                    bins[out_base + r] = UInt8(plan.encode(f, v))
            cat_flags.append(False)
            missing_bin.append(EFB_NONE)
        cat_offsets.append(len(cat_codes))

    return BinnedMatrix(
        bins^,
        n_rows,
        n_bundles,
        n_bins,
        CategoricalSpec(cat_flags^, cat_codes^, cat_offsets^),
        missing_bin^,
    )


struct BundledMatrix(Copyable, Movable):
    """A fitted plan together with the matrix it produced, or neither.

    This is the value `tree.grow_tree` takes: `active` False is "grow on the
    original matrix", which is both the default and the fallback, and `active`
    True means histograms come from `data` and split search unbundles through
    `plan`. It is a training-time view of one matrix, not a second
    representation of the model: nothing here is serialized, predicted from,
    or kept once the last tree is grown.
    """

    var plan: FeatureBundling
    var data: BinnedMatrix
    var active: Bool

    def __init__(
        out self,
        var plan: FeatureBundling,
        var data: BinnedMatrix,
        active: Bool,
    ):
        self.plan = plan^
        self.data = data^
        self.active = active

    @staticmethod
    def none() -> BundledMatrix:
        """No bundling: the value every caller that is not bundling passes,
        and the one a negative `use_bundling` verdict falls back to."""
        return BundledMatrix(
            FeatureBundling.none(),
            BinnedMatrix(List[UInt8](), 0, 0, 0),
            False,
        )


def columns_for_features(
    plan: FeatureBundling, features: List[Int]
) raises -> List[Int]:
    """The ascending, duplicate-free bundle columns a set of original features
    occupies, or every column when `features` is empty.

    Feature subsampling picks original features; histograms are accumulated by
    column. A column has to be accumulated when *any* of its members was
    picked, so this is where the two meet. The extra members' statistics ride
    along in that column and are never scanned, which is exactly what the
    unbundling in `split.find_best_split` makes safe: a member's histogram is
    recovered from the column whatever else shares it.

    The empty list is the "every feature" convention the histogram builders
    already use, and it maps to the "every column" convention unchanged.
    """
    var out = List[Int]()
    if len(features) == 0:
        return out^
    var n = plan.n_bundles()
    var seen = List[Bool](capacity=n)
    seen.resize(n, False)
    for i in range(len(features)):
        var f = features[i]
        if f < 0 or f >= plan.n_features:
            raise Error("feature index out of range for this bundling plan")
        seen[plan.bundle_of[f]] = True
    for b in range(n):
        if seen[b]:
            out.append(b)
    return out^


def prepare_bundling(
    data: BinnedMatrix, settings: EfbSettings
) raises -> BundledMatrix:
    """Fit and apply a bundling plan, or decide not to.

    Three ways to get `BundledMatrix.none()`, and all three are the same fit:

    - `settings.enabled` is False, which is the default;
    - the plan packs nothing (`n_bundles == n_features`) or does not clear
      `EfbParams.min_reduction`, so `fit_bundles_dense` reports
      `use_bundling = False`;
    - the matrix has no rows or no features to bundle.

    That is the conservative fallback the integration keeps: bundling changes
    how histograms are laid out and nothing else, so declining it is always
    available and always produces the same trees.

    Called once per training call, not once per tree: the plan is a function
    of the matrix and the parameters, so refitting it per tree would cost the
    same answer many times over. It is deterministic, so a continued run
    (`train_more`) rebuilds exactly the plan the first call used.
    """
    if not settings.enabled:
        return BundledMatrix.none()
    settings.check()
    if data.n_rows < 1 or data.n_features < 2:
        return BundledMatrix.none()
    var plan = fit_bundles_dense(data, settings.params)
    if not plan.use_bundling:
        return BundledMatrix.none()
    var bundled = bundle_dense(data, plan)
    return BundledMatrix(plan^, bundled^, True)


def bundle_csc(
    data: SparseBinnedMatrix, plan: FeatureBundling
) raises -> SparseBinnedMatrix:
    """Apply a plan, producing the bundled binned matrix.

    One column per bundle, canonical CSC. Entries at a member's default bin
    are dropped, since the bundle's default bin recovers them; that is what
    makes an explicitly stored zero and an absent entry produce identical
    output. On a collision the earliest member in the bundle's member order
    keeps the row (see the module docstring), so the result is a deterministic
    function of the matrix and the plan.

    A singleton bundle is identity encoded, so its category table and its
    reserved missing bin carry over unchanged. A multi-member bundle is
    numerical and carries a reserved missing bin only when exactly one of its
    members reserves one; `plan.missing_bins` reports the full list either
    way.
    """
    if plan.n_features != data.n_features:
        raise Error("plan and matrix must agree on n_features")
    if plan.n_rows != data.n_rows:
        raise Error("plan and matrix must agree on n_rows")

    var n_bundles = plan.n_bundles()
    var out_rows = List[Int]()
    var out_bins = List[UInt8]()
    var col_offsets = List[Int](capacity=n_bundles + 1)
    col_offsets.append(0)
    var default_bin = List[UInt8](capacity=n_bundles)
    var missing_bin = List[Int](capacity=n_bundles)
    var cat_flags = List[Bool](capacity=n_bundles)
    var cat_codes = List[Int]()
    var cat_offsets = List[Int](capacity=n_bundles + 1)
    cat_offsets.append(0)

    for b in range(n_bundles):
        var lo = plan.bundle_start[b]
        var hi = plan.bundle_start[b + 1]
        var size = hi - lo
        var acc_rows = List[Int]()
        var acc_bins = List[UInt8]()
        for k in range(lo, hi):
            var f = plan.members[k]
            var d = data.default_bin[f]
            var m_rows = List[Int]()
            var m_bins = List[UInt8]()
            for i in range(
                data.col_offsets[f], data.col_offsets[f + 1]
            ):
                if data.bin[i] == d:
                    continue
                m_rows.append(data.row_index[i])
                m_bins.append(UInt8(plan.encode(f, Int(data.bin[i]))))
            if k == lo:
                acc_rows = m_rows^
                acc_bins = m_bins^
                continue
            # Merge, with the already-accumulated (earlier) member winning
            # every collision.
            var new_rows = List[Int](capacity=len(acc_rows) + len(m_rows))
            var new_bins = List[UInt8](capacity=len(acc_rows) + len(m_rows))
            var i = 0
            var j = 0
            while i < len(acc_rows) and j < len(m_rows):
                if acc_rows[i] <= m_rows[j]:
                    if acc_rows[i] == m_rows[j]:
                        j += 1
                    new_rows.append(acc_rows[i])
                    new_bins.append(acc_bins[i])
                    i += 1
                else:
                    new_rows.append(m_rows[j])
                    new_bins.append(m_bins[j])
                    j += 1
            while i < len(acc_rows):
                new_rows.append(acc_rows[i])
                new_bins.append(acc_bins[i])
                i += 1
            while j < len(m_rows):
                new_rows.append(m_rows[j])
                new_bins.append(m_bins[j])
                j += 1
            acc_rows = new_rows^
            acc_bins = new_bins^

        for i in range(len(acc_rows)):
            out_rows.append(acc_rows[i])
            out_bins.append(acc_bins[i])
        col_offsets.append(len(out_rows))
        default_bin.append(UInt8(plan.bundle_default_bin(b)))

        var mb = plan.missing_bins(b)
        missing_bin.append(mb[0] if len(mb) == 1 else EFB_NONE)

        var is_cat = size == 1 and data.cats.is_cat(plan.members[lo])
        cat_flags.append(is_cat)
        if is_cat:
            var f = plan.members[lo]
            for i in range(
                data.cats.offsets[f], data.cats.offsets[f + 1]
            ):
                cat_codes.append(data.cats.codes[i])
        cat_offsets.append(len(cat_codes))

    return SparseBinnedMatrix(
        out_rows^,
        out_bins^,
        col_offsets^,
        default_bin^,
        data.n_rows,
        n_bundles,
        plan.max_bundle_bins(),
        CategoricalSpec(cat_flags^, cat_codes^, cat_offsets^),
        missing_bin^,
    )


def _recover_member_into(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    out_base: Int,
    out_bins: Int,
    plan: FeatureBundling,
    bundle: Int,
    slot_rank: Int,
    grad: List[Float64],
    hess: List[Float64],
    count: List[Int],
    base: Int,
) raises:
    """One member's local histogram, written into
    `out_*[out_base : out_base + out_bins)`.

    The single copy of the recovery arithmetic. Both public forms below go
    through it, so a caller reading one member and a caller expanding a whole
    bundled histogram cannot get different numbers. Bins past the member's own
    count are zeroed, which is what a rectangular histogram holds for them
    anyway, so the whole output range is written and a reused buffer never
    leaks a previous node's statistics.
    """
    if bundle < 0 or bundle >= plan.n_bundles():
        raise Error("bundle index out of range")
    var lo = plan.bundle_start[bundle]
    var hi = plan.bundle_start[bundle + 1]
    if slot_rank < 0 or slot_rank >= hi - lo:
        raise Error("slot rank out of range for this bundle")
    var width = plan.bundle_bins[bundle]
    if base < 0 or base + width > len(grad):
        raise Error("histogram block is outside the supplied arrays")
    if len(hess) != len(grad) or len(count) != len(grad):
        raise Error("grad, hess, and count must have equal length")

    var k = lo + slot_rank
    var n_local = plan.slot_bins[k]
    if n_local > out_bins:
        raise Error(
            "member needs more bins than the output histogram has per feature"
        )
    if out_base < 0 or out_base + out_bins > len(out_grad):
        raise Error("output block is outside the supplied arrays")
    if len(out_hess) != len(out_grad) or len(out_count) != len(out_grad):
        raise Error("output grad, hess, and count must have equal length")

    for b in range(n_local, out_bins):
        out_grad[out_base + b] = 0.0
        out_hess[out_base + b] = 0.0
        out_count[out_base + b] = 0

    if hi - lo == 1:
        # Identity encoded: the block is already the member's histogram.
        for b in range(n_local):
            out_grad[out_base + b] = grad[base + b]
            out_hess[out_base + b] = hess[base + b]
            out_count[out_base + b] = count[base + b]
        return

    var total_grad = 0.0
    var total_hess = 0.0
    var total_count = 0
    for b in range(width):
        total_grad += grad[base + b]
        total_hess += hess[base + b]
        total_count += count[base + b]

    var d = plan.slot_default[k]
    var start = plan.slot_offset[k]
    for i in range(n_local - 1):
        var local = i if i < d else i + 1
        out_grad[out_base + local] = grad[base + start + i]
        out_hess[out_base + local] = hess[base + start + i]
        out_count[out_base + local] = count[base + start + i]
        total_grad -= grad[base + start + i]
        total_hess -= hess[base + start + i]
        total_count -= count[base + start + i]
    out_grad[out_base + d] = total_grad
    out_hess[out_base + d] = total_hess
    out_count[out_base + d] = total_count


def unbundle_histogram_into(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    plan: FeatureBundling,
    bundle: Int,
    slot_rank: Int,
    grad: List[Float64],
    hess: List[Float64],
    count: List[Int],
    base: Int,
) raises:
    """`unbundle_histogram` writing into buffers the caller owns.

    The lists are resized to the member's local bin count and fully written,
    so a reused buffer never leaks its previous contents. This is the same
    `_into` split the histogram builders and the row partitioner already make,
    and for the same reason.
    """
    if bundle < 0 or bundle >= plan.n_bundles():
        raise Error("bundle index out of range")
    var lo = plan.bundle_start[bundle]
    var hi = plan.bundle_start[bundle + 1]
    if slot_rank < 0 or slot_rank >= hi - lo:
        raise Error("slot rank out of range for this bundle")
    var n_local = plan.slot_bins[lo + slot_rank]
    out_grad.resize(n_local, 0.0)
    out_hess.resize(n_local, 0.0)
    out_count.resize(n_local, 0)
    _recover_member_into(
        out_grad,
        out_hess,
        out_count,
        0,
        n_local,
        plan,
        bundle,
        slot_rank,
        grad,
        hess,
        count,
        base,
    )


def expand_bundled_histogram(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    out_n_bins: Int,
    plan: FeatureBundling,
    grad: List[Float64],
    hess: List[Float64],
    count: List[Int],
    src_n_bins: Int,
    features: List[Int],
) raises:
    """Turn a per-bundle histogram back into a per-feature one.

    This is where a bundle stops existing. `grad`/`hess`/`count` are a
    histogram over the bundled matrix, laid out `[bundle * src_n_bins + b]`;
    the output is laid out `[feature * out_n_bins + b]` in the plan's original
    feature space, which is exactly the shape `split.find_best_split` and
    `tree._leaf_value` have always read. Every consumer downstream of it --
    the scan, the tree, prediction, serialization, importance, contributions
    -- therefore needs no knowledge of bundling at all.

    A non-empty `features` expands only those features and leaves every other
    slice untouched, matching `build_histogram_into`'s convention that
    unselected features stay zero; the caller zeroes the buffer once per tree
    and the selected slices are fully rewritten per node.

    Cost and exactness. The expansion is O(#features x out_n_bins) per node,
    the same order as the sibling subtraction the grower already does, while
    the accumulation it feeds off is O(#rows x #bundles) instead of
    O(#rows x #features) -- which is the whole point. It is also linear, so
    expanding a parent and a child and subtracting gives the same numbers as
    subtracting first and expanding the difference; that is what lets the
    grower keep the histogram-subtraction trick unchanged. A member's default
    bin is recovered by subtraction from its bundle's block total rather than
    accumulated directly, so it agrees with an unbundled fit exactly in exact
    arithmetic and to floating-point association in practice, which is the
    same trade sibling subtraction already makes.
    """
    if out_n_bins < 1 or src_n_bins < 1:
        raise Error("histogram bin counts must be positive")
    if len(out_grad) != plan.n_features * out_n_bins:
        raise Error("expanded histogram must be n_features x out_n_bins")
    var use_all = len(features) == 0
    var n_active = plan.n_features if use_all else len(features)
    for i in range(n_active):
        var f = i if use_all else features[i]
        if f < 0 or f >= plan.n_features:
            raise Error("feature index out of range for this bundling plan")
        var bundle = plan.bundle_of[f]
        _recover_member_into(
            out_grad,
            out_hess,
            out_count,
            f * out_n_bins,
            out_n_bins,
            plan,
            bundle,
            plan.slot_of[f] - plan.bundle_start[bundle],
            grad,
            hess,
            count,
            bundle * src_n_bins,
        )


def unbundle_histogram(
    plan: FeatureBundling,
    bundle: Int,
    slot_rank: Int,
    grad: List[Float64],
    hess: List[Float64],
    count: List[Int],
    base: Int,
) raises -> LocalHistogram:
    """One member's local histogram, recovered from its bundle's.

    `grad`, `hess`, and `count` are a histogram whose bundle-`bundle` block
    starts at index `base` and runs for `plan.bundle_bins[bundle]` bins;
    `slot_rank` selects the member within the bundle, 0-based in member
    order.

    The member's own range copies straight back. Its default bin is recovered
    by subtracting that range from the block total, because a row where this
    member is at its default sits either in the shared bin or inside another
    member's range. That subtraction is exact when the bundle is lossless; a
    collision the member lost is folded into its default bin, which is the
    approximation `max_conflict_rate` buys.

    The allocating form, kept because it is the one a caller reading a single
    member wants. The split search uses `unbundle_histogram_into`, which is
    the same arithmetic in the same order writing into reused buffers; the two
    cannot drift because this one is that one.
    """
    var out_grad = List[Float64]()
    var out_hess = List[Float64]()
    var out_count = List[Int]()
    unbundle_histogram_into(
        out_grad,
        out_hess,
        out_count,
        plan,
        bundle,
        slot_rank,
        grad,
        hess,
        count,
        base,
    )
    return LocalHistogram(out_grad^, out_hess^, out_count^)
