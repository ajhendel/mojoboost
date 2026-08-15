"""Cost-effective gradient boosting (CEGB): the whole of LightGBM's four
`cegb_*` controls, in one place.

CEGB does not change what a tree *is*. It changes what a candidate split is
worth: every candidate is charged the cost of the data access it would force,
and the charge is subtracted from the split's gain before the split competes.
A feature that buys little accuracy for a lot of I/O stops winning, so the
fitted ensemble reads fewer features and fewer rows at prediction time.

The gain adjustment, in full
----------------------------
For a node `N` whose active rows are `R(N)` and a dataset feature `f`:

    gain_cegb(f, N) = contri[f] * gain_raw(f, N) - delta(f, N)

    delta(f, N) = tradeoff * ( penalty_split * |R(N)|
                             + coupled[f] * [f has never been split on]
                             + lazy[f]    * unread(f, R(N)) )

    unread(f, R) = |{ r in R : row r has never had feature f read for it }|

`gain_raw` is the ordinary second-order split gain, `contri[f]` is LightGBM's
`feature_contri` multiplier, and `[P]` is 1 when `P` holds and 0 otherwise.
Three properties of that formula are the whole design:

- **`contri` is a multiplier and the CEGB terms are absolute.** The multiplier
  scales the gain first; the costs are then subtracted as amounts of gain, not
  rescaled by the multiplier. A caller who sets both gets LightGBM's
  composition and not a product of the two.
- **`contri` is applied exactly once, and not here.** This module never
  multiplies by `contri`. `tree_parameters_extra.FeaturePenalties.contri_of`
  is that multiplier's only owner, and `cegb_adjusted_gain` takes a gain that
  has *already* been multiplied. Double-applying `feature_contri` is the one
  way these two mechanisms can silently corrupt each other, so the seam is
  drawn here on purpose. See "Fusing the pre-existing fragment" below.
- **Only the split term is a pure function of the node.** The coupled term
  reads a per-feature flag that lives for the whole ensemble, and the lazy
  term reads a per-(feature, row) flag that does too. Those two need
  `CegbLedger`, which is training state a grower must carry and hand back.

Active rows, bagging, and GOSS
------------------------------
`|R(N)|` is the node's *active* row count: the rows the node actually holds,
which under row bagging or GOSS are the sampled rows and not the dataset's.
That is deliberate and it matches LightGBM, whose data partition is built over
the bagged index set. Two consequences a caller should know:

- A given `cegb_penalty_split` buys a different amount of regularization at
  `bagging_fraction=0.5` than at 1.0, because the charge halves while the
  gains, computed from the same sampled gradients, do not. CEGB's costs are
  absolute quantities of gain, so they do not renormalize themselves. This
  module does not rescale them either: silently dividing by the bagging
  fraction would be a different penalty from LightGBM's, and inventing one
  knob to correct another is worse than saying so.
- Under GOSS the gradients of the `other` rows are already multiplied by
  `GossSelection.multiplier`, so gains are on a full-data scale while
  `|R(N)|` is on a sampled scale. The mismatch is GOSS's, not CEGB's, and it
  is recorded here rather than corrected.

The lazy ledger is indexed by *global* row id for the same reason: a row that
sits out one round's bag must keep whatever read-state it had, so that the
next round in which it is sampled charges for it correctly. Every row list
this module accepts (`grow_tree`'s `bag`, and the child row lists
`partition_rows_into` derives from it) already holds global ids.

Determinism
-----------
Every quantity here is either an integer count or one multiplication of three
Float64 values, in a fixed order, from state updated only when a split is
committed. Nothing depends on scan order, thread count, or how many nodes were
scored before this one:

- `unread` is counted as an integer and multiplied once. LightGBM accumulates
  `lazy[f]` once per unread row (`CalculateOndemandCosts` does `total +=
  penalty` inside its row loop, confirmed against LightGBM master), which for
  a large leaf is a different floating-point number from `lazy[f] * count`.
  Ours is the exact one, and `docs/CEGB.md` records it as an intentional
  difference so the parity table does not call the lazy penalty
  bit-comparable.
- The ledger is written once per *committed* split, never during a scan, so a
  parallel scan cannot observe a half-updated ledger and two scans of the same
  node score identically.
- In a distributed run every rank commits the same split, so every rank's
  coupled flag flips at the same moment with no message. The lazy count does
  not have that property; see `check_cegb_distributed`.

One home for the parameters
---------------------------
`tree_parameters_extra.FeaturePenalties` holds a `CegbConfig` and nothing else
CEGB-shaped: its `penalized_gain` applies the `feature_contri` multiplier and
stops there, and every CEGB term is subtracted by this module, through
`CegbNodeCosts`, at `split._feature_gain`. There is exactly one implementation
of the arithmetic and exactly one place a caller can set the parameters.

The import runs one way -- `tree_parameters_extra` imports `cegb`, never the
reverse -- and this module imports nothing from the package at all, which is
what keeps that edge acyclic: `binning` and `categorical` both import
`tree_parameters_extra`, so any import from here into the data layer would
close a loop. That is why the one bundling-shaped question CEGB asks, which
dataset feature a split in a bundled search space is charged to, lives on
`efb.FeatureBundling.charged_feature` instead of here.

Where this is charged
---------------------
`tree.grow_tree` is the grower that carries the ledger: it builds one
`CegbNodeCosts` per node before the scan, passes it to `tree._search`, and
calls `cegb_commit_split` when a split is chosen. `boosting.fit` and
`boosting.fit_multiclass` own the `CegbLedger` for the whole ensemble and hand
the same one to every tree. Every other grower leaves the ledger inert, which
charges the split cost and *refuses* the coupled and lazy penalties through
`check_cegb_grower_support` rather than ignoring them.
"""

# LightGBM's `cegb_tradeoff` default. A tradeoff of 0.0 switches every CEGB
# term off at once whatever the individual penalties are, which is the
# documented way to disable the whole mechanism without clearing its vectors.
comptime CEGB_DEFAULT_TRADEOFF = 1.0

# 64 rows per word in the lazy read bitset.
comptime _ROW_BITS = 6
comptime _ROW_MASK = 63


@always_inline
def _is_finite(v: Float64) -> Bool:
    """False for NaN and for either infinity.

    A private copy of `tree_parameters_extra._is_finite`, repeated rather than
    imported so this module reaches into no other module's private names. The
    handoff asks for one shared copy at integration.
    """
    return v == v and abs(v) <= Float64.MAX_FINITE


# ---------------------------------------------------------------------------
# The parameters
# ---------------------------------------------------------------------------


struct CegbConfig(Copyable, Movable):
    """LightGBM's four `cegb_*` parameters, with LightGBM's defaults.

    - `tradeoff` (`cegb_tradeoff`, 1.0): one scalar multiplying every cost, so
      a user tunes the accuracy/cost exchange rate without rewriting the
      per-feature vectors. 0.0 disables CEGB entirely.
    - `penalty_split` (`cegb_penalty_split`, 0.0): charged per split, per
      active row in the node being split. This is the cost of *evaluating* a
      split at prediction time, which every row reaching the node pays.
    - `penalty_feature_coupled` (`cegb_penalty_feature_coupled`, empty):
      charged once, the first time a feature is split on anywhere in the
      ensemble. This is the cost of having to compute the feature at all --
      paid once, by the model, not per row.
    - `penalty_feature_lazy` (`cegb_penalty_feature_lazy`, empty): charged per
      active row in the node that has not yet had this feature read for it.
      This is the cost of computing a feature on demand, for the rows that
      have not already forced it.

    An empty per-feature vector means every entry is 0.0, so the default
    bundle charges nothing and `is_active` is False. Growers test that once
    per tree rather than adding 0.0 per candidate.

    INTENTIONAL DIFFERENCES FROM LightGBM

    - Every value must be finite and nonnegative. A negative cost is a bonus
      for reading data, which inverts the mechanism's meaning and can make a
      chosen split's adjusted gain exceed its raw gain; LightGBM does not
      check. Zero is accepted everywhere and is the inactive value.
    - The two vectors are indexed by *dataset* feature id. LightGBM indexes
      `feature_used_` by its internal post-filter feature index and the
      penalty vectors by the original index; mojotrees has one index for both,
      except under feature bundling, where `cegb_dataset_feature` recovers it.
    """

    var tradeoff: Float64
    var penalty_split: Float64
    var penalty_feature_coupled: List[Float64]
    var penalty_feature_lazy: List[Float64]

    def __init__(out self):
        """LightGBM's defaults, which charge nothing."""
        self.tradeoff = CEGB_DEFAULT_TRADEOFF
        self.penalty_split = 0.0
        self.penalty_feature_coupled = List[Float64]()
        self.penalty_feature_lazy = List[Float64]()

    @staticmethod
    def none() -> CegbConfig:
        """The inactive bundle, spelled out at a call site."""
        return CegbConfig()

    @staticmethod
    def of(
        tradeoff: Float64,
        penalty_split: Float64 = 0.0,
        penalty_feature_coupled: List[Float64] = [],
        penalty_feature_lazy: List[Float64] = [],
    ) raises -> CegbConfig:
        """A bundle from the four LightGBM values."""
        var out = CegbConfig()
        out.tradeoff = tradeoff
        out.penalty_split = penalty_split
        out.penalty_feature_coupled = penalty_feature_coupled.copy()
        out.penalty_feature_lazy = penalty_feature_lazy.copy()
        return out^

    def coupled_of(self, feature: Int) -> Float64:
        """This feature's first-use cost, 0.0 when unset or out of range."""
        if feature < 0 or feature >= len(self.penalty_feature_coupled):
            return 0.0
        return self.penalty_feature_coupled[feature]

    def lazy_of(self, feature: Int) -> Float64:
        """This feature's per-unread-row cost, 0.0 when unset."""
        if feature < 0 or feature >= len(self.penalty_feature_lazy):
            return 0.0
        return self.penalty_feature_lazy[feature]

    def split_cost_active(self) -> Bool:
        """Whether the per-split, per-row cost would change a gain. This is
        the one CEGB term a split search can charge with no ledger at all: it
        is a function of (tradeoff, penalty_split, active rows)."""
        return self.tradeoff != 0.0 and self.penalty_split != 0.0

    def coupled_active(self) -> Bool:
        """Whether any first-use cost would be charged. An all-zero vector
        charges nothing however long it is, so length alone is not the test."""
        if self.tradeoff == 0.0:
            return False
        for f in range(len(self.penalty_feature_coupled)):
            if self.penalty_feature_coupled[f] != 0.0:
                return True
        return False

    def lazy_active(self) -> Bool:
        """Whether any per-unread-row cost would be charged."""
        if self.tradeoff == 0.0:
            return False
        for f in range(len(self.penalty_feature_lazy)):
            if self.penalty_feature_lazy[f] != 0.0:
                return True
        return False

    def is_active(self) -> Bool:
        """Whether anything here would change a gain."""
        return (
            self.split_cost_active()
            or self.coupled_active()
            or self.lazy_active()
        )

    def needs_feature_ledger(self) -> Bool:
        """Whether a per-feature first-use flag has to be carried across the
        whole ensemble for this bundle to be charged correctly."""
        return self.coupled_active()

    def needs_row_ledger(self) -> Bool:
        """Whether a per-(feature, row) read flag has to be carried across the
        whole ensemble. Only the lazy penalty needs it, and it is the reason
        that penalty is a subsystem rather than a formula."""
        return self.lazy_active()

    def needs_ledger(self) -> Bool:
        """Whether `CegbLedger` is required at all. False for the split cost
        alone, which is why that half is live in the split search today."""
        return self.needs_feature_ledger() or self.needs_row_ledger()

    def n_ledger_features(self) -> Int:
        """The number of features the ledger must cover for this bundle, from
        the vectors themselves. `CegbLedger.create` sizes against the dataset
        instead; this is the lower bound `check` enforces."""
        var n = len(self.penalty_feature_coupled)
        if len(self.penalty_feature_lazy) > n:
            n = len(self.penalty_feature_lazy)
        return n

    def check(self, n_features: Int) raises:
        """Raise unless every value is usable and every vector fits a dataset
        with `n_features` columns.

        Values are rejected rather than clamped, in the style of
        `params._validate`: a caller who wrote a negative cost or a vector of
        the wrong length asked for something this cannot do, and finding out
        before the first histogram is better than finding out from a model
        that quietly ignored it.
        """
        if not _is_finite(self.tradeoff) or self.tradeoff < 0.0:
            raise Error("cegb_tradeoff must be a finite nonnegative number")
        if not _is_finite(self.penalty_split) or self.penalty_split < 0.0:
            raise Error(
                "cegb_penalty_split must be a finite nonnegative number"
            )
        var n_coupled = len(self.penalty_feature_coupled)
        if n_coupled > 0 and n_coupled != n_features:
            raise Error(
                "cegb_penalty_feature_coupled has ",
                n_coupled,
                " entries but the data has ",
                n_features,
                " features",
            )
        for f in range(n_coupled):
            var v = self.penalty_feature_coupled[f]
            if not _is_finite(v) or v < 0.0:
                raise Error(
                    "cegb_penalty_feature_coupled[",
                    f,
                    "] must be a finite nonnegative number",
                )
        var n_lazy = len(self.penalty_feature_lazy)
        if n_lazy > 0 and n_lazy != n_features:
            raise Error(
                "cegb_penalty_feature_lazy has ",
                n_lazy,
                " entries but the data has ",
                n_features,
                " features",
            )
        for f in range(n_lazy):
            var v = self.penalty_feature_lazy[f]
            if not _is_finite(v) or v < 0.0:
                raise Error(
                    "cegb_penalty_feature_lazy[",
                    f,
                    "] must be a finite nonnegative number",
                )


# ---------------------------------------------------------------------------
# The ledger: the state CEGB needs that a node does not carry
# ---------------------------------------------------------------------------


struct CegbLedger(Copyable, Movable):
    """What the ensemble has already paid for.

    Two flags, both written only when a split is *committed* and both living
    for the whole ensemble rather than for one tree:

    - `feature_used[f]`: feature `f` has been split on somewhere. The coupled
      cost is charged while this is False and never again after.
    - `row_read`, a bitset over (feature, global row id): row `r` has had
      feature `f` read for it, because `r` passed through a node that split on
      `f`. The lazy cost is charged per row in the node for which this is
      False.

    Ensemble lifetime, not tree lifetime, is the whole point: CEGB's premise
    is that the *model* pays for a feature once, so a second tree that reuses
    a feature the first tree already needed gets it free. A ledger reset per
    tree would charge the first-use cost once per tree, which is a different
    (and much harsher) regularizer. Confirmed against LightGBM master:
    `CostEfficientGradientBoosting::Init` sizes `is_feature_used_in_split_`
    and the `feature_used_in_data_` bitset once and `BeforeTrain` does not
    clear either, while it *does* clear the per-tree `splits_per_leaf_` cache
    beside them. Both lifetimes are deliberate there and both are matched
    here.

    LIFECYCLE AND CONTINUED TRAINING

    The ledger is training state, not model state. It is not serialized: a
    saved model records the trees the ledger produced, not the ledger. That is
    correct for scoring and for `save_model`/`load_model`, and it is a real
    limitation for `boosting.fit_more` after a round trip through disk -- a
    resumed run starts with an empty ledger and recharges every first-use
    cost, growing a different ensemble from the one an uninterrupted run would
    have grown. `handoffs/remaining_04_cegb.md` carries the serialization
    request that fixes it; until then `check_cegb_continued_training` refuses
    the combination rather than letting it silently diverge.
    """

    var feature_used: List[Bool]
    var row_read: List[UInt64]
    var n_features: Int
    var n_rows: Int
    var words_per_feature: Int
    var tracks_features: Bool
    var tracks_rows: Bool

    def __init__(out self):
        """An inert ledger, tracking nothing. This is what a grower carries
        when the configuration needs no ledger, so the ledger argument is
        never optional and never a null check at a call site."""
        self.feature_used = List[Bool]()
        self.row_read = List[UInt64]()
        self.n_features = 0
        self.n_rows = 0
        self.words_per_feature = 0
        self.tracks_features = False
        self.tracks_rows = False

    @staticmethod
    def none() -> CegbLedger:
        """The inert ledger, spelled out at a call site."""
        return CegbLedger()

    @staticmethod
    def create(
        config: CegbConfig, n_features: Int, n_rows: Int
    ) raises -> CegbLedger:
        """The ledger `config` needs over a dataset of this shape.

        Allocates only what is actually charged: nothing for a configuration
        with no coupled or lazy cost, one Bool per feature for the coupled
        cost, and `n_features * ceil(n_rows / 64)` words on top for the lazy
        cost. The row bitset is the expensive one -- one bit per (feature,
        row), so 12.5 MB for 100 features over 10 million rows -- which is why
        it is allocated only when a nonzero lazy penalty is actually set.
        """
        if n_features < 0 or n_rows < 0:
            raise Error("cegb ledger needs nonnegative dimensions")
        var out = CegbLedger()
        if not config.needs_ledger():
            return out^
        if config.n_ledger_features() > n_features:
            raise Error(
                "cegb penalty vectors cover ",
                config.n_ledger_features(),
                " features but the data has ",
                n_features,
            )
        out.n_features = n_features
        out.n_rows = n_rows
        out.tracks_features = True
        out.feature_used = List[Bool](capacity=n_features)
        out.feature_used.resize(n_features, False)
        if config.needs_row_ledger():
            out.tracks_rows = True
            out.words_per_feature = (n_rows + _ROW_MASK) >> _ROW_BITS
            var words = n_features * out.words_per_feature
            out.row_read = List[UInt64](capacity=words)
            out.row_read.resize(words, UInt64(0))
        return out^

    def is_tracking(self) -> Bool:
        """Whether this ledger records anything at all."""
        return self.tracks_features or self.tracks_rows

    def bytes_allocated(self) -> Int:
        """How much memory this ledger holds.

        The row bitset is `n_features * n_rows` bits and is the only part that
        can be large: 12.5 MB for 100 features over 10 million rows, 1.25 GB
        at 10000 features. Reported rather than capped, because any cap would
        be a number this library invented; a caller that wants to refuse a
        configuration can ask before training and say so in its own terms.
        """
        return len(self.feature_used) + 8 * len(self.row_read)

    def feature_is_used(self, feature: Int) -> Bool:
        """Whether `feature` has been split on before now.

        An untracked ledger answers True: with no coupled cost configured
        there is no first-use charge to make, and True is the answer that
        charges nothing. `split._feature_gain` passes the same constant today
        for the same reason.
        """
        if not self.tracks_features:
            return True
        if feature < 0 or feature >= len(self.feature_used):
            return True
        return self.feature_used[feature]

    def mark_feature_used(mut self, feature: Int) raises -> Bool:
        """Record that `feature` has now been split on. Returns True when this
        is the transition from unused to used, which is exactly when cached
        candidate gains elsewhere in the tree need their coupled charge
        refunded (see `cegb_commit_split`)."""
        if not self.tracks_features:
            return False
        if feature < 0 or feature >= len(self.feature_used):
            raise Error(
                "cegb ledger: feature ",
                feature,
                " is out of range for ",
                len(self.feature_used),
                " features",
            )
        if self.feature_used[feature]:
            return False
        self.feature_used[feature] = True
        return True

    def _row_word(self, feature: Int, row: Int) raises -> Int:
        if feature < 0 or feature >= self.n_features:
            raise Error(
                "cegb ledger: feature ",
                feature,
                " is out of range for ",
                self.n_features,
                " features",
            )
        if row < 0 or row >= self.n_rows:
            raise Error(
                "cegb ledger: row ",
                row,
                " is out of range for ",
                self.n_rows,
                " rows",
            )
        return feature * self.words_per_feature + (row >> _ROW_BITS)

    def row_has_read(self, feature: Int, row: Int) raises -> Bool:
        """Whether feature `feature` has already been read for global row
        `row`. An untracked ledger answers True, which charges nothing."""
        if not self.tracks_rows:
            return True
        var w = self._row_word(feature, row)
        var bit = UInt64(1) << UInt64(row & _ROW_MASK)
        return (self.row_read[w] & bit) != UInt64(0)

    def mark_row_read(mut self, feature: Int, row: Int) raises -> Bool:
        """Record that `feature` has now been read for global row `row`.
        Returns True when this is the first time."""
        if not self.tracks_rows:
            return False
        var w = self._row_word(feature, row)
        var bit = UInt64(1) << UInt64(row & _ROW_MASK)
        if (self.row_read[w] & bit) != UInt64(0):
            return False
        self.row_read[w] = self.row_read[w] | bit
        return True

    def count_unread(self, feature: Int, rows: List[Int]) raises -> Int:
        """How many of `rows` have not yet had `feature` read for them.

        `rows` holds global row ids, so the answer is stable across bagging
        rounds: a row that sat out a round keeps whatever state it had. Counts
        are integers, so this is exact and independent of the order `rows`
        happens to be in.
        """
        if not self.tracks_rows:
            return 0
        var n = 0
        for i in range(len(rows)):
            if not self.row_has_read(feature, rows[i]):
                n += 1
        return n

    def mark_rows_read(mut self, feature: Int, rows: List[Int]) raises -> Int:
        """Record that `feature` was read for every row in `rows`. Returns how
        many were newly marked, which is the count the just-committed split
        was charged for."""
        if not self.tracks_rows:
            return 0
        var n = 0
        for i in range(len(rows)):
            if self.mark_row_read(feature, rows[i]):
                n += 1
        return n

    def n_features_used(self) -> Int:
        """How many distinct features the ensemble has split on. The quantity
        the coupled penalty exists to hold down, so a caller can report what
        it bought."""
        var n = 0
        for f in range(len(self.feature_used)):
            if self.feature_used[f]:
                n += 1
        return n

    def reset(mut self):
        """Forget everything, keeping the allocation. For a caller that trains
        a second, independent ensemble over the same data; not for a new tree
        inside one ensemble, which must keep the ledger it has."""
        for f in range(len(self.feature_used)):
            self.feature_used[f] = False
        for w in range(len(self.row_read)):
            self.row_read[w] = UInt64(0)


# ---------------------------------------------------------------------------
# The gain adjustment
# ---------------------------------------------------------------------------


@always_inline
def cegb_split_cost(config: CegbConfig, n_active_rows: Int) -> Float64:
    """`tradeoff * penalty_split * |R(N)|`, the term every feature at a node
    pays equally.

    Charged per active row because that is what a split costs at prediction
    time: every row that reaches the node is compared against the threshold.
    A node with no rows costs nothing.

    `CegbNodeCosts` stores the same quantity factored as
    `split_rate = tradeoff * penalty_split`, so it can charge a row count it
    learns later; the product is the same one, in the same order.
    """
    if config.tradeoff == 0.0 or config.penalty_split == 0.0:
        return 0.0
    if n_active_rows <= 0:
        return 0.0
    return config.tradeoff * config.penalty_split * Float64(n_active_rows)


@always_inline
def cegb_coupled_cost(
    config: CegbConfig, ledger: CegbLedger, feature: Int
) -> Float64:
    """`tradeoff * coupled[f]`, charged only while `f` is unused.

    Once the ensemble has split on `f` anywhere, the model already computes
    it, so a second split on it adds no feature-level cost at all.
    """
    if config.tradeoff == 0.0:
        return 0.0
    if ledger.feature_is_used(feature):
        return 0.0
    return config.tradeoff * config.coupled_of(feature)


def cegb_lazy_cost(
    config: CegbConfig, feature: Int, unread_rows: Int
) raises -> Float64:
    """`tradeoff * lazy[f] * unread(f, R(N))`, from a count the caller has
    already taken with `CegbLedger.count_unread`.

    Split from the counting so the count happens once per (node, feature)
    rather than once per candidate: the count is a function of the node's row
    set, which no candidate changes. It takes no ledger for the same reason --
    the ledger was already consulted to produce `unread_rows`.
    """
    if config.tradeoff == 0.0:
        return 0.0
    if unread_rows < 0:
        raise Error("cegb: unread row count must be nonnegative")
    if unread_rows == 0:
        return 0.0
    var lazy = config.lazy_of(feature)
    if lazy == 0.0:
        return 0.0
    return config.tradeoff * lazy * Float64(unread_rows)


def cegb_delta_gain(
    config: CegbConfig,
    ledger: CegbLedger,
    feature: Int,
    n_active_rows: Int,
    unread_rows: Int = 0,
) raises -> Float64:
    """`delta(f, N)`: the whole cost charged against a candidate on `feature`
    at a node holding `n_active_rows` rows, `unread_rows` of which have not
    read this feature.

    The three terms are summed in a fixed order -- split, coupled, lazy --
    so the result does not depend on which of them a caller happens to have
    configured. Always nonnegative, since `check` forbids a negative cost.
    """
    var delta = cegb_split_cost(config, n_active_rows)
    delta += cegb_coupled_cost(config, ledger, feature)
    delta += cegb_lazy_cost(config, feature, unread_rows)
    return delta


def cegb_adjusted_gain(
    gain: Float64,
    config: CegbConfig,
    ledger: CegbLedger,
    feature: Int,
    n_active_rows: Int,
    unread_rows: Int = 0,
) raises -> Float64:
    """`gain` after this feature's CEGB costs.

    `gain` must already carry `feature_contri` (see
    `tree_parameters_extra.FeaturePenalties.contri_of`) and must not already
    carry any CEGB term. This and `CegbNodeCosts.adjusted_gain` are the only
    two entry points that subtract, and they compute the same sum: this one
    for a caller holding a ledger, that one for a scan holding prepared
    per-node costs. The result may be negative, which is not an error -- the
    caller's `min_gain_to_split` floor rejects it, exactly as it rejects a
    candidate whose raw gain was too small.

    Charged once per feature, after that feature's scan and before its best
    candidate is compared against the running best. That placement is
    LightGBM's and it is the only one that is correct: the split cost is a
    property of the node, and the coupled and lazy costs are properties of the
    feature, so charging per candidate would multiply a per-feature cost by
    the number of thresholds the feature happens to offer.
    """
    return gain - cegb_delta_gain(
        config, ledger, feature, n_active_rows, unread_rows
    )


# ---------------------------------------------------------------------------
# One node's prepared costs
# ---------------------------------------------------------------------------


struct CegbNodeCosts(Copyable, Movable):
    """Every CEGB cost at one node, computed once before the scan.

    The scan then charges a candidate in one addition and one subtraction,
    with no ledger lookup and no row walk: `delta_of` is O(1). That matters
    because the lazy term costs O(|R(N)|) to count, and counting it per
    candidate -- which is what a literal reading of the formula would do --
    would make split search quadratic in the leaf size.

    `prepared` records which features were costed. A restricted scan (feature
    subsampling, interaction constraints) costs only the features it will
    look at, and asking for any other feature raises rather than returning a
    silently uncharged 0.0.

    `split_rate` is `tradeoff * penalty_split` and is kept apart from the
    node's row count so `delta_at` can charge the split term against a count
    the caller only learns inside its own scan. `split.find_best_split` is
    that caller: it falls back to the histogram's own total when it is given
    no row count, and that fallback must keep working.
    """

    var split_rate: Float64
    var coupled: List[Float64]
    var lazy: List[Float64]
    var prepared: List[Bool]
    var restricted: Bool
    var active: Bool
    var lazy_counted: Bool
    var n_active_rows: Int

    def __init__(out self):
        """No costs: what an inactive configuration prepares."""
        self.split_rate = 0.0
        self.coupled = List[Float64]()
        self.lazy = List[Float64]()
        self.prepared = List[Bool]()
        self.restricted = False
        self.active = False
        self.lazy_counted = False
        self.n_active_rows = 0

    @staticmethod
    def inactive() -> CegbNodeCosts:
        return CegbNodeCosts()

    def split_cost(self) -> Float64:
        """The split term at the row count this was prepared for."""
        if self.n_active_rows <= 0:
            return 0.0
        return self.split_rate * Float64(self.n_active_rows)

    def _check_feature(self, feature: Int) raises:
        if feature < 0 or feature >= len(self.prepared):
            raise Error(
                "cegb: feature ",
                feature,
                " was not costed for this node",
            )
        if self.restricted and not self.prepared[feature]:
            raise Error(
                "cegb: feature ",
                feature,
                " is outside this node's costed feature set; prepare the"
                " costs with the same feature list the scan uses",
            )

    def delta_of(self, feature: Int) raises -> Float64:
        """`delta(feature, N)` for the node this was prepared for."""
        if not self.active:
            return 0.0
        self._check_feature(feature)
        return (
            self.split_cost() + self.coupled[feature] + self.lazy[feature]
        )

    def delta_at(self, feature: Int, n_active_rows: Int) raises -> Float64:
        """`delta(feature, N)` with the split term charged against
        `n_active_rows` rather than the prepared count.

        For a caller that costs a node before it knows how many rows the node
        holds. It is refused once a lazy penalty has been counted: the lazy
        term was counted over a specific row set, and charging the split term
        against a different size would mean the two halves of one node's cost
        disagree about how big the node is.
        """
        if not self.active:
            return 0.0
        self._check_feature(feature)
        if self.lazy_counted and n_active_rows != self.n_active_rows:
            raise Error(
                "cegb: the lazy penalty was counted over ",
                self.n_active_rows,
                " rows but the split penalty is being charged against ",
                n_active_rows,
            )
        var split = 0.0
        if n_active_rows > 0:
            split = self.split_rate * Float64(n_active_rows)
        return split + self.coupled[feature] + self.lazy[feature]

    def adjusted_gain(self, gain: Float64, feature: Int) raises -> Float64:
        """`gain` minus this feature's costs. `gain` must already carry
        `feature_contri` and no CEGB term; see `cegb_adjusted_gain`."""
        if not self.active:
            return gain
        return gain - self.delta_of(feature)

    def adjusted_gain_at(
        self, gain: Float64, feature: Int, n_active_rows: Int
    ) raises -> Float64:
        """`adjusted_gain` with `delta_at`'s row count."""
        if not self.active:
            return gain
        return gain - self.delta_at(feature, n_active_rows)


def prepare_cegb_node(
    config: CegbConfig,
    ledger: CegbLedger,
    n_features: Int,
    n_active_rows: Int,
    rows: List[Int] = [],
    features: List[Int] = [],
) raises -> CegbNodeCosts:
    """Cost one node's features, once, before its scan.

    `n_active_rows` is the node's active row count, which under bagging or
    GOSS is a sampled count (see the module docstring). `rows` is the node's
    global row ids and is required only when a lazy penalty is configured;
    when it is required it must have exactly `n_active_rows` entries, since
    two different answers to "how many rows does this node hold" would charge
    the split term and the lazy term against different node sizes.

    `features` is the node's scan set (feature subsampling, interaction
    constraints); empty means every feature. Only the listed features are
    costed, so a node scanning 10 of 1000 features pays for 10 row walks.
    """
    var out = CegbNodeCosts()
    if not config.is_active():
        return out^
    if n_features < 0:
        raise Error("cegb: n_features must be nonnegative")
    if n_active_rows < 0:
        raise Error("cegb: n_active_rows must be nonnegative")
    # Each half of the ledger is checked against the half of the
    # configuration that reads it. `is_tracking()` alone would let a ledger
    # built for a coupled-only run cost a lazy penalty, and an untracked row
    # bitset answers "already read" to every question, so the lazy term would
    # come out as a plausible zero instead of an error.
    if config.needs_feature_ledger() and not ledger.tracks_features:
        raise Error(
            "cegb_penalty_feature_coupled is charged against a per-ensemble"
            " feature-use ledger; this node was costed with one that does not"
            " track features, which would charge every first use again. Build"
            " it with CegbLedger.create from this same configuration and"
            " carry it across the whole ensemble"
        )
    if config.needs_row_ledger() and not ledger.tracks_rows:
        raise Error(
            "cegb_penalty_feature_lazy is charged against a per-(feature,"
            " row) read bitset; this node was costed with a ledger that does"
            " not track rows, which would charge nothing at all. Build it"
            " with CegbLedger.create from this same configuration and carry"
            " it across the whole ensemble"
        )
    if config.needs_row_ledger():
        if len(rows) != n_active_rows:
            raise Error(
                "cegb_penalty_feature_lazy needs the node's row ids: got ",
                len(rows),
                " for a node of ",
                n_active_rows,
                " rows",
            )
    out.active = True
    out.n_active_rows = n_active_rows
    out.lazy_counted = config.needs_row_ledger()
    out.split_rate = 0.0
    if config.split_cost_active():
        out.split_rate = config.tradeoff * config.penalty_split
    out.coupled = List[Float64](capacity=n_features)
    out.coupled.resize(n_features, 0.0)
    out.lazy = List[Float64](capacity=n_features)
    out.lazy.resize(n_features, 0.0)
    out.prepared = List[Bool](capacity=n_features)
    out.restricted = len(features) > 0
    out.prepared.resize(n_features, not out.restricted)

    var want_coupled = config.coupled_active()
    var want_lazy = config.lazy_active()
    # The marking loop runs whenever the scan set is restricted, even for a
    # configuration with only the split cost: `prepared` starts all-False
    # there, so skipping the loop would leave every feature uncosted and make
    # `delta_of` raise for exactly the features the scan is about to ask
    # about. Returning early is only safe when every feature is prepared.
    if not out.restricted and not want_coupled and not want_lazy:
        return out^

    var n_scan = n_features if not out.restricted else len(features)
    for i in range(n_scan):
        var f = i if not out.restricted else features[i]
        if f < 0 or f >= n_features:
            raise Error(
                "cegb: feature ",
                f,
                " is out of range for ",
                n_features,
                " features",
            )
        if out.restricted:
            out.prepared[f] = True
        if want_coupled:
            out.coupled[f] = cegb_coupled_cost(config, ledger, f)
        if want_lazy:
            out.lazy[f] = cegb_lazy_cost(
                config, f, ledger.count_unread(f, rows)
            )
    return out^


# ---------------------------------------------------------------------------
# Committing a split: the ledger's only writer
# ---------------------------------------------------------------------------


@fieldwise_init
struct CegbCommit(Copyable, Movable):
    """What committing a split changed.

    - `coupled_refund`: `tradeoff * coupled[f]`, nonzero exactly when this
      split is the first use of its feature. Every *cached* best split
      elsewhere in the tree that was scored on the same feature was charged
      that amount and is now stale by it; see `cegb_stale_cached_gain`.
    - `rows_newly_read`: how many of the node's rows had this feature read for
      them for the first time, which is the count the lazy term charged.
    - `feature_newly_used`: whether this was the feature's first split.
    """

    var coupled_refund: Float64
    var rows_newly_read: Int
    var feature_newly_used: Bool

    @staticmethod
    def nothing() -> CegbCommit:
        return CegbCommit(0.0, 0, False)


def cegb_commit_split(
    mut ledger: CegbLedger,
    config: CegbConfig,
    feature: Int,
    rows: List[Int] = [],
) raises -> CegbCommit:
    """Record that `feature` was split on over `rows`, and report what that
    changed.

    The ledger's only writer. Called once, after a split has been *chosen* --
    never during a scan, so no scored candidate can observe a ledger that a
    concurrent scan is updating, and re-scoring a node gives the same answer
    it gave the first time.

    `rows` is the split node's global row ids and is needed only when a lazy
    penalty is configured: those are precisely the rows for which the feature
    now has to be computed. Rows outside the node are untouched, which is what
    makes the lazy penalty a per-row cost rather than a per-feature one.
    """
    if not ledger.is_tracking():
        return CegbCommit.nothing()
    var newly_used = ledger.mark_feature_used(feature)
    var refund = 0.0
    if newly_used:
        refund = config.tradeoff * config.coupled_of(feature)
    var newly_read = 0
    if ledger.tracks_rows:
        if config.lazy_active() and len(rows) == 0:
            raise Error(
                "cegb_penalty_feature_lazy: committing a split needs the"
                " node's row ids, so the rows that just read this feature can"
                " be marked; an empty list would recharge them forever"
            )
        newly_read = ledger.mark_rows_read(feature, rows)
    return CegbCommit(refund, newly_read, newly_used)


def cegb_stale_cached_gain(
    commit: CegbCommit, cached_feature: Int, split_feature: Int
) -> Float64:
    """How much a cached candidate gain is now understated, given what the
    just-committed split changed.

    Leaf-wise growth scores each leaf's best split once and keeps it in the
    frontier until that leaf is chosen. A cached candidate on feature `f` that
    was charged `f`'s first-use cost is stale the moment some *other* leaf's
    split uses `f` first: the model now computes `f` anyway, so the cached
    candidate's real gain is higher by `tradeoff * coupled[f]`.

    Add this to a cached gain rather than re-running the scan. Only candidates
    on the same feature were charged, so only they are refunded; the split
    term and the lazy term are properties of the cached candidate's own node
    and are unchanged by a split somewhere else.

    CONFIRMED AGAINST LightGBM master, and one difference remains.
    `CostEfficientGradientBoosting::UpdateLeafBestSplits` walks every leaf but
    the one just split, adds `cegb_tradeoff * cegb_penalty_feature_coupled[f]`
    to that leaf's cached candidate *on the newly used feature*, and installs
    it as the leaf's best split if it now beats it. The amount and the
    condition are this function's: only candidates on `f` are refunded, and
    the refund is exactly what was charged.

    What LightGBM can do and this cannot: it keeps a candidate per (leaf,
    feature) -- `splits_per_leaf_`, sized `num_leaves * num_features` and
    cleared per tree -- so a leaf whose best split is on some *other* feature
    can still have its runner-up on `f` promoted once `f` is free. mojotrees's
    frontier caches one candidate per leaf, so a refund can improve a cached
    best but never resurrect a candidate that was not it. Matching that needs
    the per-(leaf, feature) table, which is `n_leaves * n_features` split
    records against the one this grower keeps; `docs/CEGB.md` section 10
    carries it as the remaining difference rather than an open question.
    """
    if not commit.feature_newly_used:
        return 0.0
    if cached_feature != split_feature:
        return 0.0
    return commit.coupled_refund


# ---------------------------------------------------------------------------
# Combinations that are refused rather than approximated
# ---------------------------------------------------------------------------


def check_cegb_grower_support(
    config: CegbConfig, carries_ledger: Bool, carries_rows: Bool
) raises:
    """Raise unless a grower can honor this configuration.

    The repository's rule for a backend that cannot apply a setting is to say
    so, never to ignore it (see `tree._search`'s `grower_applies_extra`).
    CEGB splits three ways along exactly that line:

    - The split cost needs only the node's active row count, which every
      caller of `tree._search` already passes. Live everywhere.
    - The coupled cost needs a per-feature ledger threaded through every tree
      of the ensemble, so a grower must set `carries_ledger`.
    - The lazy cost needs the node's row ids as well, so a grower that has no
      materialized row list per node must leave `carries_rows` False.
    """
    if config.needs_feature_ledger() and not carries_ledger:
        raise Error(
            "cegb_penalty_feature_coupled is charged the first time a feature"
            " is split on anywhere in the ensemble, which needs a per-model"
            " feature-use ledger this grower does not carry. Train on a"
            " grower that threads CegbLedger, or leave the vector empty"
        )
    if config.needs_row_ledger() and not carries_rows:
        raise Error(
            "cegb_penalty_feature_lazy is charged per node row that has not"
            " yet read the feature, which needs the node's row ids and a"
            " per-(feature, row) ledger this grower does not carry. Train on"
            " a grower that threads CegbLedger with row ids, or leave the"
            " vector empty"
        )


def check_cegb_device_split_search(config: CegbConfig) raises:
    """Raise when CEGB is active under the GPU device split search.

    That path scores every candidate inside a kernel and downloads one record
    per node (`train_gpu._search_leaf_device`), so a host-side cost has
    nowhere to be applied: the winning candidate is already chosen by the time
    the host sees it, and a cost applied afterwards would rank nothing. The
    host split-search strategy routes through `tree._search` and does not have
    this problem.

    The lazy penalty is further out of reach: its bitset is per (feature,
    row), which for a GPU run means a device-resident bitset updated from the
    row-compaction pass. That is a device data structure, not a scoring
    tweak.
    """
    if not config.is_active():
        return
    raise Error(
        "cost-effective gradient boosting is not applied by the GPU device"
        " split search: candidates are ranked inside the kernel, so a"
        " host-side cost cannot change which one wins. Use"
        " split_search=SPLIT_SEARCH_HOST, or leave the cegb_* parameters at"
        " their defaults"
    )


def check_cegb_distributed(config: CegbConfig) raises:
    """Raise for the CEGB terms a data-parallel run cannot agree on.

    - The split cost is safe: every rank scores from the reduced histogram and
      the exact global row count, so every rank computes the same charge with
      no message.
    - The coupled cost is safe for the same reason once a ledger exists: every
      rank commits the same chosen split, so every rank's first-use flag flips
      at the same split, in the same order, with no message.
    - The lazy cost is not. `unread(f, R(N))` counts rows, and the rows are
      sharded: the true count is the sum over ranks, which needs one integer
      allreduce per (node, feature) scanned. That message does not exist in
      `distributed_transport`, and a rank that counted only its own shard
      would charge a fraction of the cost and could rank features differently
      from its peers, which is a split disagreement rather than a slightly
      wrong penalty.
    """
    if config.needs_row_ledger():
        raise Error(
            "cegb_penalty_feature_lazy is not supported by the distributed"
            " grower: the unread-row count is a sum over shards and would"
            " need one allreduce per node and feature, so a rank counting"
            " only its own shard could choose a different split from its"
            " peers. cegb_tradeoff, cegb_penalty_split, and"
            " cegb_penalty_feature_coupled are rank-consistent"
        )


def check_cegb_continued_training(config: CegbConfig, resumed: Bool) raises:
    """Raise when a resumed run would recharge costs the first run already
    paid.

    `CegbLedger` is training state and is not in the model file, so a
    `fit_more` that follows a `save_model`/`load_model` starts from an empty
    ledger: every feature the loaded ensemble already uses is charged its
    first-use cost a second time, and every row is treated as never having
    read anything. The resulting ensemble is not the one an uninterrupted run
    would have produced.

    `resumed` is False for a continued run that still holds the ledger the
    first run built, which is correct and needs no refusal. The split cost
    alone survives a round trip untouched, since it reads no ledger.
    """
    if not resumed:
        return
    if config.needs_ledger():
        raise Error(
            "cegb_penalty_feature_coupled and cegb_penalty_feature_lazy are"
            " charged against a ledger that spans the whole ensemble and is"
            " not stored in the model file, so training on from a loaded"
            " model would charge every feature's first use again. Continue"
            " from the in-memory booster, or leave those vectors empty"
        )
