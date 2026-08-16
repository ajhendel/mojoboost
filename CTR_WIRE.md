# CTR_WIRE.md

Edits for files `lane/ctr-serialize` does not own. Nothing below has been
applied. Every claim cites `file:line` at the state of this branch after the
lane's commit, so each one is checkable without trusting the note.

The one-line summary you need before reading any of it: **a CTR-carrying
model now saves and loads correctly, and nothing else about CTRs became
reachable.** Every edit below either corrects a reason that has become false
or adds a refusal to a path that would now index past the end of its own
arrays.

---

## Part 1. `python/mojotrees/sklearn.py` (2 sites, both text-only)

Neither site's *behavior* changes. Both still refuse. Both currently refuse
for a reason that is now false, and a false reason in a user-facing message
is the thing this campaign throws work away over.

The load-bearing fact, and the one to check first:
`grep -rn "ctr\|Ctr\|CTR" bindings/*.mojo` returns **nothing**. No Python
entry point can construct a `ctr_columns.SimpleCtrConfig`, which is the only
bundle `trainset.Dataset.from_raw` reads (`src/mojotrees/trainset.mojo:726`,
the `ctr` parameter). So "no Python fit enables it" is still true; only the
*explanation* attached to it is stale.

### Site 1.1 -- `sklearn.py:1396-1411`, the `max_ctr_complexity == 1` arm

**Exact current text** (lines 1396-1411, the comment block and the `raise`
that follows the `!= 1` arm):

```python
            # 1 is built and reaches a design matrix, and this still refuses,
            # because the blocker moved rather than cleared. The fitted CTR
            # tables are MODEL STATE -- built from the target -- and the model
            # format has no section for them, so a CTR fit that produced a
            # model would write a file that loads with empty tables, keeps
            # every tree referencing its CTR columns, and bins them as if the
            # feature were absent. Refusing here beats producing a model that
            # loads and scores wrong.
            raise ValueError(
                "max_ctr_complexity=1 is built but no Python fit enables it: "
                "an enabled CTR bundle is refused at the trainer boundary "
                "(ctr.check_ctr_model_support) and at every model writer "
                "(serialize.check_ctr_serializable), because the fitted CTR "
                "tables are model state and the model format carries no "
                "section for them. Follow catalog A29."
            )
```

**Exact replacement text:**

```python
            # 1 is built, reaches a design matrix, trains, saves and loads,
            # and this still refuses -- but for a smaller reason than it used
            # to give. The model-file blocker cleared: serialize.mojo's v5
            # `ctr` section carries the fitted tables, so a CTR model round
            # trips (catalog A31) and trainset.mojo's six trainer refusals are
            # gone. What is missing is the switch. The bundle a dataset takes
            # is ctr_columns.SimpleCtrConfig, a Mojo-API argument to
            # Dataset.from_raw, and bindings/_mojotrees.mojo contains no CTR
            # symbol at all, so nothing here can turn it on.
            raise ValueError(
                "max_ctr_complexity=1 is built and trainable, but no Python "
                "entry point can enable it: the bundle a dataset takes is "
                "ctr_columns.SimpleCtrConfig (a Mojo-API argument to "
                "Dataset.from_raw) and the bindings expose no CTR parameter. "
                "The model file is no longer the blocker -- serialize.mojo "
                "carries the fitted tables as format v5's ctr section, "
                "catalog A31. Two surfaces do still refuse a CTR model and "
                "would have to be answered before wiring this: prepared "
                "tables (ctr_columns.check_ctr_dataset_serializable) and the "
                "model dump (ctr.check_ctr_model_support)."
            )
```

**Traced reason it is now correct**, claim by claim:

- "serialize.mojo's v5 `ctr` section carries the fitted tables" --
  `src/mojotrees/serialize.mojo`, `_write_ctr` and `_read_ctr`; the section
  is written from `save_model` and `save_multiclass_model` and read from
  `_read_mapper`. `CURRENT_FORMAT_VERSION` is 5.
- "trainset.mojo's six trainer refusals are gone" --
  `grep -n check_ctr_model_support src/mojotrees/trainset.mojo` now returns
  only comment lines; the import was removed. The six former call sites are
  `train_dataset`, `train_dataset_multiclass`, `train_dataset_ranker`,
  `train_dataset_ranker_advanced`, `train_dataset_more`,
  `train_dataset_multiclass_more`.
- "the bindings expose no CTR parameter" --
  `grep -rn "ctr\|Ctr\|CTR" bindings/*.mojo` is empty.
- "prepared tables ... still refuse" --
  `ctr_columns.check_ctr_dataset_serializable`, called from
  `serialize.save_dataset`.
- "the model dump ... still refuses" -- `ctr.check_ctr_model_support`, called
  from `model_dump._build`. That is now its only caller.

**Do NOT change** the `max_ctr_complexity != 1` arm above it
(`sklearn.py:1387-1395`). Its reason is unaffected: no grow loop drives
`ctr_combinations.grow_tree_ctr_projections`, which is catalog A30 and is
still true.

### Site 1.2 -- `sklearn.py:346-348`, the class docstring's refusal list

**Exact current text:**

```
    `max_ctr_complexity` (the CTR modules are implemented; nothing imports
    them, so no fit builds a CTR column). None of these is accepted and
    ignored.
```

**Exact replacement text:**

```
    `max_ctr_complexity` (the CTR modules are implemented and a Mojo-API fit
    now builds, trains on, saves and loads CTR columns; the bindings expose
    no parameter that turns them on, so no *Python* fit builds one). None of
    these is accepted and ignored.
```

**Traced reason:** the current parenthetical says "nothing imports them",
which is false at two levels. `src/mojotrees/binning.mojo:314` imports
`ctr_columns`, and `src/mojotrees/trainset.mojo:92` imports it too and calls
`plan_ctr_columns` / `fit_ctr_tables` from `_build_ctr`
(`trainset.mojo:464`). The true statement is the narrower one about the
bindings, which is Site 1.1's fact.

---

## Part 2. Guards for consumers sized by `mapper.n_features`

These are the half-wired paths the lane's own guard (`model_dump._build`)
covers for one consumer and cannot cover for the rest. **Read this part
before deciding whether to apply it**, because it is the one place where
"apply my note" is a judgement call rather than a text substitution.

**The shared cause, stated once.** `BinMapper.n_features` counts base
features. A CTR mapper's *binned* width is `n_total_features()` =
`n_features + ctr.n_columns()` (`src/mojotrees/binning.mojo:2629`). A tree
grown on a CTR dataset may split on any id below the binned width, because
`boosting.train` is handed `dataset.data`, whose `n_features` was set to
`n_base + n_ctr_columns` by `binning.append_ctr_columns`
(`src/mojotrees/binning.mojo:2522`). So any consumer that allocates by
`mapper.n_features` and then indexes by a tree's `feature[i]` reads past its
own array.

Before this lane, no such model could exist, because the trainer refused to
produce one. It can now. Each site below is therefore a **new** reachable
defect, not a pre-existing one, and that is why they are listed here rather
than left for a later audit.

### Site 2.1 -- `src/mojotrees/contrib.mojo:507` and `:520` (highest priority)

Both pass `model.mapper.n_features` as the contribution vector's width while
the trees index it by `feature[i]`. `predict_contrib_bins` allocates
`n_features + 1` and writes `phi[feature]`.

**Exact current text at 506-508:**

```mojo
    return predict_contrib_bins(
        model.booster, model.mapper.bin_row(row), model.mapper.n_features, rng
    )
```

**Exact replacement:**

```mojo
    # Catalog A19/A31. A CTR model's trees split on ids up to
    # `n_total_features() - 1`, while this vector is `n_features + 1` long
    # and is indexed by those ids. Widening it is not the fix it looks like:
    # a CTR column is not a feature of the raw row, so a contribution
    # attributed to one has no name and does not belong beside the others.
    # Refuse until that has an answer.
    if model.mapper.has_ctr():
        raise Error(
            "feature contributions are not defined for a model carrying ctr"
            " columns: the vector is one entry per raw feature and a ctr"
            " column is not one (catalog A19). The model itself predicts and"
            " saves; this attribution does not exist yet"
        )
    return predict_contrib_bins(
        model.booster, model.mapper.bin_row(row), model.mapper.n_features, rng
    )
```

The same three lines, with `predict_contrib_bins_multiclass`, at `:519-521`.

**Check before applying:** confirm `predict_contrib` is still `raises`
(`contrib.mojo:499-501` declares `raises -> List[Float64]`), so the guard adds no
signature change.

### Site 2.2 -- `src/mojotrees/model_editing.mojo:357` and `:391`

Same shape, lower priority: these are refit paths and a CTR refit would also
have to answer what happens to the fitted tables when the rows change. A
guard is the honest placeholder.

### Site 2.3 -- `src/mojotrees/onnx_export.mojo:191` and `:302`

These **already raise** on `feature >= mapper.n_features`
(`onnx_export.mojo:191`), so a CTR model fails rather than corrupting. The
message says "feature id N out of range", which reads as a corrupt model
rather than as an unsupported feature. Text-only fix, no behavior change:
name the cause. Low priority; it is a wrong message on a correct refusal.

### Site 2.4 -- `src/mojotrees/lgbm_model_io.mojo:2459` and `:2501`

`max_feature_idx=n_features - 1` in the LightGBM text export. A CTR model
would export a header that disagrees with its own trees. Same class, same
fix, and it should probably refuse rather than widen, since a CTR column has
no LightGBM counterpart.

### Site 2.5 -- `src/mojotrees/efb.mojo:808`

Already raises on out of range, like 2.3. Message only.

---

## Part 3. One stale cross-reference outside this lane's files

`src/mojotrees/ctr_combinations.mojo:1283` reads:

```
A19's `check_ctr_trainer_support` says the same thing about CTRs at all: no
```

It is comparing A30's refusal to A19's, and A19's no longer says what it
said. `check_ctr_trainer_support` now refuses an enabled `ctr.CtrParams` on
the ground that no trainer reads that bundle
(`src/mojotrees/ctr.mojo:1012`), not on the ground that CTRs are unreachable.
A30's own refusal is unaffected and still correct. Comment-only; whoever owns
`ctr_combinations.mojo` should restate the comparison.

---

## Part 4. What NOT to change, listed so it is not changed by accident

- ~~`python/mojotrees/inspection.py:104`,
  `SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3, 4)`.~~ **SUPERSEDED.** The
  reasoning above stands and became the specification: the tuple now reads
  `(1, 2, 3, 4, 5)` because the parser was taught the sections, not because
  the number was raised. `_parse_ctr` reads the `ctr` section and
  `_parse_usable` reads the v5 `usable` section (the split-search pool,
  `serialize._write_usable`); the `linear` section is **refused by name** in
  `parse_model_string`, since rebuilding linear leaves there would duplicate
  `linear_tree.mojo` and skipping it would describe a linear model as a
  constant-leaf one. `dump_model`'s text fallback separately refuses a CTR
  model by name, mirroring `ctr.check_ctr_model_support`, because the dump
  schema is still sized by `mapper.n_features`. Parsing a section and
  describing it are two claims and only the first was made.
- `src/mojotrees/model_dump.mojo:93`, `MODEL_FORMAT_VERSION = 4`. It is the
  **base** version, not the maximum: a model with neither v5 section still
  writes v4, and `linear_model_format_version` raises it to 5 when the linear
  sidecar is active. It is correct at 4 and the comment above it now says so.
- `sklearn.py:1387-1395`, the `max_ctr_complexity != 1` arm. See Site 1.1.
