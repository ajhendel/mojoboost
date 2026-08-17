# Three switch contracts, repaired

Written 2026-08-17 by the lane that owns `src/mojotrees/gpu_tree_tables.mojo`
and `src/mojotrees/device_policy.mojo`. Everything below was established by
reading the call graph at head. **Nothing here was measured and nothing here
was compiled.** Where a fix lands in a file this lane may not write, the edit
is quoted in full so the owning lane can apply it verbatim.

Companion to `docs/design/SWITCH_GRID.md`, which is the census. This file is
the resolution of three of its findings, sections 6, 7B and 7C.

---

## 1. `MOJOTREES_GPU_TREE_RESIDENT`, one variable read three ways

### What was there

Three predicates over one name, each with its own `getenv`.

| predicate | spelling | unset means | correct? |
| --- | --- | --- | --- |
| `gpu_resident_round.resident_round_enabled` | `!= "0"` | on | yes, this is the gate |
| `gpu_resident_round.resident_round_explicitly_requested` | `== "1"` | not requested | yes, a different question |
| `gpu_tree_tables.tree_resident_requested` | `== "1"` | off | **no** |

The first two ask genuinely different questions and both are right. The third
asked the gate's question with the diagnostic's spelling, so it reported the
pre 2026-08-16 default. Two predicates over one variable disagreed about that
variable's default and nothing would have failed if a caller reached for the
wrong one. `resident_round_enabled`'s own docstring flagged it as a live
hazard and asked whoever owned `gpu_tree_tables.mojo` to fix it.

### Which default the shipped behavior actually has

**On.** Established from the call graph rather than from the docstrings, which
disagreed.

- `train_gpu.mojo:2112` gates the device owned growth plane on
  `not resident_frontier_disabled()`, and `train_gpu.mojo:2121` states in a
  comment that the plane is the default.
- `gpu_resident_round.resident_round_enabled` at line 724 is `!= "0"`.
- `tests/test_gpu_resident_gate.mojo` asserts that polarity on four values and
  is marked `cpu-safe`, so it runs on both halves of the CI matrix.
- The evidence for the flip is rule S1 in `bench/results/PROFILE_PROTOCOL.md`
  answered by `bench/results/session3_2026-08-16/RESULTS.md`.

Nothing consulted `tree_resident_requested`, which is exactly why the default
could move in one module and not the other with no test failing anywhere.

### What changed

The single source of truth now lives in `gpu_tree_tables.mojo`, which is the
**lower** layer. That direction is forced rather than chosen.
`gpu_resident_round.mojo` imports `gpu_tree_tables` (its import block at line
623 names `DeviceTreeTables` among others), so `gpu_tree_tables` cannot import
`gpu_resident_round` back without a cycle. A predicate both layers call has
exactly one cycle free home and this is it.

Added to `gpu_tree_tables.mojo`.

- `comptime TREE_RESIDENT_VAR = "MOJOTREES_GPU_TREE_RESIDENT"`, so the string
  literal has one home too. A rename that reached one of two literals is the
  same defect one level down.
- `tree_resident_enabled()`, `getenv(TREE_RESIDENT_VAR) != "0"`. The gate.
- `tree_resident_explicitly_requested()`, `getenv(TREE_RESIDENT_VAR) == "1"`.
  The diagnostic, documented as never a routing input.
- `tree_resident_requested()` kept as a body free alias that delegates to
  `tree_resident_explicitly_requested`. It reads no variable of its own, so it
  can no longer disagree with anything. It survives only because
  `tests/test_gpu_tree_tables.mojo:106` imports it and `:581` calls it, and
  this lane may not edit tests. The assertion there is
  `assert_false(tree_resident_requested())` with the variable unset, which
  stays true through the change.

Also corrected in the same file, because both statements were load bearing and
both were false at head.

- The module docstring said "Nothing in `train_gpu.mojo` calls anything here,
  `MOJOTREES_GPU_TREE_RESIDENT` is off by default, and no shipping fit changes
  in any way." All three clauses are now wrong.
  `histogram_gpu.GpuHistogramBuilder.open_resident_tables` constructs
  `DeviceTreeTables` and is called from `train_gpu.mojo:1975` and
  `train_gpu.mojo:2150`.
- `DeviceTreeTables`'s docstring named `tree_resident_requested` as the gate.
  It now names `tree_resident_enabled`.

### The edit the other lane needs

In `src/mojotrees/gpu_resident_round.mojo`, add to the existing
`from .gpu_tree_tables import (...)` block at line 623.

```
    tree_resident_enabled,
    tree_resident_explicitly_requested,
```

Then replace the body at line 724.

```
    return getenv("MOJOTREES_GPU_TREE_RESIDENT") != "0"
```

with

```
    return tree_resident_enabled()
```

and the body at line 1883.

```
    return getenv("MOJOTREES_GPU_TREE_RESIDENT") == "1"
```

with

```
    return tree_resident_explicitly_requested()
```

Keep both docstrings, and replace their "stale duplicate, which is a live
hazard" section with a pointer here. After that there are still three function
bodies but only two `getenv` calls in the package, and a delegation cannot
drift from what it delegates to.

If `getenv` then has no other user in `gpu_resident_round.mojo`, the import may
need trimming. It does have others (`RESIDENT_TRACE_VAR` and the speculation
variables are read there), so no trim is expected.

### Other switches in the two owned files

Checked, and none of them is read twice anywhere in `src/`, `bench/`, `cli/`,
`python/`, `capi/`, `bindings/` or `tools/`.

| switch | reader | spelling | consistent with the convention |
| --- | --- | --- | --- |
| `MOJOTREES_GPU_TABLE_RESET` | `gpu_tree_tables.mojo:2830` | `!= "0"` | yes, default on |
| `MOJOTREES_GPU_PACKED_DOWNLOAD` | `gpu_tree_tables.mojo:2831` | `!= "0"` | yes, default on |
| `MOJOTREES_DISABLE_GPU` | `device_policy.mojo:905` | `== "1"` | yes, default off |
| `MOJOTREES_AUTO_MIN_CELLS` | `device_policy.mojo:965` | value parse | not a Boolean switch |
| `MOJOTREES_GPU_BACKEND` | `device_policy.mojo:1072` | value parse | not a Boolean switch |

One near miss worth naming rather than fixing. `bench/bench_train_gpu.mojo`
around line 1009 re-spells five of these names as string literals to build the
provenance label. That is a second copy of each **name**, not a second copy of
each **decision**, and it is deliberate, because the label has to report what
the environment said even when nothing read it. It belongs to the bench lane.

---

## 2. Two switches that raise where they should not, or raise badly

The deciding question for each is not "is a raise defensible" but "could
silently ignoring the request make somebody publish a number for an arm that
never ran". The two switches answer that question differently, so they get
different treatments.

`bench/bench_train_gpu.mojo:1009-1010` is what makes the question sharp. It
builds the provenance label from `_env_word("MOJOTREES_GPU_TREE_RESIDENT")` and
`_env_word("MOJOTREES_GPU_SPLIT_RESIDENT")`, which is to say **from the
environment and not from what ran**. A silent degrade therefore does not merely
lose information, it mislabels the record.

### 2A. `MOJOTREES_GPU_SPECULATION` under `grow_policy=oblivious`. Degrade, and record it.

Ignoring this request cannot make a symmetric measurement wrong, because there
is no symmetric arm for it to have selected. `OBLIVIOUS_SPECULATION`'s own
docstring says so. "The speculation predicts which single leaf a leaf-wise pick
will take next, and an oblivious level takes every leaf. There is nothing to
speculate about." A symmetric run with the variable exported and a symmetric run
without it are the same arm, not two arms, so no number can be published for an
arm that never ran.

The correctness hazard the refusal cites is a hazard of **arming** speculation,
not of declining it. `_pick_runner_up_kernel` publishes a live leaf whose
histogram a batched level plan would overwrite. Declining is the safe side of
that, and the symmetric grower is already on it structurally. Verified by
reading `grow_tree_device_oblivious` end to end, from
`gpu_resident_round.mojo:3097` to the end of the function. The **only**
occurrence of `speculative_build_enabled` inside it is the raise itself. The
loop never reads a `spec` flag, never calls the runner up kernel, and never
stages a speculative build. The three real consumption sites are all in
`grow_tree_device_resident`, a different function, reached through
`var spec = speculative_build_enabled()` at line 2471.

So removing the refusal cannot enable the hazard, and keeping it costs every
symmetric arm in any shell where a leaf-wise arm exported the variable.

Recording the decline matters anyway, because an operator who typed
`MOJOTREES_GPU_SPECULATION=1` did ask by hand, and the package already has a
template for exactly that case in `resident_round_report_refusal`. Two sinks,
standard output only for the caller who asked by hand, and the trace sink
always.

**Edit 1**, `src/mojotrees/gpu_resident_round.mojo`, in
`oblivious_device_supported`, delete lines 1812 and 1813.

```
    if speculative_build_enabled():
        return OBLIVIOUS_SPECULATION
```

**This edit has a test attached and the test must move with it.**
`tests/test_gpu_oblivious_device.mojo:1119`,
`test_the_speculation_and_a_level_build_refuse_to_combine`, sets
`MOJOTREES_GPU_SPECULATION=1` and asserts
`oblivious_device_supported(...) == OBLIVIOUS_SPECULATION` at line 1142. After
Edit 1 that predicate returns `OBLIVIOUS_OK`. The test should be rewritten to
assert the new contract, which is that the predicate stays `OBLIVIOUS_OK`, that
the fit still grows, and that the trace records `speculation=declined`. That
last assertion is the one worth having, because it is what makes the decline
non-silent, and the file already reads the trace at line 878. Keep the test's
existing comment about `_pick_runner_up_kernel`, since the hazard it describes
is still real and is still the reason the symmetric loop never arms the
speculation. `test_every_refusal_names_itself` at line 212 keeps
`OBLIVIOUS_SPECULATION` in its list and needs no change, provided Edit 6 keeps
the constant and its `oblivious_reason_name` arm.

**Edit 2**, same file, in `grow_tree_device_oblivious`, delete the raise at
lines 3233 to 3240.

```
    if speculative_build_enabled():
        raise Error(
            "the K=1 speculative prebuild and a batched level build must not"
            " combine: the runner-up kernel publishes a leaf that is still"
            " live, and a plan that builds both of its children overwrites the"
            " histogram the next pick reads on a miss. See"
            " OBLIVIOUS_SPECULATION"
        )
```

**Edit 3**, same file, add beside `resident_round_report_refusal` at line 1886.

```
def oblivious_report_speculation_declined() raises:
    """Say once per fit that a symmetric tree declined the K=1 speculation.

    Modeled on `resident_round_report_refusal`, two sinks and the same rule
    for each. Standard output only when the operator set
    `MOJOTREES_GPU_SPECULATION=1` by hand, which is always true when this is
    called, so the print is unconditional here and the shape is kept so the
    two functions stay readable against each other. The trace sink whenever
    `MOJOTREES_GPU_TREE_RESIDENT_TRACE` names one.

    Why a decline rather than the refusal that stood here. The switch has no
    symmetric arm to select. `OBLIVIOUS_SPECULATION` says the speculation
    predicts which single leaf a leaf-wise pick takes next and an oblivious
    level takes every leaf, so a symmetric fit with this set and a symmetric
    fit without it are the same arm. Raising cost every symmetric arm in any
    shell that exported the variable for a leaf-wise arm, and bought nothing,
    because `grow_tree_device_oblivious` never arms the speculation in the
    first place. What a decline must not do is disappear, which is what the
    two sinks are for.

    The line deliberately does not carry `OBLIVIOUS_TRACE_MARK`, for the
    reason `resident_round_report_refusal` does not carry
    `plane=device-resident`. That token is what
    `tests/test_gpu_oblivious_device.mojo:878` counts to prove the plane ran,
    and a decline writing it would turn a positive control into a test that
    passes when the plane refused.
    """
    print(
        "MOJOTREES_GPU_SPECULATION does not apply to grow_policy=oblivious;"
        " a level splits every leaf, so there is no next pick to speculate"
        " about. The symmetric plane ran with speculation off"
    )
    _resident_trace_emit(
        resident_trace_sink(),
        String(
            "mojotrees.oblivious speculation declined:"
            " MOJOTREES_GPU_SPECULATION does not apply to"
            " grow_policy=oblivious\n"
        ),
    )
```

**Edit 4**, same file, in `grow_tree_device_oblivious`, where the raise was.

```
    if speculative_build_enabled():
        oblivious_report_speculation_declined()
```

**Edit 5**, same file, in the `_resident_trace_emit` call at line 3640, so the
per tree record carries the arm as well. Insert after the `levels=` field.

```
            " speculation=",
            "declined" if speculative_build_enabled() else "off",
```

**Edit 6**, same file, rewrite the `OBLIVIOUS_SPECULATION` docstring at line
1697 to say the code is retained as the name of a recorded decline rather than
of a refusal, and keep its arm in `oblivious_reason_name`. Leaving the constant
in place with a stale docstring reproduces the defect this whole file is about.

### 2B. `MOJOTREES_GPU_SPLIT_RESIDENT=0` under `grow_policy=oblivious`. Keep the raise, fix the message.

The opposite call, and the asymmetry is the point.

On the leaf-wise and depth-wise planes the `0` selects
`_device_search_incremental`, a real second arm that produces a correct tree.
On the symmetric plane there is no second arm. The resident frontier is the
only device grower for a symmetric tree, so the request cannot be honored.
Degrading would run the resident frontier anyway, which is precisely the arm
the operator asked not to run, and the harness would then label that timing
`split_resident=off` from `_env_word`. That is a published number for an arm
that never ran, so the raise stays.

What is wrong is the message. `resident_frontier_disabled()` at
`train_gpu.mojo:1593` folds into `opened` at `train_gpu.mojo:1972`, `opened`
false sends `_oblivious_route_reason` to `RESIDENT_NO_POOL` at
`train_gpu.mojo:1809-1810`, and the raise then reports a pool failure and
offers a remedy list about `max_depth`, categorical features and constraints.
None of those is the cause and none of them will help.

**Edit 7**, `src/mojotrees/train_gpu.mojo`, in `_grow_tree_gpu_device_search`.
Replace lines 1970 to 1976.

```
        var budget = oblivious_leaf_budget(params)
        var opened = (
            not resident_frontier_disabled()
            and budget >= 2
            and builder.open_resident(budget, OBLIVIOUS_MAX_ITEMS)
            and builder.open_resident_tables(budget)
        )
```

with

```
        var budget = oblivious_leaf_budget(params)
        # MOJOTREES_GPU_SPLIT_RESIDENT=0 IS REFUSED BY NAME UNDER THIS POLICY,
        # ahead of the pool, because folding it into `opened` reported it as a
        # pool failure and handed the caller a remedy list about max_depth and
        # categorical features. It is neither.
        #
        # The refusal is kept rather than degraded, and the asymmetry with the
        # leaf-wise plane is deliberate. There the `0` selects
        # `_device_search_incremental`, a real second arm. Here there is no
        # second arm: the resident frontier is the only device grower for a
        # symmetric tree. Degrading would run the very arm the operator asked
        # not to run, and `bench/bench_train_gpu.mojo:1010` labels the record
        # `split_resident=` out of the environment rather than out of what ran,
        # so the timing would be published under the wrong arm's name.
        if resident_frontier_disabled():
            raise Error(
                "MOJOTREES_GPU_SPLIT_RESIDENT=0 does not apply to"
                " grow_policy=oblivious. It selects the incremental"
                " split-search loop, and that loop grows leaf-wise trees only,"
                " so a symmetric tree has no arm on this backend to fall back"
                " to and the request cannot be honored or silently ignored."
                " Unset it for this fit, which is the default, or keep it set"
                " and pass device='cpu', which grows the same symmetric tree."
                " On grow_policy=leafwise and depthwise the same 0 is a benign"
                " fallback, which is how one shell carries it into a symmetric"
                " arm without anybody noticing"
            )
        var opened = (
            budget >= 2
            and builder.open_resident(budget, OBLIVIOUS_MAX_ITEMS)
            and builder.open_resident_tables(budget)
        )
```

`resident_frontier_disabled` keeps its other caller at `train_gpu.mojo:2112`
unchanged, which is the leaf-wise gate where the `0` remains benign.

---

## 3. Is the `random_strength` device path reached? No, and both sides of the
argument were wrong about why

> **SUPERSEDED 2026-08-17, LATER THE SAME DAY. THE ANSWER IS NOW YES, AND
> EDITS 8 THROUGH 11 BELOW MUST NOT BE APPLIED.**
>
> This section's finding rested on `params.mojo` supplying CatBoost mode's
> `random_strength = 1.0` only under `config.device == CPU_DEVICE`. That
> condition was removed later the same day, as a bug and with no switch,
> because it made one parameter string build two different models. The write is
> now `if not saw_random_strength and not config.is_multiclass():` at
> `params.mojo:1755-1756`, so a defaulted CatBoost-mode fit gets 1.0 on either
> backend and the shipped symmetric GPU fit makes **six** `_copy_noise` drains
> per tree, not zero. `docs/design/RANDOM_STRENGTH_UNITS.md` section 2 is the
> record of the removal, including the four surfaces that had already retired
> the identical device test and left this one behind.
>
> What survives from this section, and it is the larger half. The subsection
> below headed "The mechanism the comments give is false" is untouched and was
> the real defect. The two `train_gpu.mojo`
> comments claiming the call site was **NOT REACHED BY ANY FIT TODAY** gave a
> mechanism that was already false, `_check_device_search_supported` refusing
> `ExtraTreeParams.is_active()`, and that mechanism has been replaced in the
> file with the traced call graph. The census's finding is intact and its
> paragraph now records this whole episode in place. Edit 12, deleting the
> stale "refused here" paragraph from `gpu_resident_round.mojo`, was correct
> for its own reasons and has been applied.
>
> Kept rather than deleted because the reasoning chain is what the next reader
> needs in order to not re-derive it, and because a section that quietly turned
> into its own opposite is exactly what LANE_RULES rule 7 is about. Read
> everything below as the state of the source between 2026-08-16 and the
> afternoon of 2026-08-17.

### The contradiction

Two comments in `train_gpu.mojo`, at lines 1911 and 1931, say the
`set_random_score` call at line 1942 is **NOT REACHED BY ANY FIT TODAY**, and
both give the same mechanism.

> `_check_device_search_supported` above refuses `params.extra.is_active()`
> and `random_strength > 0.0` is one of its arms, so this line is dead

`_device_search_unsupported_reason` in the same file, at lines 491 to 494, says
the refusal is retired. `gpu_resident_round.mojo` at lines 3292 to 3294 goes
further and asserts the call site is reached.

> The scale reaches this searcher before this line runs.
> `_grow_tree_gpu_device_search` calls `set_random_score` ahead of the
> oblivious route decision

### The mechanism the comments give is false

`_check_device_search_supported` does not call `is_active()`. It calls
`_device_search_unsupported_reason` at `train_gpu.mojo:1372`, which delegates
at line 521 to `ExtraTreeParams.device_unsupported_reason`. That function, at
`tree_parameters_extra.mojo:2031`, refuses `random_strength` only in one case.

```
        if has_categorical and self.random_strength > 0.0:
            return String("random_strength beside a categorical feature")
```

There is no unconditional arm. The comment at
`tree_parameters_extra.mojo:2025-2030` records that the allowance was
reinstated on 2026-08-17 on measured device movement, after an earlier
retirement was withdrawn for producing a bit identical fit. So a fit with
`random_strength=1.0`, `device='gpu'` and no categorical column is **not**
refused, and it does reach `train_gpu.mojo:1942`. The comments' stated reason
is wrong and `_device_search_unsupported_reason` is right.

### The conclusion is nonetheless correct, for a reason nobody wrote down

A **default** CatBoost-mode fit on the GPU has `random_strength = 0.0`.

`params.mojo:1698-1703`, the only caller being `params.mojo:1569`.

```
    if (
        not saw_random_strength
        and config.device == CPU_DEVICE
        and not config.is_multiclass()
    ):
        config.booster.tree.extra.random_strength = CATBOOST_RANDOM_STRENGTH
```

The CatBoost inheritance is declined on anything that is not `CPU_DEVICE`, so a
GPU fit keeps `ExtraTreeParams.__init__`'s `0.0` at
`tree_parameters_extra.mojo:1830`. The docstring at `params.mojo:1659-1664`
states the reason and calls it out as a decline. Then
`random_score_stdev()` is `random_strength * random_score_scale`, which is
`0.0`, so the guard at `train_gpu.mojo:1942` is false and `set_random_score`
never runs.

### The answer

**No.** The `random_strength` device path is not reached by a default
CatBoost-mode fit today. It is reached by a fit that names
`random_strength` explicitly with `device='gpu'` and no categorical column, and
that fit is not refused anywhere.

Proof chain, all verified at head.

1. `params.mojo:1698-1703`. The CatBoost-mode `random_strength = 1.0` is
   written only when `config.device == CPU_DEVICE`.
2. `tree_parameters_extra.mojo:1830`. The field's own default is `0.0`.
3. `tree_parameters_extra.mojo:2046-2048`. `random_score_stdev()` is
   `random_strength * random_score_scale`, so it is `0.0`.
4. `train_gpu.mojo:1942`. `if params.extra.random_score_stdev() > 0.0:` is
   false, so `set_random_score` is not called.
5. `gpu_split_search.mojo:8100-8101`. `_copy_noise` returns before its
   `enqueue_copy` when `not (self.noise_stdev > 0.0)`.
6. `gpu_split_search.mojo:8006`. `_launch_oblivious_search` is passed
   `self.noise_stdev > 0.0`, which is false, so the no noise arm runs.
7. `tree_parameters_extra.mojo:2031`. The refusal is conditional on
   `has_categorical`, so the gate the comments blame is genuinely retired.

### What this does to `docs/design/OBLIVIOUS_WAIT_CENSUS.md`

The finding at line 111, "six drains per tree that are not in anyone's count",
rests on one sentence at line 119.

> **The CatBoost-mode default set does set it.** `params.mojo:1703` writes
> `CATBOOST_RANDOM_STRENGTH`, 1.0, whenever the user did not name
> `random_strength`

That sentence drops the `config.device == CPU_DEVICE` condition that sits two
lines above the line it cites. A CatBoost-mode fit on the GPU does not set it,
so the **shipped symmetric GPU fit makes zero of these drains, not six.**

The rest of the census survives intact and is internally consistent with the
correction already. Its own table at line 70 reads "1 when
`random_strength > 0`" and its totals at lines 104 and 105 read "copies 4 with
`random_strength = 0`, 10 with `random_strength > 0`". Only the sentence that
picks which of those two rows the default lands on is wrong, and it picks the
wrong one.

The finding therefore does not disappear. It narrows, from a property of the
shipped default to a property of a fit that names `random_strength` on the GPU,
and it loses its claim on the 76.5 percent idle profile, which was taken on a
default fit.

**Edit 8**, `docs/design/OBLIVIOUS_WAIT_CENSUS.md`, replace lines 119 to 123.

```
**The CatBoost-mode default set does set it.** `params.mojo:1703` writes
`CATBOOST_RANDOM_STRENGTH`, 1.0, whenever the user did not name
`random_strength`, and `train_gpu._device_search_unsupported_reason` stopped
refusing `random_strength` on the device search when the noise plane was
wired. So the shipped symmetric fit makes six of these per tree.
```

with

```
**The CatBoost-mode default set does NOT set it on the GPU, and this
paragraph said the opposite until 2026-08-17.** `params.mojo:1703` writes
`CATBOOST_RANDOM_STRENGTH`, 1.0, only under the condition two lines above it,
`config.device == CPU_DEVICE`, so a GPU fit keeps `ExtraTreeParams`'s own 0.0.
The shipped symmetric GPU fit therefore makes **zero** of these drains per
tree, and this section's own totals already say so at "copies 4 with
random_strength = 0".

The finding narrows rather than disappearing. What is retired is the refusal,
not the drain: `ExtraTreeParams.device_unsupported_reason` refuses
`random_strength` only beside a categorical column, so a fit that names
`random_strength` explicitly with `device='gpu'` reaches
`train_gpu.mojo:1942`, arms the plane and makes six drains per tree. That is a
real and reachable configuration and the count below holds for it. It is not
the default, and the 76.5 percent idle profile was taken on a default fit, so
that profile is no longer evidence for this shape.
```

**Edit 9**, `src/mojotrees/gpu_resident_round.mojo`, lines 1029 to 1031, the
`OBLIVIOUS_NOISE_HOIST_VAR` docstring, carries the same error.

```
shipped level loop calls it once per level, so a depth-6 tree whose fit sets
`random_strength` makes six of them, and the CatBoost-mode default set does
set it (`params.CATBOOST_RANDOM_STRENGTH` is 1.0). Each one sits between the
```

Replace with.

```
shipped level loop calls it once per level, so a depth-6 tree whose fit sets
`random_strength` makes six of them. The CatBoost-mode default set does NOT
set it on the GPU: `params.mojo:1703` is conditioned on
`config.device == CPU_DEVICE` two lines above, so this arm is reachable only
by a fit that names `random_strength` explicitly. Each one sits between the
```

**Edit 10**, same file, lines 3172 to 3174, the same error a third time.

```
    not enqueued yet. The CatBoost-mode default set turns `random_strength`
    on, so the shipped symmetric fit makes six of them per tree.
```

Replace with.

```
    not enqueued yet. The CatBoost-mode default set does NOT turn
    `random_strength` on for a GPU fit (`params.mojo:1703` is conditioned on
    `config.device == CPU_DEVICE`), so the shipped symmetric fit makes none of
    them and a fit that names `random_strength` makes six per tree.
```

**Edit 11**, `src/mojotrees/train_gpu.mojo`, lines 1911 to 1941. The whole
block states a mechanism that is false. Replace it with.

```
    # **NOT REACHED BY A DEFAULT CatBoost-MODE GPU FIT, AND THE REASON IS NOT
    # THE ONE THIS COMMENT GAVE UNTIL 2026-08-17.**
    #
    # It used to say `_check_device_search_supported` above refuses
    # `params.extra.is_active()`. That is false at head and has been since the
    # gate split. `_check_device_search_supported` calls
    # `_device_search_unsupported_reason`, which delegates to
    # `ExtraTreeParams.device_unsupported_reason`, which refuses
    # `random_strength` only beside a categorical column. The gate is genuinely
    # retired and this line is genuinely reachable: a fit with
    # `random_strength=1.0`, `device='gpu'` and no categorical column arrives
    # here with a positive standard deviation and arms the plane.
    #
    # What keeps a DEFAULT fit off this line is a different mechanism entirely,
    # in a different file. `params.mojo:1698-1703` writes CatBoost's
    # `random_strength = 1.0` only when `config.device == CPU_DEVICE`, so a GPU
    # fit keeps `ExtraTreeParams`'s own 0.0, `random_score_stdev()` is 0.0, and
    # the guard below is false. The plane is wired, the scale is computed on
    # both arms of `_train_gpu_rounds`, and the default simply does not ask for
    # it. See `docs/design/SWITCH_CONTRACT_REPAIRS.md` section 3.
    #
    # `ExtraTreeParams.check_scalars` still says "the device loops do not
    # compute it and this refusal is correct for them". That sentence is false
    # and is the last copy of the old story still standing.
```

**Edit 12**, `src/mojotrees/gpu_resident_round.mojo`, lines 3241 to 3270. A
stale refusal paragraph headed "random_strength, refused here and wired on the
leaf-wise plane" sits immediately **above** the corrected paragraph at line
3271 that retires it. The two contradict each other line for line, and the
stale one states that `_launch_oblivious_search` "takes no noise plane at all",
which `gpu_split_search.mojo:7977` and `:8006` disprove. Delete the stale
paragraph outright; the block at 3271 already says everything it needs to.

---

## 4. What could not be verified without a compiler

Ranked, most likely to bite first.

1. Whether `gpu_resident_round.mojo` importing two more names from
   `gpu_tree_tables.mojo` compiles. The edge already exists so no new cycle is
   created, but the import block is long and Mojo 1.0 is strict about the
   syntax.
2. Whether `comptime TREE_RESIDENT_VAR` is accepted as an argument to `getenv`
   in this Mojo version. Every other switch name in the package is a
   `comptime` used the same way, `RESIDENT_TRACE_VAR` and
   `SPECULATION_BUILD_VAR` among them, so the pattern is established, but this
   lane did not compile it.
3. Whether deleting the two speculation raises leaves `OBLIVIOUS_SPECULATION`
   and `speculative_build_enabled` with live users in
   `gpu_resident_round.mojo`. Both keep at least one, the constant through
   `oblivious_reason_name` and the predicate through the new decline report and
   the trace field, so neither should go unused. Not compiled.
4. Whether any test asserts the raise in 2B, and whether the 2A test rewrite
   is complete. One test asserting 2A was found and is specified above
   (`tests/test_gpu_oblivious_device.mojo:1119`). No test was found asserting
   the 2B raise, by grep on `SPLIT_RESIDENT` and on the message text, but a
   test could match on a substring this grep did not guess.
5. Whether `tests/test_gpu_tree_tables.mojo` compiles unchanged against the
   alias. The signature and the return type are identical and the assertion's
   truth value is unchanged, so it should, but the file was not compiled.
6. Every timing claim quoted here. All of it is read from existing artifacts.
   This lane ran no benchmark, no test, and no fit.
