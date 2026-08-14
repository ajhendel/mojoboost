# Connect 21: C ABI, CLI, and native inference artifacts

Lane 21. Owned files: `capi/`, `cli/`, `packaging/native/`,
`docs/C_API.md`, `docs/CLI.md`, and this file.

Nothing outside that list was edited. Everything that needs to change
elsewhere is a patch request in section 7, quoted exactly.

**Nothing was compiled, run, built, or tested.** No claim here is a claim
about correctness, performance, parity, packaging, or hardware. The
commands that would establish any of that are in section 11 and are marked
UNRUN. The one exception is section 6, where facts were read off an
already-built binary with `otool`; each such fact says where it came from.

---

## 0. A preservation note that is not about code

While this lane was working, two other lanes committed the whole working
tree — `dc21f03 Connect accelerator and public API foundations` and
`860b1cf Integrate training and interoperability subsystems` — and both
swept this lane's in-progress edits to `capi/` and `cli/` into their
commits. This lane did not commit and did not ask for that.

Verified afterwards with `git show <sha> -- capi/ cli/`: every hunk in both
commits' versions of the owned files is this lane's own text, and no other
lane's content appears in them. Nothing was lost and nothing of anyone
else's was overwritten.

`dc21f03` also captured a stray editor temp file,
`cli/mojoboost_cli.mojo.tmp.60468.6ae06ade28a2`, which `860b1cf` then
deleted. It is gone from `HEAD` and from the working tree. It was an
edit-tool artifact, not a source file, and no reference to it exists
anywhere.

Recorded because the checkout is shared and a whole-tree commit can capture
another lane's half-finished state. **Commit by explicit path.**

---

## 1. Implementations found

Inventory taken before any edit.

| # | Where | What it implements | State found |
| --- | --- | --- | --- |
| 1 | `capi/mojoboost_capi.mojo` | The C ABI: opaque model and error handles, dense train, predict, save/load, three accessors. | Complete for what it covered, and already a genuine forwarding layer for training and serialization. **Its own prediction loop** — see #4. |
| 2 | `capi/mojoboost.h` | The contract, 195 lines, well documented. | ABI version 1. Declared no device, no iteration range, no inspection, no second ownership class. |
| 3 | `cli/mojoboost_cli.mojo` | The `mojoboost` tool: `train`, `predict`, `info`, `version`, `help`, plus a table reader and option parser. | Same shape as #1, including **its own prediction loop** and its own hand-written model summary. |
| 4 | `src/mojoboost/model.mojo` | `Model.predict_batch` / `MulticlassModel.predict_batch`: the authoritative dense prediction path. One binning pass over the whole matrix, `IterationRange` slicing, device dispatch into `gpu_predict.mojo`. | Complete and authoritative. Reached by `bindings/_mojoboost.mojo` and the Python package. **Not reached by #1 or #3.** |
| 5 | `src/mojoboost/inspection.mojo` | `dump_model` / `dump_multiclass_model`: the one implementation of the documented, versioned inspection schema. | Complete. Reached from Python. Not reached by #1 or #3. |
| 6 | `src/mojoboost/device.mojo` over `device_policy.mojo` | The device vocabulary, `resolve_device`, `gpu_available`. | Complete. Reached by #1 and #3 **for training only**, through `device=` in the parameter string. Not reachable at all for prediction. |
| 7 | `src/mojoboost/serialize.mojo` | The model file format, `v3`, with `model_file_kind` for kind detection. | Already the single source for #1 and #3. No duplicate reader or writer anywhere in the owned files. |
| 8 | `src/mojoboost/params.mojo` | `parse_params`, and `params_names_mojo_api_only` for the "real feature, wrong door" distinction. | Already shared by #1 and #3, verbatim. This was already right. |

**There was no second trainer, no second file format, no second parameter
parser, and no second device policy.** The duplication was narrower and
entirely inside the two owned front ends: each had written its own
row-at-a-time prediction loop against the per-row `predict`, and each had
its own private `_row` helper to feed it.

Also found and deliberately left alone: `src/mojoboost/lgbm_model_io.mojo`
(LightGBM text model import and export) and `src/mojoboost/importance.mojo`
are complete and unreachable from both front ends. See section 8.

---

## 2. Call path before and after

### Prediction

Before, in both front ends:

```
C caller / CLI
  -> hand-written loop over rows
       -> _row(features, ...)            (private copy in each front end)
       -> mapper.bin_row(row)            (re-bins one row at a time)
       -> booster.predict_bins(bins)     (whole ensemble, CPU only)
```

After, in both:

```
C caller / CLI
  -> Model.predict_batch(features, n_rows, rng, raw, device)
       -> resolve_device(...)            (device_policy.mojo decides)
       -> mapper.transform(features)     (one binning pass, parallel)
       -> predict_gpu(...)               when the policy resolved to GPU
       -> booster.predict_bins_range(...) otherwise
```

The consequence is not stylistic. Before the change there was **no way** for
a C caller or a CLI user to reach the accelerator or an iteration range for
prediction, and no way to add one without writing a third implementation.
After it, both arrived without a line of prediction logic in `capi/` or
`cli/`.

### Inspection

Before: `mojoboost info` printed five hand-assembled fields. C had nothing.

After: both reach `dump_model` / `dump_multiclass_model`, so the C ABI and
the CLI emit the same documented, versioned schema the Python package does.

### Device

Before: training only, via `device=` in the parameter string. Prediction was
unconditionally CPU with no way to say otherwise.

After: training unchanged, deliberately. Prediction takes a device through
`mojoboost_predict_ex(..., device, ...)` and `--device`, both of which go
through `parse_device` / `resolve_device` rather than any local mapping.

---

## 3. Connections completed

### `capi/mojoboost_capi.mojo`

1. `_predict_into` rewritten to call `_ModelBox.predict_batch`, which
   forwards to `Model.predict_batch` / `MulticlassModel.predict_batch`. The
   per-row loop and the private `_row` helper are gone.
2. `_predict_into` gained `start_iteration`, `num_iteration`, `flags`, and
   `device`. `mojoboost_predict` and `mojoboost_predict_raw` now call it
   with those fixed to the whole ensemble, the CPU, and one flag — see
   section 9 on why.
3. New export `mojoboost_predict_ex`, the full prediction surface.
4. New export `mojoboost_model_num_iterations`. Ranges are expressed in
   boosting iterations, which for a multiclass model differ from the tree
   count by a factor of `num_class`, so exposing the range without exposing
   its unit would have been a trap.
5. New export `mojoboost_gpu_available`, forwarding to `gpu_available` in
   `device.mojo`.
6. New exports `mojoboost_model_dump_json` and `mojoboost_string_free`,
   forwarding to `inspection.mojo`. This adds the ABI's second ownership
   class: a caller-owned `char *` alongside the existing opaque handles.
7. `DEVICE_CPU` / `_GPU` / `_AUTO` are defined as `Int32(CPU_DEVICE)` and so
   on, from the codes in `device.mojo`, rather than written out. The ABI
   cannot drift from the policy module that owns their meaning.
8. `ABI_VERSION` 1 -> 2.

### `capi/mojoboost.h`

9. All of the above declared, each marked "Since ABI version 2".
10. `MOJOBOOST_DEVICE_*` and `MOJOBOOST_PREDICT_*` constants added.
11. The version rule rewritten. It said "incremented only for a breaking
    change", which was unusable: additive releases were invisible to a
    caller loading the library dynamically. It now says versions are
    cumulative, each adds and none removes, so a caller tests
    `mojoboost_abi_version() >= N` for the version that introduced the
    newest symbol it calls. A genuinely breaking change would ship under a
    different library name.

### `cli/mojoboost_cli.mojo`

12. `predictions_text` rewritten onto `predict_batch`; the private `_row`
    helper is gone and the two near-identical branches collapsed onto
    shared `_check_width` and `_format_matrix` helpers.
13. New options `--start-iteration`, `--num-iteration`, `--device`, `--json`.
14. `--device` goes through `parse_device`, so the CLI accepts exactly what
    the parameter string and the Python API accept and rejects the rest with
    the same message.
15. `info --json` emits `dump_model` / `dump_multiclass_model`.
16. `info` now reports the iteration count beside the tree count.
17. The train summary reports the requested device via `device_name`, so
    `device=gpu` in a parameter string is visible in the output.

### Documentation

18. `docs/C_API.md` and `docs/CLI.md` written. Both new files.
19. `capi/README.md` and `cli/README.md` updated for everything above.
20. `packaging/native/` created: `layout.toml` and `README.md`.

---

## 4. Duplicates fused or quarantined

| Duplicate | Disposition |
| --- | --- |
| `_row` in `capi/mojoboost_capi.mojo` | **Removed.** Its only caller was the prediction loop that no longer exists. |
| `_row` in `cli/mojoboost_cli.mojo` | **Removed.** Same. |
| The per-row prediction loop in `capi/` | **Removed**, replaced by one `predict_batch` call. |
| The per-row prediction loop in `cli/`, duplicated across the single-output and multiclass branches | **Removed.** Both branches are now four lines against `predict_batch`. |
| Two copies of the shape check and the output formatter in `cli/predictions_text` | **Fused** into `_check_width` and `_format_matrix`. |
| The hand-assembled model summary in `cli/command_info` | **Kept**, and not a duplicate on inspection: it is a five-line human summary, not a second schema. The schema now comes from `dump_model` under `--json`. Both are documented as what they are. |

Nothing was quarantined. Every duplicate found inside the owned files was
safe to delete outright because each had exactly one caller, in the same
file, that the change replaced.

---

## 5. Bit-exactness of the fused path

The two existing test suites assert exact float equality between a front end
and the Mojo API — `tests/test_capi.mojo` against `reference.predict(row)`,
`tests/test_cli.mojo` against `model.predict(row)`. Switching to
`predict_batch` had to preserve that, so it was checked by reading, not
assumed. **This is a reading of the source, not a test run.**

1. `BinMapper.transform` and `BinMapper.bin_value` perform the identical
   NaN-routing and identical binary search over the identical edge array
   (`binning.mojo:267` and `binning.mojo:315`). Same bin for every value.
2. `predict_raw_bins_range` over a full range is `base_score` followed by
   the same `learning_rate * tree.predict_bins(bins)` accumulation, in the
   same order, as `predict_raw_bins` (`boosting.mojo:859` and `:868`). The
   docstring at `:874` states this explicitly. Same for the multiclass pair
   at `:1402` and `:1425`.
3. `IterationRange.clamp(n, 0, 0)` yields `[0, n)`, since `num <= 0` means
   every remaining iteration, and `includes_base()` is true at start 0.

So the CPU path is the same arithmetic in the same order. **Unverified by
execution.** If any of the three readings is wrong, the failure surfaces as
an exact-equality assertion in one of those two suites, which is the right
place for it to surface.

`transform` stores bins as `UInt8`, so it carries the same 255-bin ceiling
the trainer does. Not a new constraint: a model that could exceed it could
not have been trained.

---

## 6. Native artifact layout

`packaging/native/layout.toml` is the machine-readable layout;
`packaging/native/README.md` is the reasoning. Every status is `designed`,
using the vocabulary already established in
`packaging/matrix/platform_matrix.toml` rather than a second one. **Nothing
was built and no artifact was produced.**

Specified: a GNU-style prefix (`include/mojoboost/mojoboost.h`, `lib/`,
`bin/mojoboost`, `lib/mojoboost-runtime/`, `share/doc/mojoboost/`);
library names carrying the **ABI** version, not the release version
(`libmojoboost.so.2`, `libmojoboost.2.dylib`), because a caller links
against an ABI and two releases sharing one must be swappable without
relinking; loader paths that are `@loader_path` / `$ORIGIN` relative, with
`DYLD_LIBRARY_PATH` and `LD_LIBRARY_PATH` explicitly outside the contract.

### Facts read off the built library

These came from `otool -L` and `otool -l` on `capi/libmojoboost.dylib`, an
artifact already present in the checkout. Reading a binary is inspection;
nothing was built or run to obtain them.

The Mojo runtime closure a macOS artifact must ship, transitively: direct
`libKGENCompilerRTShared.dylib` and `libAsyncRTMojoBindings.dylib`, and
through those `libMSupportGlobals.dylib` and
`libAsyncRTRuntimeGlobals.dylib`. Four libraries, none of them system
libraries, all referenced by `@rpath`. Resolved by the OS and not shipped:
`libSystem`, `libc++`, `libobjc`, and the CoreFoundation, Foundation (weak),
IOKit, CoreGraphics, and Metal frameworks. Metal arrives through the Mojo
runtime whether or not the build has an accelerator.

### Four blocking defects, all in what the build scripts emit today

1. **The install name is a relative build path.** `LC_ID_DYLIB` is
   `capi/libmojoboost.dylib`. Anything linking it records that path as the
   thing to load, so it resolves only when the process runs from the
   directory above `capi/`. Needs
   `install_name_tool -id @rpath/libmojoboost.2.dylib`.
2. **The rpath points into the developer checkout.** The built library
   carries an absolute `LC_RPATH` into `.pixi/envs/default/lib`.
3. **No runtime is staged.** `capi/build.sh` copies none of the four
   libraries above.
4. **No Linux shared library exists in this checkout**, so every Linux row
   in `layout.toml` is a design and its runtime list is expected rather than
   observed.

`capi/run_c_tests.sh` passes today despite 1 and 2 because it adds an
absolute `-rpath` to the build tree for the test binary. That is correct for
an in-tree test and says nothing about whether the artifact is relocatable.

**No staging script was written.** Assembling a tree before defects 1 to 3
are fixed would produce one that does not load, and a script that appeared
to work in the build tree would hide exactly the failure it needs to
expose.

---

## 7. Cross-lane patch requests

None is a blocker; the owned files are consistent without them. Each is
quoted exactly so the owning lane can apply it verbatim.

### 7.1 `README.md` — the C API feature bullet is now incomplete

Line 276-278 reads:

```
- a small stable C ABI (`capi/`) with opaque model handles, dense training,
  prediction, save/load, error retrieval, and destruction, meant as the base
  for bindings in other languages (see [C API](#c-api))
```

Requested replacement:

```
- a small stable C ABI (`capi/`) with opaque model handles, dense training,
  prediction over any slice of the ensemble on either device, model
  inspection as JSON, save/load, error retrieval, and destruction, meant as
  the base for bindings in other languages (see [C API](#c-api))
```

### 7.2 `README.md` — the C API section should point at the new reference

Lines 1731-1733 read:

```
bindings in any language that speaks C. Full documentation, including the
parameter string both it and the CLI take, is in
[capi/README.md](capi/README.md); the header
```

Requested replacement:

```
bindings in any language that speaks C. The reference is
[docs/C_API.md](docs/C_API.md), the practical guide including the
parameter string both it and the CLI take is in
[capi/README.md](capi/README.md), and the header
```

### 7.3 `README.md` — the CLI example should show `--json`

Line 1774 reads:

```
cli/mojoboost info --model model.mbst
```

Requested replacement:

```
cli/mojoboost info --model model.mbst --json
```

and, in the sentence at 1769-1771 that begins "The data format, column
roles, and exit statuses are documented in", the addition of
`[docs/CLI.md](docs/CLI.md)` as the reference beside
`[cli/README.md](cli/README.md)`.

### 7.4 `pixi.toml` — no change requested, recorded so it is not re-derived

`build-capi`, `build-cli`, `test-capi`, `test-cli`, and `test-c` all still
name the right files and need no edit. The `test` aggregate already includes
`tests/test_capi.mojo` and `tests/test_cli.mojo`.

### 7.5 `tools/connectivity_audit.py` — no change requested, and why

`audit_c_abi` compares header declarations to `@export` definitions in both
directions, and `audit_cli` requires every dispatched command to be named by
`docs/CLI.md` or `cli/README.md`. Both were checked against the new state:
the header declares and the implementation exports the same 19 symbols, and
`docs/CLI.md` names `train`, `predict`, `info`, `version`, and `help`, which
is every command `CLI_COMMAND` matches. The audit already carries
`owner="connect_21"` on both checks, so it was written expecting this lane.
**This was checked by reading the regexes and comparing the two symbol sets
by hand; the audit was not run.**

---

## 8. Remaining disconnections

Listed as found, deliberately not closed by this lane.

1. **`lgbm_model_io.mojo` is unreachable from both front ends.** It
   implements LightGBM text model load and save
   (`load_lgbm_model`, `save_lgbm_model`, and the multiclass pair). Neither
   the C ABI nor the CLI can read or write that format. The natural shape is
   a `--format lightgbm|mojoboost` flag on CLI `train`/`predict`/`info` and
   a `mojoboost_load_lgbm_model` in the ABI. Not done here: it is a second
   file format on the public surface, it needs a decision about what the
   default is and what happens on a format mismatch, and it is a larger
   question than "connect what exists".
2. **`importance.mojo` is unreachable from both front ends.**
   `split_importance` and `gain_importance` are not exposed by either. Note
   that gains are not serialized, so a model loaded from a file reports zero
   gain importance — any exposure has to say so at the call site or it will
   read as a bug. The JSON dump partially covers this, since it carries
   per-node split gains when they are present.
3. **`mojoboost_train_dense` cannot express bagging, GOSS, monotone or
   interaction constraints, categorical features, custom objectives, or
   ranking.** This is by design and correctly signalled:
   `params_names_mojo_api_only` turns each into
   `MOJOBOOST_ERROR_UNSUPPORTED` with a message rather than ignoring it.
   Recorded because the list will grow and the mechanism is the thing to
   keep, not the list.
4. **No sparse entry point in either front end.** `fit_csc` /
   `predict_csr` exist and neither is reachable. A CSR triple across the ABI
   is a real design question, not a wiring one.
5. **No incremental or continued training across either front end.** Left
   alone deliberately: exposing a partially covered training mode is worse
   than exposing none.
6. **`bindings/_mojoboost.mojo` still has per-row `predict_raw` and
   `predict_proba` entry points** (`:1275`, `:1295`) beside its
   `predict_batch` ones (`:1700`). Not this lane's file and not a patch
   request, because those may be a deliberate scalar API rather than a
   duplicate. Flagged for the bindings lane to decide.

---

## 9. Fallbacks preserved

- **`mojoboost_predict` and `mojoboost_predict_raw` are unchanged in
  behavior.** They forward to the new body with the whole ensemble, the CPU,
  and one flag fixed. A caller compiled against ABI version 1 sees exactly
  what it saw before. Reaching the accelerator or a range is an explicit
  choice at a new symbol, never something a rebuild changes underneath a
  caller.
- **The CLI defaults to `--device cpu` and the whole ensemble.** Existing
  invocations produce byte-identical output, subject to section 5.
- **`info` without `--json` is the same summary as before**, plus one line.
  The JSON is opt-in.
- **Training device policy is untouched.** Both front ends still read
  `device=` from the parameter string and pass it to `fit`. No `--device`
  was added to `train`, so there remains exactly one place to look.
- **`AUTO_DEVICE` is passed through, not interpreted.** Whatever `auto`
  resolves to is `device_policy.mojo`'s decision, today and after it grows a
  measured crossover.

---

## 10. Serialization and public-API effects

**No serialization change.** The model file format is untouched; this lane
edited nothing under `src/`. Both front ends read and write through
`serialize.mojo` exactly as before, and `model_file_kind` still decides
single-output versus multiclass.

**No new model state needs serializing.** Everything added reads existing
state: iteration counts come from the ensemble, the dump comes from the
mapper and the trees, and the device is a per-call argument that is never
stored in a model.

Public API effects:

- **ABI 1 -> 2, additive.** Five new functions, five new constants, no
  signature changed and none removed.
- **The version rule itself changed meaning**, from "breaking changes only"
  to cumulative. This is the one thing a downstream reader has to notice,
  and it is stated in the header, `capi/README.md`, and `docs/C_API.md`.
  Under the old rule a caller had no way to detect an additive release at
  all, so nothing that worked before stops working.
- **A second ownership class in C**: `mojoboost_model_dump_json` returns a
  caller-owned `char *` released by `mojoboost_string_free`, which is not
  interchangeable with the error-object-owned pointer from
  `mojoboost_error_message`. Documented in both tables.
- **Four new CLI flags**, all with defaults that reproduce the previous
  behavior.
- **`docs/C_API.md` and `docs/CLI.md` are new public documents.**
- **The `mojoboost train` summary line gained `, device <name>`.** The only
  output-format change in this lane. It is a human summary, not a parsed
  format, and the substrings the existing tests match on are unaffected.

---

## 11. Risks

1. **Nothing was compiled.** The Mojo changes are unverified. The most
   likely failure is a signature or convention mismatch in the new
   `_new_c_string` / `mojoboost_string_free` pair, which is the only place
   this lane allocates raw bytes rather than a typed handle.
2. **`comptime DEVICE_CPU: Int32 = Int32(CPU_DEVICE)`** relies on
   `CPU_DEVICE` being usable in a comptime `Int32(...)`. Written with the
   explicit conversion rather than a bare assignment for exactly that
   reason, but unverified.
3. **The bit-exactness argument in section 5 is a reading.** If it is wrong,
   `tests/test_capi.mojo` and `tests/test_cli.mojo` fail on exact equality.
   That is the intended detector and it already exists.
4. **`Optional.value()` is called on `self` in the new `_ModelBox` methods**
   the same way the existing accessors do it. Consistent with what is
   already there, still unverified.
5. **`mojoboost_gpu_available` is declared non-raising**, on the reading
   that `gpu_available` in `device.mojo` is not marked `raises`. If that
   changes upstream, this function stops compiling — which is the correct
   failure, since a raising call cannot be allowed to unwind into C.
6. **The GPU prediction path has never run from either front end.** It is
   now reachable, which is a change in exposure. The policy still refuses
   what it always refused, and `MOJOBOOST_DEVICE_CPU` remains the default
   everywhere, so nothing reaches it without being asked to.
7. **Every native packaging claim is a design.** No artifact was built, no
   tree was staged, and the Linux runtime closure is expected rather than
   observed.

---

## 12. Smallest later commands, all UNRUN

Ordered so the cheapest thing that can fail runs first. **None of these was
run by this lane.**

```sh
# 1. Does the C ABI still compile? Nothing else matters until it does.
pixi run build-capi

# 2. Does the CLI still compile?
pixi run build-cli

# 3. The one test that would catch a bit-exactness regression in the ABI.
pixi run test-capi

# 4. The same for the CLI.
pixi run test-cli

# 5. The C side: the header compiles as C99 and the lifecycle holds.
pixi run test-c

# 6. Structural: header declarations against exports, CLI commands against
#    docs. Both checks already name this lane as owner.
pixi run python tools/connectivity_audit.py
```

Then, and only for the packaging work in section 6, which nothing above
touches:

```sh
# 7. Confirm defect 1 and 2 are still real before fixing them.
otool -l capi/libmojoboost.dylib | grep -A2 -E 'LC_ID_DYLIB|LC_RPATH'

# 8. The Linux runtime closure, which layout.toml currently guesses.
#    Requires a Linux .so that does not exist yet.
readelf -d capi/libmojoboost.so
```

No commit was made by this lane.
