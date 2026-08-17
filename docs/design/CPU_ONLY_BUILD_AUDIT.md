# The CPU-only build audit

`constraint failed: Unknown GPU architecture detected`

That is a COMPILE error on a build with no accelerator, and it is the one
class of defect this project's development machine physically cannot see. An
Apple M4 has an accelerator, so it compiles every GPU kernel it can reach and
never produces the message. CI's x86-64 and ARM64 Linux runners have none, so
they produce it for any GPU kernel the compiler can reach, whatever the kernel
does. The compiler prints ONE error stack and stops.

Those two facts together are expensive. On 2026-08-17 a single symptom took
four CI cycles and three separate fixes, because each round of guarding
revealed only the next site rather than proving the set was complete. This
page ends that by enumeration. It records the three mechanisms, the complete
site table, what was fixed, what is knowingly left, and the checklist and gate
that let a contributor on an accelerator machine catch this class before
pushing.

A green local build on a machine with a GPU proves NOTHING about this class.
Say that out loud before trusting one.

## 1. The three mechanisms

### Mechanism 1. The guard is a `comptime if` with an `else`, not an early return

```mojo
comptime if not has_accelerator():
    raise Error("... needs an accelerator; this build has none")
else:
    <the body that touches the GPU>
```

and the positive spelling, used where the CPU has a real answer rather than a
refusal, as in `device_policy.build_has_accelerator`

```mojo
comptime if has_accelerator():
    <the body that touches the GPU>
else:
    <the CPU answer>
```

An early `return` does NOT prune. The `comptime if` paired with an `else` is
what removes the branch at compile time, and it prunes wherever it sits, so a
narrow wrap around one launch is as valid as a whole-body wrap
(`gpu_active_rows.set_row_compaction` is the narrow form). The whole-body form
exists only because an early return does not prune, which is why
`gpu_sparse._enqueue_zero_i32` and
`gpu_gradient_stream.enqueue_range_histogram_interleaved` take the whole-body
form. Both have an early return in the middle of the body.

A `comptime if not has_accelerator(): raise` with NO `else`, followed by the
body at the original indentation, is the shape that looks right and is not.
Nothing is pruned. `tools/check_gpu_guards.py` treats it as unguarded.

### Mechanism 2. Any reachable `enqueue_function` is a hazard, not only the expensive ones

The first round of guards in `99fa298` derived its set from which kernels
allocate shared memory, observed that `_iota_kernel` had elaborated fine on
the failing build, and deliberately left `begin_tree` and
`set_row_compaction` unguarded. CI disagreed and named the chain
`histogram_gpu.mojo:1929` to `:1940` to `gpu_active_rows.begin_tree` to the
launch inside it. `b5e0df8` guarded both.

Do not reason about what a kernel does. Reason about whether it can be
elaborated on a build with no accelerator. If it can, guard it.

### Mechanism 3. In a TEST module the guard belongs on the HELPER

`TestSuite.discover_tests[__functions_in_module()]()` enumerates every
function in the module, so every function is INSTANTIATED whether or not a
live call reaches it. `tests/test_const_hessian.mojo` had every caller of
`_gpu_leaf_matches` wrapped in `comptime if not has_accelerator()` and still
failed to compile, because the helper itself built a `GpuHistogramBuilder`.
`1f7efd8` guarded the helper.

Guarding only the callers is what every other CPU-set test in this suite does,
and it is sufficient there ONLY because their helpers touch no device API. The
moment a helper does, the helper needs its own guard.

## 2. How the library half stays latent, and why

CI runs `pixi run test-cpu`, which is `tools/run_tests.sh cpu`. That script
first builds `build/mojotrees.mojopkg` with `mojo precompile -I src
src/mojotrees` and then runs each selected test file with `mojo run -I build`.

The header of `run_tests.sh` says the precompile "elaborates every module in
`src/mojotrees/`". At module granularity that is true. At FUNCTION-BODY
granularity it is not, and the difference is the whole reason a library full
of unguarded launchers has green CI. The evidence is behavioral rather than
documented. CI reported nine failing TEST FILES, which requires the package
build in front of them to have succeeded, and at that moment `src/` held
dozens of unguarded `enqueue_function` sites. So a launch in `src/` is
elaborated when something reaches it, and the roots that reach are

- every function in a test module that CPU-only CI compiles, by mechanism 3,
- `main()` and the closure below it,
- every function a bindings, capi, or cli build registers or exports.

This is an inference from CI's behavior, not a statement from the compiler's
documentation. If it is ever falsified, the entire baseline in section 4
becomes a live build failure at once, which is worth knowing before trusting
it.

`tools/run_tests.sh` decides which tests CPU-only CI compiles. A file is
GPU-only if it is named in the `GPU_ONLY` list OR is named `test_gpu_*`
without a `# run_tests: cpu-safe` marker line. Everything else is in the CPU
set. Today that is 106 CPU-set files against 51 GPU-set files. A site
reachable only from a GPU-set test is a lower priority, because no CPU-only
build compiles that file at all.

## 3. The site table

Counted by `python3 tools/check_gpu_guards.py --list` over
`src/mojotrees/`, `bindings/`, `capi/`, and `cli/`. "Hazard" means
`enqueue_function`, a bare `DeviceContext(` construction, or
`_accelerator_arch()`. Docstrings and comments are excluded, which matters in
this tree because its prose quotes both constructs constantly.

| | at `1f7efd8` | after this audit |
| --- | --- | --- |
| hazard sites | 126 | 126 |
| unguarded | 83 | 73 |
| unguarded and module-level | 10 | 0 |

By kind, the 126 are 117 `enqueue_function`, 8 `DeviceContext(`, and 1
`_accelerator_arch()`.

Per file, after this audit.

| file | sites | unguarded |
| --- | ---: | ---: |
| `src/mojotrees/gpu_active_rows.mojo` | 31 | 15 |
| `src/mojotrees/gpu_split_search.mojo` | 15 | 2 |
| `src/mojotrees/gpu_objectives_native.mojo` | 15 | 13 |
| `src/mojotrees/gpu_leaf_batching.mojo` | 13 | 13 |
| `src/mojotrees/gpu_sparse.mojo` | 12 | 11 |
| `src/mojotrees/gpu_tree_tables.mojo` | 12 | 1 |
| `src/mojotrees/gpu_gradient_stream.mojo` | 7 | 3 |
| `src/mojotrees/gpu_multiclass_batch.mojo` | 6 | 6 |
| `src/mojotrees/gpu_predict.mojo` | 6 | 6 |
| `src/mojotrees/gpu_categorical.mojo` | 3 | 0 |
| `src/mojotrees/gpu_fused_round.mojo` | 1 | 1 |
| `src/mojotrees/gpu_runtime.mojo` | 1 | 1 |
| `src/mojotrees/histogram_gpu.mojo` | 1 | 1 |
| `src/mojotrees/gpu_blocked_bins.mojo` | 1 | 0 |
| `src/mojotrees/gpu_packed_bins.mojo` | 1 | 0 |
| `src/mojotrees/device_policy.mojo` | 1 | 0 |

`bindings/`, `capi/`, and `cli/` hold no hazard site of their own.
`bindings/_mojotrees.mojo` registers its GPU validation entry points inside
`comptime if has_accelerator():` with an `else:` binding refusal stubs, which
is why the CPU-only extension build in CI's `python` job survives. That job
compiles the bindings on the same accelerator-free runners, so it is a second
witness to this class and not only the Mojo suite.

The full, current, per-line table is not transcribed here on purpose. A
transcription drifts. Regenerate it.

```
python3 tools/check_gpu_guards.py --list
```

## 4. Reachability, and where this analysis is approximate

Which of the 73 unguarded sites can a CPU-set test reach? Reachability was
determined by hand, working BACKWARD from each unguarded site through its
callers until each path ended at a guard, at a test file, or at nothing. That
is a call graph read by hand and it has limits, stated here rather than
papered over.

A name-based transitive closure was built first and then discarded as
unusable. In this tree `route`, `pack`, `estimate`, `compute`, `build`,
`train`, `fill_grad_hess`, `leaf_indices`, and `begin_tree` all collide
between the host plane and the device plane, so a closure over names taints
essentially every entry point and answers nothing. Four specific collisions
were checked and all four were false positives.

- `fill_grad_hess`. `test_cpu_parallel` and `test_cpu_partition` call
  `mojotrees.boosting.fill_grad_hess`, not `GpuObjectiveState.fill_grad_hess`.
- `leaf_indices`. `test_mojotrees` calls `Model.leaf_indices`, not
  `GpuPredictor.leaf_indices`.
- `compute`. `test_text_features` calls a CTR calcer's `compute`, not
  `GpuCategoryStats.compute`.
- `begin_tree`, `build_leaf`, `apply_split`. `test_const_hessian` drives
  `GpuHistogramBuilder`, and `histogram_gpu.mojo` names
  `GpuSparseHistogramBuilder` nowhere outside prose, so the identically named
  sparse methods are not on that path.

What replaced the closure is type-directed and is the criterion the gate now
uses. A CPU-set test can reach a hazard only by naming a symbol that carries
one, which in practice means constructing a device-owning struct or calling a
module-level launcher it imported. Working from the import lists of all 106
CPU-set files, the device-owning structs named anywhere in the CPU set are

- `GpuHistogramBuilder`, in `tests/test_missing.mojo` (guarded) and
  `tests/test_const_hessian.mojo` (three helpers, one guarded, two NOT, see
  section 6),
- `GpuSplitSearcher`, in `tests/test_cosine_device_split.mojo` (guarded).

No other CPU-set file names one. Every other `mojotrees.gpu_*` symbol a CPU-set
test imports is host arithmetic or host policy that opens no device, which is
what `test_gpu_tile_floor`'s `# run_tests: cpu-safe` marker asserts about that
file and what `gpu_split_policy`, `gpu_resident_round`, `gpu_bin_packing`,
`gpu_binned_layout`, `gpu_packed_bins`, `gpu_blocked_bins`, `gpu_tiling`, and
the host halves of `gpu_split_search` all are.

Five CPU-set files import `train_gpu` or `train_custom_gpu` from
`mojotrees.train_gpu` (`test_categorical`, `test_custom_objective`,
`test_grow_policy`, `test_interaction`, `test_missing`). Every public entry
point in `train_gpu.mojo` and `train_gpu_sparse.mojo` carries its own guard,
19 of them, and that is the firewall that keeps the training half of the
library latent. `grow_tree_gpu`, `grow_tree_gpu_profiled`,
`grow_tree_gpu_sparse`, and `device_gradients` are public and UNGUARDED in
those two modules; no CPU-set test imports any of them today, and one that did
would be a live build failure.

**Where this is approximate.** The backward walk is a hand reading, so a call
this analysis did not see is possible. The type-directed check is precise for
imports and blind to any other route to a symbol. Neither can prove the
inference in section 2 about what `mojo precompile` elaborates. What would
falsify the whole set is simple and worth stating, which is a CPU-only CI run
failing at a site not in the table.

## 5. What this audit changed

Every module-level launcher in the library is now guarded. That is a statable
invariant, it has no exemptions, and `tools/check_gpu_guards.py` rule R1
enforces it. It is the right line to hold because a module-level launcher is
one `from mojotrees.gpu_x import launcher` away from a CPU-set test, whereas a
method of a device-owning struct needs the struct constructed first.

Ten sites across six functions were guarded to reach it.

| function | form | sites |
| --- | --- | ---: |
| `gpu_objectives_native.enqueue_abs_sum` | whole body | 1 |
| `gpu_objectives_native.enqueue_sq_sum` | whole body | 1 |
| `gpu_categorical.enqueue_category_stats` | whole body | 2 |
| `gpu_categorical.apply_categorical_split_pooled` | whole body | 1 |
| `gpu_sparse._enqueue_zero_i32` | whole body, early return | 1 |
| `gpu_gradient_stream.enqueue_range_histogram_interleaved` | whole body, early return | 4 |

Four modules gained `from std.sys import has_accelerator`, placed last among
the `std.*` imports the way `gpu_active_rows`, `gpu_packed_bins`, and
`gpu_blocked_bins` place it.

**The accelerator path is provably unchanged.** Each new `else:` body was
dedented by four columns and diffed against the original text at `1f7efd8`.
Four of the six are byte-identical. The other two differ only in comment
rewraps forced by the extra indentation, plus one split string literal in
`apply_categorical_split_pooled` that reconcatenates byte-identically
(`"... region of the pool;" " call CatSetPool.upload ..."` became
`"... region of the" " pool; call CatSetPool.upload ..."`). No statement, no
expression, and no argument changed. This is the same standard `99fa298` held
itself to and for the same reason, which is that a comptime guard that
accidentally edits the compiled branch is worse than the bug it fixes.

## 6. What remains, and it is not zero

**Two live defects, found and then closed while this page was being written.**
`tests/test_const_hessian.mojo` is in the CPU set, and two of its helpers
constructed `GpuHistogramBuilder` with no guard.

- `_gpu_feature_group_matches`
- `_gpu_fused_subtraction_matches`

These were the same mechanism 3 that `1f7efd8` fixed on `_gpu_leaf_matches` in
the same file, at the same time, and left on the other two. Their callers were
guarded, which does not help. On the evidence at the time CPU-only CI would
still have failed to compile that file, which is a fourth cycle of the same
symptom that nobody had spent yet. Both now carry the wrap `_gpu_leaf_matches`
already carried, and rule R3 is green.

This is the one useful data point about whether the gate earns its place. It
was written from three known failures, and the first thing it did on the tree
it was written against was name two more of the same kind that no one had
found by reading.

**Seventy-three unguarded sites in the library**, all of them methods of
device-owning structs, all of them recorded in
`tools/gpu_guard_baseline.txt`. They are latent because no CPU-set test
constructs those structs outside a guard. They were left rather than guarded
for a stated reason. Each one is a large re-indentation of a function this
lane could not compile, and a botched wrap costs the same CI cycle the audit
exists to save. Guarding a public method that a caller reaches prunes the
private helpers below it, so the honest way to shrink the baseline is
top-down, one struct at a time, by somebody who can build.

**Four public and unguarded training entry points**, `grow_tree_gpu`,
`grow_tree_gpu_profiled`, `grow_tree_gpu_sparse`, and `device_gradients`. No
CPU-set test imports them. A test that did would break the build.

## 7. The checklist, before you push

You are almost certainly on a machine with an accelerator. It cannot catch any
of this. Run the gate.

```
python3 tools/check_gpu_guards.py      # or: pixi run check-gpu-guards
```

It is part of `tools/check_gates.sh`, so `pixi run check-gates` covers it too.
It reads the tree as text, invokes no compiler, and takes well under a second.

If you are reviewing rather than running, four questions.

1. Did you add an `enqueue_function`, a `DeviceContext()`, or any other API
   that needs to name a GPU architecture at compile time? If it is in a
   module-level function, it must be guarded. No exemptions.
2. Did you add or change a function in a test file that CPU-only CI compiles?
   Check `tools/run_tests.sh list` classification. If ANY function in that
   file touches a device API, that function needs its own guard, helpers
   included, because `TestSuite` instantiates all of them.
3. Did you write `comptime if not has_accelerator(): raise` and then continue
   at the original indentation? That prunes nothing. It needs the `else:`.
4. Did you guard something by wrapping a body? Dedent your new `else:` block
   and diff it against the original. Anything but comment rewraps means you
   changed the accelerator path, which is a worse bug than the one you were
   fixing.

And the fact worth keeping in front of you the whole time. The compiler emits
one error stack and stops, so a CI run that gets further than the last one has
not told you the set was complete. It has told you the next site.

## 8. The gate

`tools/check_gpu_guards.py` is standard-library Python over files in the tree,
in the shape `tools/audit_test_structure.py` established. It never invokes
`mojo`. Three rules.

- **R1**, module-level launchers. Every `enqueue_function` in a function that
  is not a struct method must be guarded. No baseline, no exemption.
- **R2**, everything else ratchets. Every other unguarded hazard site must
  appear in `tools/gpu_guard_baseline.txt`, keyed by
  `file::owner.function::kind` rather than by line number so the key survives
  edits above it. The set may shrink and may not grow. `--write-baseline`
  regenerates, and is for REMOVING lines.
- **R3**, CPU-set test modules. A function in a CPU-set test file that names a
  GPU-carrying symbol imported from a `mojotrees.gpu_*`, `histogram_gpu`, or
  `train_gpu*` module, or that names `DeviceContext`, must contain a
  `comptime if ... has_accelerator()` of its own. This is mechanism 3, and it
  is the rule that would have caught the 2026-08-17 failure before it was
  pushed.

R3's symbol list is derived from `src/` at run time by DIRECT containment,
which means a struct with a method that itself launches or opens a context,
plus a public module-level function that does. It is deliberately not a call
graph closure, for the collision reasons in section 4. Direct containment is
also exactly right for the question R3 asks, because a private helper that
only forwards to a GUARDED launcher is pruned along with it.

The gate also checks its own copy of the `GPU_ONLY` list against
`tools/run_tests.sh`, so the two cannot drift apart silently.

**What it will false-positive on.** A CPU-set test that imports a
device-owning struct's NAME for a type annotation or a doc reference, without
constructing one, is flagged. A test that guards with something other than a
`comptime if ... has_accelerator()` at the top level of the function, for
instance a guard inside a nested block, is flagged. Both are rare and both are
answered by adding the guard, which costs nothing on a machine that has an
accelerator.

**What it cannot see.** A hazard that is neither `enqueue_function`,
`DeviceContext(`, nor `_accelerator_arch()` has to be added to `HAZARDS` by
hand. It cannot prove the baseline is safe; the baseline is a hand reading
that this gate freezes. And a passing run does not mean the build is green.
CPU-only CI remains the only witness to that, by construction.

Can the gate be honest without a real compiler? For R1 and R3, yes, because
both are syntactic properties of the text and neither needs to know what
elaborates. For R2, no, and it does not claim to be. It is a ratchet over a
hand-checked set, and its value is that it makes any addition to that set a
deliberate, reviewed act instead of an accident discovered by CI four cycles
later.
