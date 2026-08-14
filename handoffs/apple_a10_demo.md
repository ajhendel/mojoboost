# Handoff, Apple lane A10, five minute Apple Silicon experience

Lane A10 of the parallel Apple round. Nothing outside the two assigned paths
was touched, nothing was committed or staged.

## What this lane produced

| Path | What it is |
| --- | --- |
| `examples/apple_silicon/README.md` | The walkthrough. Headline, install placeholders, seven steps, troubleshooting, and a table of what is not integrated yet. |
| `examples/apple_silicon/five_minute_tour.py` | One runnable script covering the same seven steps. Standard library only, no numpy, deterministic data from a splitmix64 generator. |
| `examples/apple_silicon/TIMINGS.md` | The CPU and GPU timing table. Every cell empty, with the rules for filling it and the procedure for producing a row. |

The example is written against the API as it exists in the working tree at
the time of writing, verified by reading `python/mojoboost/__init__.py`,
`src/mojoboost/device.mojo`, `python/setup.py`, and `pixi.toml`. Anything the
tour shows that does not exist is labeled `NOT AVAILABLE YET` or
`NOT IMPLEMENTED` and is never executed.

## Verified against the current tree

Used and believed to work.

- `mojoboost.gpu_available()`, exported in `__all__`.
- `device="cpu" | "gpu" | "auto"` on the estimators, `device_type` alias,
  `device_` recorded by `fit`, `_DEVICES = ("cpu", "gpu", "auto")`.
- `MOJOBOOST_DISABLE_GPU`, `MOJOBOOST_AUTO_MIN_CELLS`,
  `MOJOBOOST_NUM_WORKERS`, `MOJOBOOST_PARALLEL_MIN_OPS`.
- `fit(..., eval_set=, eval_names=, eval_metric=, early_stopping_rounds=,
  min_delta=)` and the fitted attributes `n_iter_`, `best_iteration_`,
  `best_score_`, `stopped_early_`, `evals_result_`.
- `evals_result_[valid_name][metric_name]`, where the metric key is the
  canonical name `python/mojoboost/_eval.py` resolves an alias to. The tour
  passes `eval_metric=["l2", "l1"]` and reads `["holdout"]["l2"]`, and `l2`
  and `l1` are canonical rather than aliases, so the key is the string that
  was passed. A tour written against `mse` or `mae` would have to read `l2`
  or `l1` back.
- `best_score_` is a scalar float (`float(result[4])`), which is the
  departure from LightGBM the tour prints with `:.6f`. If the compatibility
  work turns it into LightGBM's dict of every set's every metric, Step 3 of
  the script breaks on the format and needs one line changed.
- `predict(X, raw_score=, start_iteration=, num_iteration=, pred_leaf=,
  pred_contrib=, validate_features=)`.
- `model.save(path)` and `MojoBoostRegressor.load(path)`, with no `device_`
  on a loaded estimator and no gain importance in the file.
- Wheel platform tag `macosx_26_0_<arch>` from `python/setup.py`.

Shown only as a labeled placeholder, because it does not exist.

- `pip install mojoboost` and any downloadable wheel.
- `predict(..., device="gpu")`.
- `explain_device_choice(X)` (lane A9's Python policy and report layer).
- `device="auto"` ever selecting the GPU without `MOJOBOOST_AUTO_MIN_CELLS`.
- Early stopping, multiclass, sparse input, and Python objective callbacks on
  the GPU path, each of which raises today.

## Central wiring required, for the integration owner

None of this was done by this lane. All of it touches shared files.

### 1. `README.md`

Add one link in the examples or getting started area. Suggested text, with no
speed claim in it.

```markdown
- [examples/apple_silicon](examples/apple_silicon/README.md), a five minute
  tour on an Apple Silicon Mac covering GPU detection, `device="auto"`,
  validation and early stopping, prediction, saving and loading, and the
  empty CPU/GPU timing table.
```

Do not lift the headline sentence into `README.md` until the release gate at
the bottom of this document is met. Inside the example it is immediately
qualified by a status block. Standing alone in the project README it would
read as a benchmark claim.

### 2. `pixi.toml`

One task in `[tasks]`, next to the other Python entry points. It depends on
`build-python` for the same reason `test-python` does.

```toml
example-apple = { cmd = "python examples/apple_silicon/five_minute_tour.py", env = { PYTHONPATH = "python" }, depends-on = ["build-python"] }
```

If the `env` key on a task is not wanted there, the alternative is a plain
`cmd` of `PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py`,
which is what the example README documents. The script imports `mojoboost`
and nothing else outside the standard library, so no new dependency and no
new pixi feature is needed. It must not be added to `test` or `test-python`.

### 3. Packaging

`packaging/build_wheel.sh` and `python/pyproject.toml` package
`python/mojoboost`, so `examples/` is outside the wheel already and needs no
exclusion. Confirm that when the wheel becomes a release artifact, because
the example README's install placeholders assume the example ships in the
repository and not in the wheel.

### 4. CI

Do not add this script to the CI matrix. CI is CPU only Linux, the tour
trains three models, and the GPU sections would report `False` and skip,
which is a slow way to test nothing. If a guard against bit rot is wanted,
the cheap one is a compile check in an existing lint or docs job.

```sh
python -m compileall -q examples/apple_silicon
```

### 5. Documentation cross links

- `docs/DEVICE_SELECTION.md` (lane A9) should link here as the worked
  example, and this example should link there once it exists. The
  `explain_device_choice` placeholder in Step 5 is the seam.
- `docs/APPLE_GPU_BENCHMARK_PROTOCOL.md` (lane A8) is already referenced by
  `TIMINGS.md` as the authority on measurement methodology, with a note that
  it wins where the two disagree. No edit needed if that file lands under
  that name. If it lands under a different name, fix the reference in
  `TIMINGS.md`.
- `docs/GPU_VALIDATION.md` is linked from the example's unsupported hardware
  section as the record of what has actually been exercised on which device.
  That link is live today and needs nothing from anyone.
- `docs/LIGHTGBM_PARITY.md` is referenced from the example README as the
  authority on what matches LightGBM. If `tools/check_parity.py` grows a
  check that cited paths exist, note that this example cites
  `src/mojoboost/device.mojo`, `src/mojoboost/metrics.mojo`,
  `src/mojoboost/parallel.mojo`, `bench/bench_train_gpu.mojo`,
  `docs/GPU_VALIDATION.md`, and `python/setup.py`.

### 6. Follow up edits the other lanes force

When a lane lands one of these, the example has a matching edit waiting.
Each is a small, local change, and each removes a `NOT AVAILABLE YET` block.

| Lane | Landing that | Edit here |
| --- | --- | --- |
| A9 | `explain_device_choice` | Step 5 placeholder becomes a real call, and the manual explanation block shrinks to a pointer. |
| A4 | GPU prediction | Step 4 placeholder becomes real, and the `predict` row leaves the not integrated table. |
| A1 / A2 / A3 / A5 / A6 | A faster GPU trainer | Nothing in the example text. `TIMINGS.md` gets rows, and only then does the status block under the headline change. |
| A8 | Benchmark protocol | `TIMINGS.md` defers to it, already written that way. |

## Minimum release capabilities before this example can be advertised

Advertised means linked from the project README's opening, posted to the
Modular forum or Discord, or used as the landing page for a Show HN. The
example is useful in the repository today. It is not something to point the
world at until these hold.

**Hard gate, all four required.**

1. **An installable artifact.** Either a published PyPI wheel or a versioned
   GitHub release asset with a checksum. Until then the first two install
   blocks are placeholders, and a five minute experience that begins with
   `git clone` and a toolchain build is not a five minute experience for
   anyone who is not already a Mojo user.
2. **A measured Apple timing table.** At least two distinct chips and at
   least two shapes each in `TIMINGS.md`, produced under the A8 protocol,
   with CPU and GPU from the same build and the same machine. The headline
   sentence is a performance claim. Right now the only evidence in the
   repository points the other way.
3. **A GPU result worth showing on at least one Apple chip.** A ratio at or
   below 1.0 at some real shape, reproducible. If the honest state after the
   compaction work is still that the CPU wins on Apple silicon, the headline
   has to change rather than the table, and the example should be advertised
   as a portable GBDT that also runs on the GPU.
4. **The GPU path survives a redistributed build.** `gpu_available()` is a
   compile time property, so a wheel built on a GPU machine claims an
   accelerator wherever it is installed. Before an installer exists, that gap
   is invisible. After one exists, a user on hardware without Metal support
   who sets `MOJOBOOST_AUTO_MIN_CELLS` gets a failure at device open. Either
   make availability a runtime check, or document and test the failure so it
   is a clear message rather than a crash.

**Should hold, each one otherwise needs a visible caveat in the example.**

5. **Early stopping on the GPU path**, or an accepted permanent caveat. Right
   now the tour has to switch to `device="cpu"` for its most useful step,
   which undercuts the headline in the middle of the demo.
6. **GPU prediction** (lane A4). A GPU trained model that can only predict on
   the CPU is a defensible v1 and a weak demo.
7. **A device explanation API** (lane A9). Step 5 currently explains the
   policy in prose that a future policy change will silently invalidate.
8. **A macOS and toolchain support statement.** The example asserts the
   `macosx_26_0` wheel tag and points at Modular release notes for the Metal
   minimum. A release needs a tested lower bound, not a pointer.
9. **The example runs green on a machine that is not the development Mac.**
   It has never been executed anywhere. See the next section.

## Focused check that was run, and what was not

This is a documentation and example lane, so it has no assigned test file.
The one command run was a syntax and bytecode compile of the new script.

```sh
python3 -m py_compile examples/apple_silicon/five_minute_tour.py
```

That checks that the file parses. It does not check that it runs. The script
was deliberately not executed, because doing so requires `pixi run
build-python` and then three model fits, which is a build plus a benchmark
sized workload in a checkout shared with every other lane in this round.

**Nothing in `examples/apple_silicon/` has been executed. No output in the
README or in `TIMINGS.md` came from a run.** The example README shows no
sample output for exactly that reason. Before this is linked from anywhere,
somebody has to run

```sh
pixi run build-python
PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py
```

once on an Apple Silicon Mac and once with `MOJOBOOST_DISABLE_GPU=1`, and fix
whatever it says. The likeliest breakages are an estimator keyword this lane
read correctly but a peer lane has since resignatured, `best_score_` being a
scalar here where a compatibility lane may be turning it into a dict, and
`evals_result_["holdout"]["l2"]` if the metric key stops being the canonical
name. Both of those were traced through the current source and are correct as
written today, so they are staleness risks and not known defects.

After that compile check, the only further commands run were read-only source
greps and `git diff --check`. `five_minute_tour.py` has not been edited since
it was compile checked. The two later edits were to
`examples/apple_silicon/README.md` alone, removing the two links that pointed
a user facing document into this internal handoff.

`git diff --check` was run on the assigned paths. All three files are new and
untracked, so it reports nothing, which is expected rather than reassuring.

## Risks

- **Staleness by construction.** The example narrates current policy, in
  particular that `auto` always resolves to the CPU and why. That paragraph
  becomes wrong the day a crossover threshold ships. It is confined to Step 2,
  Step 5, and the status block under the headline.
- **The headline outruns the evidence.** It is the assigned lead sentence and
  it is a claim about acceleration that no measurement in this repository
  supports today. The status block directly under it says so in the first
  paragraph. Do not separate them, and do not quote the headline alone.
- **Placeholder installs read as real.** Three install blocks, only the third
  works. They are labeled, but a reader who skims copies the first one. If
  that risk is unacceptable before a release exists, cut the first two blocks
  and keep the release gate above as the record of what they were waiting on.
- **Multiple lanes will edit this directory next round.** A4, A9, and A8 each
  own a placeholder or a reference in these files. Sequence those edits rather
  than running them in parallel.
