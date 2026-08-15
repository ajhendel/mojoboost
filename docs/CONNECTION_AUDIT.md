# Connection audit

What in this repository is reachable from something a user can call, and
what is not.

mojotrees was built by many parallel lanes. A lane can finish a capability
completely, module written, tests passing, handoff filed, without the
capability ever becoming reachable, because the edit that would reach it
lives in a file the lane does not own. This document is about that state.

**`tools/connectivity_audit.py` is the authority. This file is not.** The
previous revision of this document was a hand-gathered snapshot, 520 lines
of counts collected with `rg` and `sed`, carrying a note that the script
"has not been run". It has now. The script disagreed with the snapshot in
both directions, and several of the snapshot's headline findings had been
closed for a day. A count written down by hand is wrong within the hour in
a tree this active, so the counts are gone from here and live only in the
script's output.

```sh
python3 tools/connectivity_audit.py                 # full report
python3 tools/connectivity_audit.py --section binding-modules
python3 tools/connectivity_audit.py --json          # machine readable
```

CI runs it on every pull request as a non-blocking job, so the report is
attached to the change that moved it.

---

## The entry points

Five, and only five, things can start a call into mojotrees:

| Root | What it is |
| --- | --- |
| `python/mojotrees/__init__.py` | the public Python package |
| `bindings/_mojotrees.mojo` | the CPython extension, whose `def_function` table is the entire Python-to-native surface |
| `src/mojotrees/__init__.mojo` | the public Mojo API, a block of re-exports |
| `capi/mojotrees_capi.mojo` | the C ABI, whose `@export`ed functions are the entire C surface |
| `cli/mojotrees_cli.mojo` | the `mojotrees` command |

Everything else in the repository is reachable, or is not, through one of
these. A module reached only from `tests/` or `bench/` is exercised, not
shipped, and the report says so on the row.

---

## How to read a finding

Every finding carries a status. The status, not the finding, is what tells
you whether to act:

- **DEAD** — nothing reaches it and no reason is recorded. Remove it.
- **EXPERIMENTAL** — deliberately unwired. The reason on the row says what
  would change that. A test-only reference implementation is the common
  case and is not a defect.
- **PENDING** — implemented and blocked on a named cross-lane edit. Someone
  owns it and the row says who.
- **CONNECTED** — reachable. It appears in the report only when something
  else about it is worth saying.

An unreached module is a question, not a bug. The question is which of the
four it is, and the answer belongs on the row rather than in a reader's
head.

---

## The reading on 2026-08-15

Recorded so a later reading has something to move against, not as a fact
about the tree today. Re-run the script.

**Eight findings: six EXPERIMENTAL, two PENDING, no DEAD.** Both PENDING
findings were this document naming files under `tests/parallel/`, a
directory that no longer exists, which is the whole argument for keeping
the numbers in the script.

Two native modules no entry point reaches, both EXPERIMENTAL and both
correctly so:

- `backend` — a one-function dispatch shim kept as the reference the
  CPU/GPU equivalence test compares against. Test-only by design.
- `gpu_vendor_policy` — CUDA and HIP occupancy policy, merged from the
  `gpu_cuda_policy` / `gpu_amd_policy` twins. Reached only from its test
  until a discrete-GPU trainer consults it.

Four Python modules the package root never reaches, all four intentional:
`mojotrees.dask` and `mojotrees.lgbm_model_io` are PEP 562 lazy submodules,
`mojotrees._dask_runtime` is reached from inside `dask.py`'s functions, and
`mojotrees._public_api_plan` is a plan expressed as data whose own docstring
says importing it would be the bug.

Everything else the script checks came back clean: no duplicate registries,
no orphan binding modules, no binding function without a Python caller, no
estimator parameter without a downstream consumer, no save/load
disagreement, no missing native function, no C header drift.

What closed since the 2026-08-14 pass: the five binding modules that did not
exist at runtime, the GPU surface with no Python caller, the duplicate
parameter surfaces, and the GPU sparse and categorical cluster
(`gpu_categorical`, `gpu_sparse`, `gpu_sparse_layout`, `train_gpu_sparse`),
which was four modules and roughly 4,500 lines that nothing imported.

---

## The check the audit does not do, and now something does

`mojo precompile -I src src/mojotrees` elaborates every module in the
package, including the ones no test imports. `pixi run test` did not: a test
compiles the import graph it names and nothing else, so a module that no
suite reaches was never type-checked by anything.

Four modules did not compile at all, and had not for some time:

| Module | What was wrong |
| --- | --- |
| `distributed_transport.mojo` | `from std.sys.ffi import external_call` names a module that is not there, and the `errno` shims returned an `UnsafePointer` with an unbound origin |
| `histogram_cache_policy.mojo` | two `CacheEpochs` values passed by implicit copy |
| `quantized_gradient.mojo` | two `QuantScales` values passed by implicit copy, and a closure capturing pointers whose origin was `origin_of(out.grad)` |
| `ranking_advanced.mojo` | a `QueryPartition` passed by implicit copy |

All four are fixed, and `pixi run build-pkg` is now the check that keeps them
fixed. This is a different question from reachability and the audit script
does not answer it: a module can be reachable and not compile, which is what
`ranking_advanced.mojo` was.

---

## What this audit cannot tell you

It is static. It reads imports, registrations, and names. It does not run
anything, so it cannot tell you that a reachable path is correct, that a
registered binding does what its name says, or that a parameter a user can
set changes the model. `tools/check_parity.py` owns the behavior contract
and `validation/manifests/` owns what has actually been run. This file owns
one question only: can a caller get there from outside.
