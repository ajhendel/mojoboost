# connect_22 - static tooling triage (K5)

Date: 2026-08-15. Lane: K5, decision lane. Question per tool: is it wired
into CI or a pixi task, does another tool subsume it, is it a finished
one-shot scaffold. Evidence: `.github/workflows/*.yml`, `pixi.toml`, import
graph across `tools/`, doc and manifest citations, and one dry run of each
tool on this checkout.

## Wiring found

| where | what |
| --- | --- |
| `.github/workflows/ci.yml:56` | `python3 tools/check_parity.py` |
| `pixi.toml:18` | `check-parity = "python3 tools/check_parity.py"` |
| `.github/workflows/contributor-access.yml:32,54` | `tools/governance_access.py` (not in scope, noted for completeness) |

Nothing else under `tools/` is run by CI or by a pixi task.

## Import graph inside tools/

```
audit_integration.py      -> connectivity_audit  (graph engine + CLASSIFICATION)
api_snapshot.py           -> check_parity        (I11 cross-check, guarded)
model_fixture_manifest.py -> api_snapshot        (model_format_block)
inspect_startup_artifacts -> packaging/matrix/validate_artifact.py (macho_info)
_public_api_plan.py       <- nothing (by design; connectivity_audit lists it
                             in CLASSIFICATION as EXPERIMENTAL/connect_07)
```

## Verdicts

| tool | lines | wired | subsumed by | verdict | basis |
| --- | --- | --- | --- | --- | --- |
| `check_parity.py` | 1,427 | CI + pixi | no | **keep-wired** | frozen; not touched |
| `connectivity_audit.py` | 1,584 | no | no (it is the engine) | **keep-dev-tool**, recommend CI informational job | house rules require it before/after every lane; `audit_integration` imports it; cited by README, ARCHITECTURE, INTEGRATION_INVENTORY, CONNECTION_AUDIT, CAPABILITY_LEVELS, LIGHTGBM_PARITY |
| `audit_integration.py` | 495 | no | no; it gates `docs/INTEGRATION_INVENTORY.md`, which `connectivity_audit` deliberately does not | **keep-dev-tool** | runs clean today (0 ERROR, 1 GAP); a later `--check-inventory` subcommand on `connectivity_audit` would fold it, deferred (not this round) |
| `api_snapshot.py` | 1,406 | no | no | **keep-dev-tool** (blocked, not finished) | baseline `compatibility/api_snapshot.json` has never been written; `--check` still emits real findings (I8: `MOJOTREES_ABI_VERSION` documented, unread); imported by `model_fixture_manifest`; normative docs under `compatibility/` and `jobs.toml` name it. Becomes a CI candidate once `--write` lands the baseline |
| `model_fixture_manifest.py` | 409 | no | no | **keep-dev-tool** (blocked, job not started) | `compatibility/fixtures/` holds only README + manifest.toml; no fixture files, no `checksums.json`; `--check` fails by design until fixtures exist. Cited by COMPATIBILITY/DEPRECATION policy, fixtures README, manifests |
| `validation_plan.py` | 1,482 | no | no | **keep-dev-tool** | planner over `validation/manifests/` (live lane, untouched); `--self-check` passes with 1 known gap (`tests/parallel/test_hybrid_replica.mojo` has no `[[job]]`, belongs to the manifests lane); documented in FOCUSED_VALIDATION_PLAN.md and remaining_14 handoff |
| `audit_python_compat.py` | 778 | no | no | **keep-dev-tool** | passes today; named by `python/pyproject.toml`, PYTHON_SUPPORT.md, jobs.toml; 3.8-parseable on purpose so it can run under unsupported interpreters |
| `inspect_startup_artifacts.py` | 735 | no | no | **keep-dev-tool**, one bug to file | STARTUP_LATENCY.md + jobs.toml name it; reuses the shared Mach-O reader. On this Mac it reports ERROR for `/System/Library/Frameworks/Foundation.framework/...` and `/usr/lib/libSystem.B.dylib`, which the dyld shared cache resolves and which never exist on disk on modern macOS. Same class of false positive the script already exempts for bare ELF SONAMEs; the absolute-path rule needs a macOS system-path exemption before `--strict` is trustworthy |
| `python/mojotrees/_public_api_plan.py` | 516 | no (data module, imports nothing, nothing imports it) | no | **keep-dev-tool** (record as data), not deleted | `CURRENT_TOP_LEVEL` matches `__all__` exactly today (26/26); referenced by `__init__.py` comments, `dask.py` docstring, ECOSYSTEM_INPUTS, INTEGRATION_INVENTORY, CONNECTION_AUDIT, and the frozen LIGHTGBM_PARITY.md (line 635), so a delete would leave a dangling parity-doc citation. Weakness: nothing asserts the match, so it drifts silently; a 5-line pytest comparing `CURRENT_TOP_LEVEL` to `mojotrees.__all__` would make it earn its place |

## Deletes executed

None. The deletion bar (no CI wiring AND no imports AND finished one-shot
scaffold) is not met by any tool in scope. The two closest, `api_snapshot`
and `model_fixture_manifest`, are unfinished rather than finished: their
evidence files were never produced. `_public_api_plan` is finished but is
cited by the parity doc, which this lane may not edit.

## Recommendation for the coordinator (not implemented)

Add an informational CI job, `continue-on-error: true`, no build, seconds:

```yaml
  connectivity-audit:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - run: python3 tools/connectivity_audit.py
      - run: python3 tools/audit_integration.py   # 0 ERROR today; GAP is non-fatal without --strict
```

Reasoning: `connectivity_audit` is stdlib-only, builds nothing, and is
already the mandatory before/after check for every consolidation lane; a
CI log of its output makes each lane's "my section improved" claim
auditable from the PR rather than from a laptop. `audit_integration` rides
along because it imports the same engine and costs nothing extra. Keep it
non-blocking until INTEGRATION_INVENTORY.md is caught up (1 GAP now).

Second-order, once their evidence exists: `api_snapshot.py --check` after
`--write` lands the baseline, and `model_fixture_manifest.py --check` after
the first fixtures are committed. Both are gates in the compatibility
policy and are currently promises without a runner.

## Deferred

- `inspect_startup_artifacts.py` macOS system-path false positive (above).
- `audit_integration` fold into `connectivity_audit` as a subcommand.
- Test asserting `_public_api_plan.CURRENT_TOP_LEVEL == mojotrees.__all__`.

# consolidation K1 - RNG authority (splitmix64)

Session K1 of the consolidation round, 2026-08-15. Files touched: new
`src/mojotrees/rng.mojo`; rewired `sampling.mojo`, `bagging.mojo`,
`goss.mojo`, `quantized_gradient.mojo`, `tree_parameters_extra.mojo`. One
commit per rewire (9c3e8be, 3575146, 5ef83ac, 04a7fdf, 2178a6f on main,
not pushed). No `__init__.mojo`, docs, or off-limits files touched.

## What was chosen

`src/mojotrees/rng.mojo` is the single authority for the mixer and its
constants: `splitmix64(state)`, `uniform(counter)`, `GOLDEN`,
`TWO_POW_NEG_53`. Names are public (no leading underscore) because it is
an authority meant to be imported; the old private names were what forced
each module to keep "its own copy so it stays free of another module's
private names" (the reason recorded in the quantized_gradient and
tree_parameters_extra docstrings).

Stream derivation (`_stream`, `quant_stream`, `extra_split_stream`, the
per-depth stream in sampling) stays in each module. Those functions differ
on purpose and are each sampler's reproducibility contract; only the mixer
and the counter -> uniform helper were duplicated.

## Copy diff before unifying

The prompt asked for a stop if any of the four `_splitmix64` copies
(bagging, sampling, goss, boosting_dart) differed. They do not differ in
operations, constants, types, or order. The only difference is textual:
sampling.mojo (and the `_mix64` copies in quantized_gradient and
tree_parameters_extra) spell `0x9E3779B97F4A7C15` and
`(1.0 / 9007199254740992.0)` inline, while bagging/goss/dart name them
`_GOLDEN` / `_TWO_POW_NEG_53` with the same values. rng.mojo keeps the
sampling.mojo body (inline golden literal in the mixer) and uses the named
2^-53 constant in `uniform`, which is what bagging/goss/dart already did.
Same value either way; 2^-53 is exact in Float64. Not treated as a
stop-worthy difference; recorded here so the coordinator can disagree.

`quant_uniform` in quantized_gradient was identical to `uniform` and now
delegates to it (kept as a named function because the rounding code and
its docstrings call it by that name). `extra_candidate_index` in
tree_parameters_extra had the same expression inline and now calls
`uniform(stream)`.

## Verification

- tests/test_feature_sampling.mojo: 17 passed (after sampling rewire)
- tests/test_bagging.mojo: 23 passed (after bagging rewire)
- tests/test_goss.mojo: 15 passed (after goss rewire)
- quantized_gradient: no focused test under tests/ exercises quant_stream /
  quant_uniform. A scratch probe (not committed) printed quant_stream and
  quant_uniform values across seeds {0, 3, -7, 123456789}, rounds, classes,
  and planes, plus extra_split_stream values, before any change to those
  two modules; output was byte-identical after each rewire.
- tests/parallel/test_tree_parameters_extra.mojo: 47 passed (after rewire)
- connectivity_audit before/after: `_GOLDEN` went from 6 modules to 2
  (boosting_dart, ranking_advanced, both off-limits to this lane);
  `_TWO_POW_NEG_53` finding gone. Total findings 292 -> 276, though some of
  that delta is device/device_policy movement from another lane, not K1.
- pixi run check-parity: ok (parity doc cites sampling.mojo and
  tree_parameters_extra.mojo by file, not by these symbols).

## Deleted

The private `_splitmix64` / `_mix64` / `_uniform` bodies and the `_GOLDEN`
/ `_TWO_POW_NEG_53` constants in the five rewired modules. Every caller
was traced to rng.mojo; no test referenced the private names.

## Deferred / for the coordinator (wave-2 mop-up)

- boosting_dart.mojo still defines `_GOLDEN`, `_TWO_POW_NEG_53`,
  `_splitmix64`, `_uniform`, `_stream` (connect_17 lane). Rewire is
  mechanical and identical to the goss.mojo commit 5ef83ac: import
  `GOLDEN, splitmix64, uniform` from `.rng`, delete the three private
  definitions and two constants, rename call sites.
- boosting_rf.mojo: one RNG-related reference (grep hit); check whether it
  imports from bagging or dart and follow the same rewire.
- model_editing.mojo (4 hits) and ranking_advanced.mojo (7 hits, defines
  `_GOLDEN`): both audit-unreachable orphans; rewire when their lanes land
  or when their disposition is decided.
- bagging.mojo, goss.mojo, and boosting_dart.mojo have a byte-identical
  `_stream(seed, index)` (`splitmix64(seed_bits ^ (index * GOLDEN))`).
  It could move to rng.mojo as a shared `index_stream` after dart lands;
  left alone this round because the prompt scoped rng.mojo to the mixer
  and helpers, and each module's docstring presents the derivation as
  its own contract.
- Docstrings in tree_parameters_extra and quantized_gradient still cite
  `sampling._stream` by name as the masking precedent; still accurate,
  since sampling keeps `_stream`.
- Export: rng.mojo needs no entry in `src/mojotrees/__init__.mojo`. Nothing
  outside src/ imports it, and the bindings do not expose raw RNG. My
  recommendation is NOT to export it: it is an internal authority, and
  exporting `uniform` / `GOLDEN` at package level invites collisions with
  std.random names. If C0 prefers every authority module listed, export
  `splitmix64` only.
- CLAUDE_CODE_CONSOLIDATION_PROMPTS.txt still says splitmix64 is defined
  4x; after this lane it is 1x authority + 1 copy in boosting_dart.
