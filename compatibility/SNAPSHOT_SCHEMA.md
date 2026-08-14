# API snapshot schema

The normative shape of `compatibility/api_snapshot.json`, which
`tools/api_snapshot.py` writes and checks. This document is what a
consumer of the snapshot reads; neither the tool nor any past hand-written
draft is the authority on the shape.

`schema_version` is **2**. Version 1 is
`tests/parallel/api_snapshot_manifest.json`, hand-written and superseded;
section 8 says what changed and why.

## 0. Rules that hold for the whole file

1. **Derived, never edited.** Every block below except the two named in
   section 7 is parsed out of the source tree. Editing the file by hand
   makes the next `--check` disagree with the tree, which is exactly the
   signal the file exists to give, so a hand edit is indistinguishable
   from a break.
2. **No import, no build.** The tool parses Python with `ast`, and Mojo,
   C, TOML, and YAML with `re` and `tomllib`. It never imports
   `mojoboost` and never needs a built extension module. A snapshot check
   that needs a working build gets skipped the first day it is
   inconvenient.
3. **Deterministic bytes.** `json.dump(obj, fp, indent=2, sort_keys=True)`
   plus a single trailing newline. Regenerating with no source change
   produces a byte-identical file, so the diff is the whole signal.
4. **Absence is a value.** A field the tool could not derive is `null` and
   is listed in `meta.underived`. It is never guessed, never carried over
   from a previous run silently, and never filled with a plausible
   default. A tool that invents a value to avoid an empty field is worse
   than no tool.
5. **Unknown keys are ignored by consumers.** Adding a block or a key is
   additive and does not bump `schema_version`. Removing one, retyping
   one, or redefining one does.

## 1. `meta`

| Key | Type | Source |
|---|---|---|
| `schema_version` | int | This document. `2` |
| `status` | str | `"generated"` always. The tool writes nothing else; a hand-written file is not a snapshot |
| `generated_by` | str | `"tools/api_snapshot.py"` |
| `tool_version` | int | `TOOL_VERSION` in the tool. Bumped when a parser changes what it derives, so a diff caused by the tool is separable from a diff caused by the tree |
| `library_version` | str | The agreed value from `versions.library`. The tool errors rather than writing if the three locations disagree |
| `source_commit` | str or null | `--commit <sha>` if passed, else null. The tool runs no git command |
| `underived` | list[str] | Dotted paths of every field the tool could not derive on this run. Empty on a healthy tree |
| `carried` | list[str] | Dotted paths carried forward from the previous file under section 7 |

`source_commit` is an argument rather than a `git rev-parse` because the
tool must run identically inside a release tarball with no `.git`.

## 2. `versions`

| Key | Type | Source |
|---|---|---|
| `library` | str | `pixi.toml [workspace] version`, `python/pyproject.toml [project] version`, and `python/mojoboost/__init__.py __version__`. **Disagreement is an error, not a merge** |
| `library_locations` | dict[str, str] | Each of the three, separately, so a `--check` failure names which one moved |
| `c_abi` | int | `#define MOJOBOOST_ABI_VERSION` in `capi/mojoboost.h` |
| `model_format_writer` | int | `comptime CURRENT_FORMAT_VERSION` in `src/mojoboost/serialize.mojo` |
| `model_format_writer_token` | str | `comptime _VERSION` in the same file, e.g. `"v4"`. Must equal `"v" + model_format_writer` |
| `model_format_readable` | list[int] | The versions `_read_version` accepts, parsed from its `if token == "vN"` chain |
| `model_format_dump_reports` | int | `comptime MODEL_FORMAT_VERSION` in `src/mojoboost/model_dump.mojo` |
| `model_format_python_reads` | list[int] | `SUPPORTED_MODEL_FORMAT_VERSIONS` in `python/mojoboost/inspection.py` |
| `dump_format` | int | `DUMP_FORMAT_VERSION` in `python/mojoboost/inspection.py` |
| `snapshot_schema` | int | `2` |
| `requires_python` | str | `python/pyproject.toml` |
| `mojo_toolchain`, `max_toolchain` | str | `pixi.toml [dependencies]` |

Five numbers describe the model format because five places carry one, and
they are recorded separately rather than reconciled. Section 6 states the
invariants the tool enforces between them. A schema that recorded one
"model format version" would have hidden the drift this lane found.

## 3. `python`

| Key | Type | Source |
|---|---|---|
| `all` | list[str] | The `__all__` assignment in `python/mojoboost/__init__.py`, **in source order**. Order is recorded because a reordering is visible in the diff and is harmless, and sorting would hide a genuine rewrite |
| `lazy_attributes` | list[str] | Names resolved by the module-level `__getattr__`, parsed from the tuple or set it tests against. A name in `all` but not importable eagerly is a real distinction for a consumer doing `from mojoboost import x` at module scope |
| `shared_estimator_parameters` | dict[str, value] | `_Base.__init__` keyword arguments and their defaults, with module-level `Name` defaults resolved (section 5.2) |
| `estimators` | dict | One entry per public estimator class; see 3.1 |
| `fitted_attributes` | list[str] | `_FITTED_ATTRS` on `_Base`, in source order. Order is the reset order |
| `parameter_aliases` | dict[str, dict] | One entry per alias; see 3.2 |
| `callbacks` | dict | See 3.3 |
| `eval_metric_names` | dict | `_METRICS` and `_ALIASES` keys from `python/mojoboost/_eval.py`, keys only (section 5.1) |
| `functional_api` | dict | `__all__`, `train`'s signature, and the public method names of `Dataset` and `Booster` in `python/mojoboost/basic.py` |
| `inspection` | dict | `__all__`, `DUMP_FORMAT_VERSION`, `SUPPORTED_MODEL_FORMAT_VERSIONS`, and the `OBJECTIVE_NAMES` mapping from `python/mojoboost/inspection.py` |

### 3.1 `python.estimators.<Class>`

| Key | Type | Meaning |
|---|---|---|
| `bases` | list[str] | Base class names as written |
| `own_parameters` | dict[str, value] | Keyword arguments of this class's own `__init__`, defaults resolved |
| `methods` | dict[str, dict] | Public methods. Each maps argument name to its default, with `"required"` for an argument with none. Order is preserved by insertion; the JSON object records it |
| `properties` | list[str] | Names decorated `@property` |
| `classmethods` | list[str] | Names decorated `@classmethod` |

Argument defaults are recorded because section 4.3 of the compatibility
policy makes a default change breaking. A snapshot that recorded only
argument names would pass a release that silently changed `num_leaves`.

### 3.2 `python.parameter_aliases`

There is no alias table in `python/mojoboost/__init__.py`. The pairs are
expressed as calls, `self._resolve_alias(canonical, alias, default)`, in
`_Base._params` and in the continued-training path. The tool walks the
`_Base` class body for `Call` nodes whose `func.attr` is `_resolve_alias`
and reads the three arguments.

| Key | Type | Meaning |
|---|---|---|
| `canonical` | str | First argument. The guaranteed spelling |
| `fallback_default` | value | Third argument, with a module-level `Name` resolved as in 5.2. The value used when neither spelling is set |
| `sites` | int | How many call sites express this pair |

`fallback_default` is recorded because it is a second place a default
lives. If the constructor says `min_data_in_leaf=20` and the resolver says
`20`, they agree today, and nothing but this field would notice if one of
them moved. `sites` is recorded for the same reason: the pair is expressed
twice today, and a change applied to one site and not the other is a
silent divergence between a first fit and a continued one.

An alias whose two occurrences disagree on `fallback_default` is an error,
not a merge, and the tool reports both values.

### 3.3 `python.callbacks`

| Key | Type | Meaning |
|---|---|---|
| `all` | list[str] | `__all__` of `python/mojoboost/callback.py` |
| `env_fields` | list[str] | The `CallbackEnv` field list, **in order**. It is a `namedtuple`, so positional unpacking is a caller-visible contract and the order is the contract |
| `resettable` | list[str] | `RESETTABLE`, in order. The order is a wire format |
| `reset_slots` | int | `comptime RESET_SLOTS` in `bindings/_mojoboost.mojo` |
| `reset_aliases` | dict[str, str] | `_RESET_ALIASES` |
| `integral_slots` | list[str] | `_INTEGRAL`, sorted. A slot that must round trip as a whole number through the float64 buffer |
| `factories` | dict[str, dict] | Each factory's arguments and defaults |

`env_fields` and `resettable` are the two lists in the whole snapshot
whose **order** is the contract rather than a convenience. Both are stored
in source order and both are compared order-sensitively, and the tool says
so in the failure message when one moves, because a reordering here
produces wrong numbers rather than an error.

## 4. `mojo`, `c_abi`, `parameter_string`, `environment`, `platforms`

### 4.1 `mojo`

| Key | Type | Source |
|---|---|---|
| `exports_by_module` | dict[str, list[str]] | The `from .mod import ...` blocks of `src/mojoboost/__init__.mojo`. Names sorted within a module; module keys sorted by the JSON writer |
| `export_count` | int | Total names. A scalar that moves when any module's list does, so a truncated parse is visible |
| `objective_codes` | dict[str, int] | `comptime NAME = <int>` in `src/mojoboost/boosting.mojo` and `params.mojo` for the objective constants |

### 4.2 `c_abi`

| Key | Type | Source |
|---|---|---|
| `abi_version` | int | `MOJOBOOST_ABI_VERSION` |
| `defines` | dict[str, int] | Every `#define MOJOBOOST_*` with an integer value, parenthesized negatives included |
| `opaque_types` | list[str] | `typedef struct ... *` handle types |
| `functions` | list[str] | One normalized declaration per function: return type, name, and parameter list with comments stripped and runs of whitespace collapsed to one space. Sorted by name |

Declarations are normalized rather than captured verbatim so that
rewrapping a long signature across lines is not a diff. A parameter name
change **is** a diff, deliberately: the name is documentation a caller
reads, and losing it silently is the kind of change this file exists to
show.

### 4.3 `parameter_string`

| Key | Type | Source |
|---|---|---|
| `supported_keys` | list[str] | `comptime SUPPORTED_KEYS` in `src/mojoboost/params.mojo`, split on commas and stripped. Source order preserved |
| `mojo_api_only_keys` | list[str] | The keys the parser reports as unsupported-with-a-reason rather than unknown |

### 4.4 `environment`

| Key | Type | Source |
|---|---|---|
| `declared` | list[str] | The variables section 9.5 of the compatibility policy documents, parsed from that table |
| `observed` | list[str] | Every double-quoted `"MOJOBOOST_*"` string literal in code under `src/`, `bindings/`, `python/`, `capi/`, and `cli/` |
| `read_directly` | list[str] | The subset that is the literal first argument of a `getenv(...)` or `os.environ.get(...)` call |
| `undeclared` | list[str] | `observed` minus `declared`. **Not an error.** A variable may be a diagnostic knob rather than a public control |
| `stale` | list[str] | `declared` minus `observed`. **An error.** A documented variable nothing reads is a promise the code does not keep |

`observed` is the double-quoted-literal scan and not the `getenv` call
scan, and the difference is load-bearing. `MOJOBOOST_NUM_WORKERS` and
`MOJOBOOST_PARALLEL_MIN_OPS`, two of the seven the policy documents, are
never passed to `getenv` directly: `src/mojoboost/parallel.mojo` reads
them through `_env_int(name, default)`, so `getenv` sees a computed name.
A call scan finds neither, marks both `stale`, and fails invariant I8 on a
tree that is correct. The literal scan finds both.

What the literal scan costs is precision in the other direction: a
variable named in a double-quoted string for some reason other than
reading it counts as observed. Prose mentions are not a problem in
practice, because this repository's docstrings name variables in
backticks, and `read_directly` is recorded separately so the two
populations stay distinguishable.

A variable whose name is genuinely computed, with no literal anywhere, is
invisible to both scans. That is a real limitation and it is why `stale`
is the error and `undeclared` is not: the tool can prove a documented
variable is unread only in the weak sense of finding no literal, so the
finding it reports loudest is the one it can be most confident about.

### 4.5 `platforms`

| Key | Type | Source |
|---|---|---|
| `declared` | list[str] | `pixi.toml [workspace] platforms` |
| `ci_runners` | list[str] | The `matrix.runner` lists in `.github/workflows/ci.yml`, deduplicated and sorted. These are the tier 1 rows' evidence |
| `tiers` | dict[str, int] | Carried, not derived. Section 7 |

## 5. Parsing rules the tool must follow

These are load-bearing. Each one silently produces a wrong value rather
than an error, which is the worst failure mode for a tool whose job is
detecting change. The first four are inherited from
`handoffs/task20_compatibility.md`, which found them by writing the
throwaway cross-check; the rest were found by this lane.

1. **`_METRICS` in `_eval.py` is not `literal_eval`-able.** Its values are
   module-level integer constants, so `ast.literal_eval` on the whole dict
   raises. Read `node.value.keys` and evaluate those individually.
   `_ALIASES` and `_RESET_ALIASES` are pure literals and read whole.
2. **Some estimator defaults are named constants.** `_Base.__init__` has
   `lambda_l2=_LAMBDA_L2` and `lambda_l1=_LAMBDA_L1`. A naive reader
   records the string `"_LAMBDA_L2"` and then either reports drift forever
   or never notices when the constant's value changes. Resolve
   module-level `Name` defaults against the module's own assignments, and
   record the resolved value. An unresolvable `Name` goes to `null` and
   into `meta.underived`, never to its own spelling.
3. **`CallbackEnv` is a `namedtuple()` call, not a class.** The field list
   is the second positional argument of the call.
4. **Mojo exports come in two spellings**, parenthesized multi-line and
   bare single-line. A regex handling only the first silently drops whole
   modules. `tools/check_parity.py:mojo_export_names()` already handles
   both, and the tool imports it rather than writing a second parser; see
   section 6.4.
5. **Negative `#define`s are parenthesized.** `MOJOBOOST_ERROR_IO` is
   `(-3)`, not `-3`. A regex anchored on an optional minus finds `3`.
6. **`SUPPORTED_KEYS` is one implicitly concatenated string literal split
   across lines**, and the commas that separate keys sit at the ends of
   the fragments. Join the fragments before splitting on commas, or the
   key that straddles a fragment boundary is lost.
7. **The model format's readable set is a chain of `if token == "vN"`
   returns**, not a list. Parse the chain. A build that adds v5 to the
   writer and forgets the reader is exactly what this field catches.
8. **`__all__` in `__init__.py` is interleaved with comments and is not
   sorted.** Take the `ast.List` elements in order and keep that order.
   Sorting it here would make a rewrite invisible.

## 6. Invariants the tool enforces

`--check` and `--write` both run these. A violation is reported and exits
non-zero even when the snapshot itself matches, because these are facts
about the tree rather than facts about the file.

| # | Invariant | Why |
|---|---|---|
| I1 | The three library version locations hold the same string | Compatibility policy section 1.1 |
| I2 | `model_format_writer_token` equals `"v" + str(model_format_writer)` | Two constants in one file naming one version |
| I3 | `model_format_writer` is in `model_format_readable` | A build that cannot read what it writes |
| I4 | `model_format_dump_reports` equals `model_format_writer` | `serialize.mojo` says `model_dump.mojo` has to track it |
| I5 | `max(model_format_python_reads)` is at least `model_format_writer` | The pure-Python parser must read what the writer writes, or `dump_model` fails on a model the same build just saved |
| I6 | `len(callbacks.resettable)` equals `callbacks.reset_slots` | A slot count and a name list that disagree misassign parameters silently |
| I7 | Every name in `callbacks.resettable` appears in the reset slot order in `bindings/_mojoboost.mojo`, in the same position | Compatibility policy section 9.3; a reordering produces wrong numbers |
| I8 | `environment.stale` is empty | A documented variable nothing reads |
| I9 | Every `deprecations.toml` entry in state `removed` names something absent from the snapshot, and every entry in state `soft` or `deprecated` names something present | The register and the tree cannot disagree about existence |
| I10 | No `deprecations.toml` entry has a `remove_in` that violates the overlap floor against its `since` | Deprecation policy section 1 |
| I11 | The union of `mojo.exports_by_module` values equals `check_parity.mojo_export_names()` | Two parsers over one file must agree, or one of them is dropping a module |

I5 and I11 are the two that catch a whole class of bug rather than one
bug. I5 is currently violated; see `DRIFT_REPORT.md`.

### 6.4 On importing `check_parity`

`tools/api_snapshot.py` imports `tools/check_parity.py` as a module and
calls `mojo_export_names()`. That import is safe: `check_parity` guards
`main()` behind `if __name__ == "__main__"`, and its module level does
nothing but define constants and compile regexes. Reusing it is the point.
Two independent parsers over `src/mojoboost/__init__.mojo` would drift,
and invariant I11 is what turns the reuse into a check rather than a
coincidence. If the import fails for any reason the tool degrades: I11
goes to `meta.underived` and the run continues, because a snapshot that
refuses to generate is worse than one that generates with a named gap.

## 7. The two blocks that are carried, not derived

`platforms.tiers` and `numerical_contracts` cannot be parsed out of
anything. A tier is a claim about what evidence exists, and a numerical
contract is a claim about behavior; neither is a fact about a token
stream.

`--write` reads both from the existing file, carries them forward
unchanged, lists their paths in `meta.carried`, and prints a reminder that
they were carried rather than derived. When no previous file exists they
are seeded from
[docs/COMPATIBILITY_POLICY.md](../docs/COMPATIBILITY_POLICY.md) sections
10.2 and 8.3 by hand, once, and then carried forever.

They are in the snapshot rather than left out because a tier drop and a
broken numerical guarantee are both breaking changes under section 11.2 of
the compatibility policy, and a diff that cannot show them is not a
complete answer to "what changed for a caller".

## 8. What changed from schema version 1

Version 1 is `tests/parallel/api_snapshot_manifest.json`: hand-written,
`status: "proposed"`, cross-checked once by a throwaway script that no
longer exists. It was a good draft of the shape and it is wrong about the
contents now, in the ways `DRIFT_REPORT.md` lists.

| Change | Why |
|---|---|
| `meta` block introduced; `about` and `verification` dropped | A generated file does not need a paragraph explaining that it was written by hand. `meta.underived` and `meta.carried` carry the honest version of what `verification.not_checked` was for |
| One model format version became five fields | The single field hid a four-way disagreement between the writer, the reader, the dump reporter, and the Python parser |
| `environment` split into `declared`, `observed`, `undeclared`, `stale` | Seven variables were documented and roughly forty are read. A flat list could not say that |
| `python.lazy_attributes` added | `__all__` now contains names resolved by a module-level `__getattr__`. Eager and lazy are different for a consumer and the old shape could not distinguish them |
| `c_abi.defines` added, `status_codes` folded into it | The header grew device and predict-mode constants that the old `status_codes` key had no room for |
| Argument defaults recorded for every public method, not just the estimator constructors | A default change is breaking under policy section 4.3 wherever it happens |
| `deprecations` cross-check added (I9, I10) | The register did not exist when version 1 was written |
| `status` fixed to `"generated"` | A hand-written snapshot is not a snapshot |

Version 1 is not migrated. The tool generates version 2 from the source
and the old file is superseded wholesale, which is what its own handoff
said would happen.
