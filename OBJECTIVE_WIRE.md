# OBJECTIVE_WIRE.md, lane/objective-codes

One edit, in one file this lane does not own. Everything else the reserved
objective names needed is inside the registry and lands with the commit.

Read this before applying: **the Python side needed no edit at all**, and
that is a finding rather than an omission. `python/mojotrees/_fit_args.py`
does not carry an objective name table. `_unimplemented_objectives()` at
`python/mojotrees/_fit_args.py:65-73` walks `_mojotrees.
registry_objective_unimplemented()`, which is `bindings/objective_bindings.
mojo:132-150`, which walks `unimplemented_objective_alias_names()` in the
registry. That list is the one this lane extended, so the fifteen reserved
spellings reach the Python estimators' error text without a line of Python
changing. `_objective_status` at `_fit_args.py:87-96` lowercases before it
asks (`name.strip().lower()`), which is what makes `objective="Cox"` and
`objective="QueryRMSE"` resolve case-insensitively, as
`docs/PARAMETER_NAMING.md` requires. Verified by reading both files after
the registry change, not inferred.

`python/mojotrees/inspection.py` DID need a change and it is mine, so it is
in the commit rather than here: `OBJECTIVE_NAMES` listed thirteen of the
twenty-one assigned codes.

---

## The one edit: `src/mojotrees/params.mojo`

**Not mine** (`params.mojo` is on this lane's not-yours list).

### Why

`params.objective_from_name` is the parameter-string path, and it does not
delegate to the registry: it carries its own name table
(`_objective_from_lower`, `params.mojo:265`) and its own
not-implemented list (`_raise_if_unimplemented_objective`,
`params.mojo:382-412`). The registry's module docstring already records this
mirror and names the deletion that removes it.

That mirror is missing all seven reserved names. Traced:

- `params.mojo:382` `def _raise_if_unimplemented_objective(name: String)
  raises:` handles exactly three groups: `cross_entropy_lambda`/`xentlambda`
  (`:390`), `multiclassova`/`multiclass_ova`/`ova`/`ovr`/
  `multiclassoneversusall` (`:397`), `rank_xendcg`/`xendcg` (`:407`).
- Nothing in `params.mojo` matches `cox`, `queryrmse`, `pairlogit`,
  `yetirank`, `survivalaft`, `multirmse` or any underscore spelling of them.
  Checked with `grep -n "queryrmse\|pairlogit\|yetirank\|\"cox\"\|multirmse\|
  survivalaft" src/mojotrees/params.mojo` -> no matches.
- So `parse_params("objective=YetiRank")` falls through `:373` to the raise
  at `params.mojo:374-379`, which says: `unknown objective 'yetirank';
  expected regression, binary, multiclass, poisson, huber, quantile, mae,
  gamma, tweedie, mape, fair, or cross_entropy`.

**This is a refusal, not silence.** No parameter string, no binding and no
Python entry point maps a reserved name onto a live objective code; I
checked `params.mojo`, `lgbm_model_io.mojo`, `bindings/*.mojo` and
`python/mojotrees/*.py` for every reserved spelling and there are no
matches. Nothing can be misrouted. What is wrong is only the sentence: a
loss whose gradients, pair generation and round loop are all merged is
reported to the user as an unknown name.

### The exact edit

In `src/mojotrees/params.mojo`, inside `_raise_if_unimplemented_objective`.

**Current text** (`params.mojo:407-412`, the last arm of the function):

```mojo
    if name == "rank_xendcg" or name == "xendcg":
        raise Error(
            "objective 'rank_xendcg' is not implemented; 'lambdarank' is"
            " the ranking objective mojotrees provides"
        )
```

**Replacement:**

```mojo
    if name == "rank_xendcg" or name == "xendcg":
        raise Error(
            "objective 'rank_xendcg' is not implemented; 'lambdarank' is"
            " the ranking objective mojotrees provides"
        )
    # The seven RESERVED objective codes. Their trainers are merged and
    # nothing imports them, so the refusal names the module and the entry
    # point rather than calling a real loss unknown. The registry is the
    # single statement of both the name set and the sentence; this arm
    # delegates rather than repeating it, which is the reduction the
    # registry's module docstring already asks params.mojo for.
    var reserved = objective_unimplemented_canonical(name)
    if reserved.byte_length() > 0:
        raise Error(
            "objective '",
            reserved,
            "' is not implemented; ",
            objective_unimplemented_reason(name),
        )
```

This arm subsumes the three above it as well, so if the applier prefers, the
whole body of `_raise_if_unimplemented_objective` can be replaced by the six
lines from `var reserved` on. Two cautions before doing that:

1. `params.mojo:397` accepts `multiclassoneversusall`, CatBoost's spelling,
   and the registry's `objective_unimplemented_canonical` does **not**. If
   the body is collapsed, that spelling stops being recognized. Either keep
   the `multiclassova` arm or add `multiclassoneversusall` to the registry
   chain first; the registry side of that is mine and I did not do it,
   because widening a name set nobody asked me to widen is not this lane's
   change.
2. The three existing messages and the registry's are word for word the
   same. I diffed them: `params.mojo:391-395` against
   `objective_registry.objective_unimplemented_reason`'s
   `cross_entropy_lambda` arm, and the same for the other two. Collapsing
   changes no user-visible sentence for those three.

### Imports the edit needs

`params.mojo:80` is the only registry import today and it is the one-name
form:

```mojo
from .objective_registry import MULTICLASS as _MULTICLASS
```

Add one line after it:

```mojo
from .objective_registry import (
    objective_unimplemented_canonical,
    objective_unimplemented_reason,
)
```

Do not fold the two into one statement without checking: `MULTICLASS` is
imported under an alias and rebound at `params.mojo:98` as `comptime
MULTICLASS = _MULTICLASS`, deliberately, so that the name this module's
callers import is defined in this module. A merge that drops the alias
changes what `params.MULTICLASS` is.

### How to verify it worked

The failure mode to watch for is a silent no-op, so check the sentence and
not the exception:

```
parse_params("objective=YetiRank")
```

Before: `unknown objective 'yetirank'; expected regression, binary, ...`
After: `objective 'yeti_rank' is not implemented; its gradients, its pair
generation and its round loop are implemented in
src/mojotrees/catboost_ranking.mojo (train_catboost_ranker), which nothing
imports: ...`

If the message still starts with `unknown objective`, the new arm is
unreachable, which would mean it was placed after the final `raise` at
`params.mojo:374` rather than inside `_raise_if_unimplemented_objective`.
Note that `objective_from_name` lowercases at `params.mojo:261` before
dispatching, so the arm must compare lowercase and the example above must be
spelled `YetiRank` at the call site to exercise that.
