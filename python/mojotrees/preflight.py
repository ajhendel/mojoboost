"""Native pre-flight checks: what a fit asks the extension before it converts
any data, and the same questions offered to a caller (`mojotrees.preflight`).

The extension validates every parameter again inside `_parse_params` when
the fit is dispatched, with the same checkers; these readers ask the same
questions first, so a bad value is named before the matrix is copied and
binned. Nothing here rejects what the trainer would accept, and nothing
here decides a device or a trainer.

- `check_forced_splits`: validate a forced-splits document against a
  feature count and a growth budget, and report its shape
  (`forced_splits_check`).
- `native_preflight`: the extra tree parameters and the bundling knobs of
  a params dict, against the run they belong to (`extra_params_check`,
  `efb_check`).
- `unimplemented_option_message`: the native "not implemented, here is
  what it would take" message for a LightGBM option name the repository
  knows and does not implement (`extra_option_supported`), or None.
- `bundling_defaults` / `bundling_knobs`: the exclusive-feature-bundling
  knobs' defaults as the extension states them (`efb_defaults`), and a
  fit's knobs with `None` filled from them.
"""

import json as _json

from . import _mojotrees

__all__ = [
    "bundling_defaults",
    "bundling_knobs",
    "check_forced_splits",
    "native_preflight",
    "unimplemented_option_message",
]


def _document_text(document):
    if isinstance(document, bytes):
        return document.decode("utf-8")
    if isinstance(document, str):
        return document
    if isinstance(document, (dict, list)):
        return _json.dumps(document)
    raise TypeError(
        "a forced-splits document is its text, or a dict or list to "
        f"serialize as JSON, not {type(document).__name__}"
    )


def check_forced_splits(document, n_features, num_leaves=31, max_depth=-1):
    """Validate a forced-splits document and return `{"n_nodes", "depth"}`.

    `document` is the JSON text LightGBM's `forcedsplits_filename` points
    at (read the file and pass the text), or a dict / list to serialize.
    Every error is the native parser's, naming the byte it stopped at, the
    feature index out of range, or the budget the document does not fit.
    """
    try:
        out = _mojotrees.forced_splits_check(
            _document_text(document),
            int(n_features),
            int(num_leaves),
            int(max_depth),
        )
    except Exception as exc:
        raise ValueError(str(exc)) from None
    return {"n_nodes": int(out["n_nodes"]), "depth": int(out["depth"])}


def native_preflight(params, n_features, device):
    """Run the extra-parameter and bundling checks on a fit's params dict.

    `params` is the dict `_Base._params` builds; `n_features` the matrix
    width; `device` the requested device name. Returns the routing facts
    `extra_params_check` reports (`is_active`, `needs_leaf_finish`,
    `needs_node_identity`, `needs_grower_support`). Raises the native
    message for a value out of range, a per-feature vector of the wrong
    length, a forced-splits document that does not fit the budget, or
    bundling requested on a device that cannot honor it.
    """
    shape = {
        "n_features": int(n_features),
        "num_leaves": int(params.get("num_leaves", 31)),
        "max_depth": int(params.get("max_depth", -1)),
        "min_data_in_leaf": int(params.get("min_data_in_leaf", 20)),
    }
    try:
        facts = _mojotrees.extra_params_check(params, shape)
        _mojotrees.efb_check(params, 1 if str(device) == "cpu" else 0)
    except Exception as exc:  # the extension's message, as a ValueError
        raise ValueError(str(exc)) from None
    return {str(k): bool(facts[k]) for k in facts}


def unimplemented_option_message(name):
    """The native message for a LightGBM option that is real but not
    implemented, or None for any other name (including names this
    repository has never heard of; "unknown parameter" is the caller's
    message about the caller's parameter, not this function's to give)."""
    try:
        _mojotrees.extra_option_supported(str(name))
    except Exception as exc:  # the extension raises with the native text
        return str(exc)
    return None


_BUNDLING_INT = ("max_bundle_bins", "max_bundle_size")
_BUNDLING_FLAG = ("bundle_missing",)


def bundling_defaults():
    """The exclusive-feature-bundling knobs' defaults, from the extension:
    `max_conflict_rate`, `max_bundle_bins`, `max_bundle_size`,
    `max_nondefault_rate`, `min_reduction`, `bundle_missing`. LightGBM's
    numbers, stated once, in src/mojotrees/efb.mojo."""
    d = _mojotrees.efb_defaults()
    out = {}
    for key in d:
        key = str(key)
        if key in _BUNDLING_INT:
            out[key] = int(d[key])
        elif key in _BUNDLING_FLAG:
            out[key] = bool(d[key])
        else:
            out[key] = float(d[key])
    return out


def bundling_knobs(**given):
    """The bundling knobs a fit sends, typed the way `efb_check` reads
    them, with every knob given as `None` taken from `bundling_defaults`.
    Unknown knob names raise."""
    defaults = bundling_defaults()
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ValueError(f"unknown bundling knob(s): {', '.join(unknown)}")
    out = {}
    for key, default in defaults.items():
        value = given.get(key)
        value = default if value is None else value
        if key in _BUNDLING_INT:
            out[key] = int(value)
        elif key in _BUNDLING_FLAG:
            out[key] = int(bool(value))
        else:
            out[key] = float(value)
    return out
