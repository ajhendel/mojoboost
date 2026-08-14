"""EXPERIMENTAL: read and write LightGBM's own model files.

This module is glue and nothing else. Every line of the conversion --
parsing LightGBM's text, synthesizing the bin edges its thresholds imply,
rebuilding the category tables its `cat_threshold` bitsets imply, mapping
objectives, folding the shrinkage and the base score, and refusing what
cannot be represented -- is in `src/mojoboost/lgbm_model_io.mojo`. Nothing
here parses a model file, and nothing here decides what is convertible.
Both would be a second implementation in a second language of facts Mojo
already owns, which is exactly the drift `mojoboost.inspection` is already
paying for and this module will not repeat.

    from mojoboost import lgbm_model_io

    print(lgbm_model_io.unsupported_reason("model.txt") or "convertible")
    booster = lgbm_model_io.load_lightgbm_model("model.txt")

**LightGBM's format is not mojoboost's.** `Booster.save_model` and
`Booster.model_to_string` speak mojoboost's own versioned format, which
stores floats as raw bit patterns so a round trip is bit-exact, and that is
the only format this project persists models in. LightGBM's text is an
interchange: something to import from and export to, never something to save
to. Both conversions here go *through* a mojoboost model file for that
reason -- an import's product is a native file, which is also what forces
every converted model through native serialization before anything can use
it.

**Experimental** means what `interop_status()` says it means: the converter
has been exercised against hand-written fixtures only, and no file a real
LightGBM build wrote has ever been read by it. Treat a conversion as
something to check, not something to rely on. The first use in a process
warns, once.

Two caveats survive every conversion and are worth repeating wherever this
is surfaced:

1. A model imported from LightGBM carries the bin edges its trees split on,
   which is all a model file holds. It predicts identically on raw feature
   values and it is *not* the binning LightGBM fit, so it must not be used
   to continue training. `BinMapper.matches` enforces that natively.
2. Exporting a model whose learning rate is not already 1.0 is accurate to a
   few units in the last place rather than bit-exact, because LightGBM's
   format keeps the shrinkage inside the leaf values and rounding it there
   is unavoidable. The export report says which case a given file is.

The native boundary
-------------------
Every entry point below reaches Mojo through one of four extension
functions, and the module works only as far as the installed build exposes
them. A build without them raises `LightGBMInteropUnavailable` naming the
missing entry point rather than falling back to a Python conversion, because
a Python conversion is the thing this module exists not to be. The exact
signatures the bindings owe are recorded in
`handoffs/connect_16_lgbm_interop.md`.
"""

import os as _os
import tempfile as _tempfile
import warnings as _warnings

__all__ = [
    "EXPERIMENTAL",
    "LightGBMInteropUnavailable",
    "interop_status",
    "unsupported_reason",
    "convert_to_mojoboost",
    "convert_from_mojoboost",
    "load_lightgbm_model",
    "save_lightgbm_model",
]

#: This module's API may change or be withdrawn without a deprecation
#: period. It is a flag rather than a docstring sentence so a caller can
#: gate on it.
EXPERIMENTAL = True

#: The extension entry points this module needs, and what each one is in
#: `src/mojoboost/lgbm_model_io.mojo`. Kept as data so the error message a
#: missing one produces names the Mojo function a builder has to bind.
_ENTRY_POINTS = {
    "lgbm_interop_status": "lgbm_interop_status()",
    "lgbm_file_unsupported_reason": "lgbm_file_unsupported_reason(path)",
    "lgbm_import_file": "import_lgbm_file(lgbm_path, model_path)",
    "lgbm_export_file": "export_lgbm_file(model_path, lgbm_path)",
}

_WARNED = False


class LightGBMInteropUnavailable(RuntimeError):
    """This build cannot convert LightGBM model files.

    Raised instead of converting in Python. The message names the extension
    entry point that is missing and the Mojo function behind it, so the gap
    is a build to fix rather than a feature to reimplement here.
    """


def _native(name):
    """The extension entry point called `name`, or None when this build has
    no such function. Imported inside the call so that importing this module
    costs nothing and works without the extension present."""
    try:
        from . import _mojoboost
    except ImportError:
        return None
    return getattr(_mojoboost, name, None)


def _require(name):
    hook = _native(name)
    if hook is None:
        raise LightGBMInteropUnavailable(
            f"this mojoboost build does not expose {name!r}, so LightGBM "
            "model files cannot be converted. It binds "
            f"{_ENTRY_POINTS.get(name, name)} from "
            "src/mojoboost/lgbm_model_io.mojo; see "
            "handoffs/connect_16_lgbm_interop.md for the binding contract. "
            "There is no Python fallback: converting a LightGBM model in "
            "Python would be a second implementation of the converter."
        )
    return hook


def _warn_once():
    """Say once per process that this is an experiment.

    Once, and not per call: a conversion loop over a directory of models
    should not bury its own output, and the fact does not change between
    calls.
    """
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    try:
        status = interop_status()
    except LightGBMInteropUnavailable:
        return
    _warnings.warn(
        f"mojoboost LightGBM model interop is experimental. {status}",
        UserWarning,
        stacklevel=3,
    )


def _report(raw):
    """A conversion report as a plain dict.

    The binding is expected to hand back a mapping of the fields
    `LgbmImportReport` / `LgbmExportReport` carry. A build that returns the
    struct's rendered form instead is accepted as a one-key summary rather
    than refused: the report is information about a conversion that already
    succeeded, so failing on its shape would turn a cosmetic gap into a
    failed import.
    """
    if raw is None:
        return {}
    if hasattr(raw, "keys"):
        return {str(k): raw[k] for k in raw.keys()}
    return {"summary": str(raw)}


def interop_status():
    """How far this interop has been validated, in Mojo's own words.

    Surfaced verbatim rather than paraphrased: it is the single sentence
    that says the feature is an experiment, and it changes when the
    differential fixtures against a real LightGBM build are run.
    """
    return str(_require("lgbm_interop_status")())


def unsupported_reason(lightgbm_path):
    """Why `lightgbm_path` cannot be converted, or `""` when it can.

    Never raises for an unconvertible or unreadable file -- that is the
    point of it. It is answered by running the native conversion and
    reporting what it said, so it cannot drift from what a conversion
    actually accepts. `LightGBMInteropUnavailable` is still raised when the
    build has no converter at all, which is a different question.
    """
    return str(
        _require("lgbm_file_unsupported_reason")(_os.fspath(lightgbm_path))
    )


def convert_to_mojoboost(lightgbm_path, model_path):
    """Convert a LightGBM model file into a mojoboost model file.

    Returns the conversion report as a dict: how many features and trees
    crossed, the objective, how many features turned out categorical, how
    many bin edges had to be synthesized, and whether the trees carry the
    node covers feature contributions need. `model_path` is written in
    mojoboost's own format and is what `Booster(model_file=...)` reads.
    """
    _warn_once()
    return _report(
        _require("lgbm_import_file")(
            _os.fspath(lightgbm_path), _os.fspath(model_path)
        )
    )


def convert_from_mojoboost(model_path, lightgbm_path):
    """Convert a mojoboost model file into a LightGBM model file.

    Returns the conversion report as a dict, including the objective setting
    that could not be written (`huber`'s `alpha` and its three siblings are
    not kept on a fitted model) and whether the file's predictions are
    bit-identical to the model's.
    """
    _warn_once()
    return _report(
        _require("lgbm_export_file")(
            _os.fspath(model_path), _os.fspath(lightgbm_path)
        )
    )


def load_lightgbm_model(lightgbm_path):
    """A `mojoboost.Booster` holding the model in a LightGBM model file.

    The conversion lands in a mojoboost model file first and the booster is
    read from that, which is deliberate: it means the imported model has
    been through native serialization -- category tables, node covers,
    missing-bin reservations and all -- before anything predicts with it,
    and it means this function needs no model handle to cross the language
    boundary.

    Whether the file is single-output or softmax comes from the file, the
    same way it does for a native one.
    """
    from .basic import Booster

    with _tempfile.TemporaryDirectory() as tmp:
        native = _os.path.join(tmp, "converted.mbst")
        convert_to_mojoboost(lightgbm_path, native)
        return Booster(model_file=native)


def save_lightgbm_model(booster, lightgbm_path):
    """Write a fitted `mojoboost.Booster` as a LightGBM model file.

    Returns the conversion report as a dict. Raises for a model LightGBM's
    format cannot hold -- a custom objective has nowhere to record its link
    function -- and reports, rather than hides, the objective setting a
    fitted model no longer carries.

    This is not a save path. `booster.save_model(path)` is, and it is the
    one that round-trips exactly; this writes an interchange file for
    something else to read.
    """
    with _tempfile.TemporaryDirectory() as tmp:
        native = _os.path.join(tmp, "model.mbst")
        booster.save_model(native)
        return convert_from_mojoboost(native, lightgbm_path)
