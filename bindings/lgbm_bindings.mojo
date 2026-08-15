"""LightGBM model-file interop: the four entry points `mojotrees.lgbm_model_io`
binds.

    lgbm_interop_status()                       -> str
    lgbm_file_unsupported_reason(path)          -> str ("" when convertible)
    lgbm_import_file(lgbm_path, model_path)     -> dict (LgbmImportReport)
    lgbm_export_file(model_path, lgbm_path)     -> dict (LgbmExportReport)

Every line of the conversion is in `src/mojotrees/lgbm_model_io.mojo`;
these functions cross two paths and hand the report back as a plain dict
of the struct's fields, which is the shape `lgbm_model_io._report` reads.
The status text crosses verbatim, as that module asks: it is the sentence
that says the interop is an experiment, and Python must not paraphrase it.

Both conversions go through a mojotrees model file, never a live model, so
nothing here holds a handle: an import writes the native file the Python
`Booster` loads with `model_file=`, and an export reads one.
"""

from std.python import PythonObject

from binding_support import py_dict, py_str_list

from mojotrees.lgbm_model_io import (
    export_lgbm_file as mojo_export_lgbm_file,
    import_lgbm_file as mojo_import_lgbm_file,
    lgbm_file_unsupported_reason as mojo_lgbm_file_unsupported_reason,
    lgbm_interop_status as mojo_lgbm_interop_status,
)


def lgbm_interop_status() raises -> PythonObject:
    """How far the interop has been validated, in the native module's words."""
    return PythonObject(mojo_lgbm_interop_status())


def lgbm_file_unsupported_reason(path: PythonObject) raises -> PythonObject:
    """Why the LightGBM model file at `path` cannot be converted, or `""`.
    Never raises for the file's sake; an unreadable file is a reason too."""
    return PythonObject(mojo_lgbm_file_unsupported_reason(String(py=path)))


def lgbm_import_file(
    lgbm_path: PythonObject, model_path: PythonObject
) raises -> PythonObject:
    """Convert a LightGBM model file into a mojotrees model file; returns
    the import report as a dict."""
    var report = mojo_import_lgbm_file(
        String(py=lgbm_path), String(py=model_path)
    )
    var out = py_dict()
    out["n_features"] = PythonObject(report.n_features)
    out["n_trees"] = PythonObject(report.n_trees)
    out["n_classes"] = PythonObject(report.n_classes)
    out["objective"] = PythonObject(report.objective)
    out["objective_line"] = PythonObject(report.objective_line)
    out["format_version"] = PythonObject(report.format_version)
    out["n_categorical_features"] = PythonObject(
        report.n_categorical_features
    )
    out["n_widened_tables"] = PythonObject(report.n_widened_tables)
    out["n_edges"] = PythonObject(report.n_edges)
    out["n_missing_reservations"] = PythonObject(
        report.n_missing_reservations
    )
    out["has_node_counts"] = PythonObject(report.has_node_counts)
    out["feature_names"] = py_str_list(report.feature_names)
    return out^


def lgbm_export_file(
    model_path: PythonObject, lgbm_path: PythonObject
) raises -> PythonObject:
    """Convert a mojotrees model file into a LightGBM model file; returns
    the export report as a dict."""
    var report = mojo_export_lgbm_file(
        String(py=model_path), String(py=lgbm_path)
    )
    var out = py_dict()
    out["n_trees"] = PythonObject(report.n_trees)
    out["n_classes"] = PythonObject(report.n_classes)
    out["objective_line"] = PythonObject(report.objective_line)
    out["dropped_objective_param"] = PythonObject(
        report.dropped_objective_param
    )
    out["n_categorical_features"] = PythonObject(
        report.n_categorical_features
    )
    out["shrinkage_folded"] = PythonObject(report.shrinkage_folded)
    out["base_score_folded"] = PythonObject(report.base_score_folded)
    out["n_feature_names"] = PythonObject(report.n_feature_names)
    return out^
