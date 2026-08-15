"""What this installation can say about itself.

`gpu_available`, `build_info`, and `show_versions`, moved here from the
package `__init__` in the consolidation round. Every name is bound back
into the package namespace, so `mojotrees.build_info` and
`from mojotrees import _optional_dependency_versions` are unchanged, and
`build_info()["package"]` still names the package `__init__.py`.
"""

import os as _os
import sys as _sys

from . import _compat

_mojotrees = _compat.import_extension()


def _package():
    """The `mojotrees` package module, for the version and file it owns."""
    return _sys.modules[__package__]


def gpu_available():
    """True when this build can train on an accelerator. False on a
    CPU-only build and when `MOJOTREES_DISABLE_GPU=1` is set."""
    return bool(_mojotrees.gpu_available())


#: Optional dependencies worth reporting in a bug report, by distribution
#: name rather than import name (`scikit-learn`, not `sklearn`). None is
#: required to import mojotrees or to fit a model.
_OPTIONAL_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "dask",
    "polars",
    "pyarrow",
)


def _distribution_version():
    """The version recorded for the installed `mojotrees` distribution, or
    None when this package is not an installed distribution at all, which
    is the normal answer for a source checkout on `PYTHONPATH`.

    A value that disagrees with `__version__` means an installed wheel is
    being shadowed by a checkout, or the reverse. That is the single most
    common reason a fix appears not to take effect.
    """
    try:
        from importlib import metadata

        return metadata.version("mojotrees")
    except Exception:
        return None


def _optional_dependency_versions():
    """Installed versions of the optional dependencies, without importing
    any of them: distribution metadata is enough, and importing scipy or
    scikit-learn to answer a diagnostic question would cost more than the
    answer is worth. None means not installed."""
    try:
        from importlib import metadata
    except Exception:  # pragma: no cover - importlib.metadata is stdlib
        return {}
    found = {}
    for name in _OPTIONAL_DEPENDENCIES:
        try:
            found[name] = metadata.version(name)
        except Exception:
            found[name] = None
    return found


def _install_layout():
    """Install kind and extension paths, borrowed from
    `mojotrees.diagnostics`.

    That module answers this by reading the filesystem rather than by
    importing the extension, which is what makes it usable from a cold
    interpreter. Here the extension is loaded already, so this is reuse
    rather than a second opinion. It degrades to `unknown` instead of
    raising, because this is the function people reach for when something
    is already wrong.
    """
    try:
        from . import diagnostics

        described = diagnostics.describe_install().to_dict()
    except Exception as exc:
        return {"install": "unknown", "install_error": repr(exc)}
    return {
        "install": described["kind"],
        "extension": described["extension"],
        "runtime_dir": described["runtime_dir"],
        "bundled_runtime_libs": described["bundled"],
        "missing_runtime_libs": described["missing_runtime_libs"],
    }


#: Provenance the build writes next to this file, when it writes any.
#: `packaging/macos/provenance.sh` already collects exactly these fields
#: (Mojo and MAX versions, git tag, host OS, SDK, Metal toolchain, and
#: whether an accelerator was visible at compile time) into a
#: `<wheel>.provenance.json` beside the wheel, where an installed package
#: cannot reach it. Copying that file to this name before the wheel is
#: built is what makes it readable here.
_BUILD_INFO_FILE = "_build_info.json"


def _build_provenance():
    """What the build recorded about itself, or None when it recorded
    nothing.

    None is the normal answer today and is not an error: no build script
    writes this file yet. It is read rather than compiled in so that the
    packaging lane can close the gap without editing this module.
    """
    import json

    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         _BUILD_INFO_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            recorded = json.load(handle)
    except Exception:
        return None
    return recorded if isinstance(recorded, dict) else None


def build_info():
    """Everything this installation can say about itself, as a plain dict.

    JSON-serializable, and free of interpretation except for the one
    judgment that cannot be read off any single value: whether the GPU
    path was compiled into this build.

    That judgment matters because availability is decided when the
    extension is compiled and not on the machine that runs it, so one
    wheel carries one answer to every user who installs it.
    `gpu_available()` alone cannot separate "this build has no GPU path in
    it" from "this build has one and `MOJOTREES_DISABLE_GPU=1` is masking
    it", since it returns False for both. `gpu_path_compiled_in` separates
    them, and reports the masked case as `"unknown"` rather than guessing:
    unset the variable and ask again.

    What it deliberately does not answer is whether *this machine* can
    open a device. Nothing here can, because the question is only settled
    when a device is opened. See `mojotrees.device_selection` for the
    policy and docs/DEVICE_SELECTION.md for why a GPU request raises
    rather than falling back.
    """
    import platform
    import sys

    env = {
        name: value
        for name, value in sorted(_os.environ.items())
        if value != ""
        and (name.startswith("MOJOTREES_") or name.startswith("MODULAR_"))
    }

    available = bool(_mojotrees.gpu_available())
    if available:
        gpu_path = "yes"
    elif env.get("MOJOTREES_DISABLE_GPU") == "1":
        gpu_path = "unknown"
    else:
        gpu_path = "no"

    info = {
        "mojotrees": _package().__version__,
        "distribution": _distribution_version(),
        "package": _package().__file__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu_available": available,
        "gpu_path_compiled_in": gpu_path,
        "env": env,
        "optional_dependencies": _optional_dependency_versions(),
        "build": _build_provenance(),
        "model_format_versions": model_format_versions(),
        "startup": startup_contract(),
    }
    info.update(_install_layout())
    return info


def model_format_versions():
    """The two schema versions a consumer of saved models and dumps
    branches on, as the extension states them: `model_format_version` (the
    save format a model written by this build serializes to) and
    `dump_format_version` (the inspection schema's own)."""
    versions = _mojotrees.model_format_versions()
    return {str(k): int(versions[k]) for k in versions}


def startup_contract():
    """What the extension says about its startup: the phase table
    `mojotrees.diagnostics.PHASES` mirrors (`phases`, as `[index, name,
    origin, one_time]` records in report order), what the environment asked
    of startup (`environment`: `trace_enabled`, `warmup_level`), and the
    native monotonic clock in nanoseconds at the moment of the call
    (`clock_ns`), which a harness reads right after `import mojotrees` to
    bound the extension load it cannot time from inside."""
    phases = [
        {
            "index": int(record[0]),
            "name": str(record[1]),
            "origin": str(record[2]),
            "one_time": bool(int(record[3])),
        }
        for record in _mojotrees.startup_phase_contract()
    ]
    environment = _mojotrees.startup_environment()
    return {
        "phases": phases,
        "environment": {str(k): environment[k] for k in environment},
        "clock_ns": int(_mojotrees.native_clock_ns()),
    }


def show_versions(file=None):
    """Print what this installation is, in the form a bug report needs.

        >>> import mojotrees
        >>> mojotrees.show_versions()          # doctest: +SKIP

    Everything printed comes from `build_info()`, which returns the same
    facts as a dict when you want to attach them to something. The
    closing notes appear only when there is something to say, so a clean
    install prints no warnings rather than a row of reassurances.
    """
    import sys

    out = sys.stdout if file is None else file
    info = build_info()

    def line(label, value):
        print(f"  {label:<22} {value}", file=out)

    print(f"mojotrees {info['mojotrees']}", file=out)
    line("package", info["package"])
    line("install", info.get("install", "unknown"))
    if info.get("extension"):
        line("extension", info["extension"])
    if info.get("runtime_dir"):
        line(
            "bundled runtime",
            f"{len(info['bundled_runtime_libs'])} in {info['runtime_dir']}",
        )
    line("gpu path compiled in", info["gpu_path_compiled_in"])
    line("gpu_available()", info["gpu_available"])

    print(
        f"\npython {info['python']} ({info['python_implementation']})",
        file=out,
    )
    line("executable", info["executable"])
    line("platform", info["platform"])
    line("machine", info["machine"])

    print("\noptional dependencies", file=out)
    for name, version in info["optional_dependencies"].items():
        line(name, version if version else "not installed")

    versions = info["model_format_versions"]
    line("model format", versions["model_format_version"])
    line("dump format", versions["dump_format_version"])
    startup = info["startup"]
    line(
        "startup",
        f"{len(startup['phases'])} phases, trace "
        f"{'on' if startup['environment'].get('trace_enabled') else 'off'}, "
        f"warmup {startup['environment'].get('warmup_level')}",
    )

    if info["build"]:
        print("\nbuild", file=out)
        for name, value in sorted(info["build"].items()):
            line(name, value)

    print("\nenvironment", file=out)
    if info["env"]:
        for name, value in info["env"].items():
            line(name, value)
    else:
        line("(none set)", "MOJOTREES_* and MODULAR_* are unset")

    notes = []
    if info["gpu_path_compiled_in"] == "no":
        notes.append(
            "No GPU path was compiled into this build, so device='gpu'\n"
            "raises everywhere it is installed, including on machines with\n"
            "a working accelerator. This is decided on the build machine."
        )
    elif info["gpu_path_compiled_in"] == "unknown":
        notes.append(
            "MOJOTREES_DISABLE_GPU=1 is masking whether this build has a\n"
            "GPU path. Unset it and run this again to find out."
        )
    if info.get("missing_runtime_libs"):
        notes.append(
            "Runtime libraries are missing from the bundle: "
            + ", ".join(info["missing_runtime_libs"])
            + ".\nThis package will fail to import in a fresh interpreter."
        )
    distribution = info["distribution"]
    if distribution is not None and distribution != info["mojotrees"]:
        notes.append(
            f"The installed distribution says {distribution} and the "
            f"package says {info['mojotrees']}.\nOne is shadowing the "
            "other, most often a source checkout on PYTHONPATH in front\n"
            "of an installed wheel."
        )
    if info["build"] is None and info.get("install") == "source":
        notes.append(
            "This is a source build. Add the output of "
            "`pixi run mojo --version`\nto a bug report; the extension "
            "does not carry it."
        )
    elif info["build"] is None:
        notes.append(
            "This artifact recorded no build provenance, so it cannot say\n"
            "which Mojo compiled it or what the build machine was. That is\n"
            "a gap in the packaging rather than in your install."
        )
    for note in notes:
        print(f"\n{note}", file=out)
