"""The interpreter checks that run before the extension is loaded.

Internal, on the terms of section 2 of `docs/COMPATIBILITY_POLICY.md`, which
puts every underscore-prefixed module in `python/mojoboost/` outside the
public surface in its entirety.

This module holds two checks and nothing else. It is deliberately not a
compatibility layer: the audit in `docs/PYTHON_SUPPORT.md` found no
version-gated syntax and no version-gated standard library call in
`python/mojoboost/` that any supported interpreter lacks, so there is nothing
to shim and a shim written now would be abstraction ahead of a need.
`tools/audit_python_compat.py` re-checks that on demand, and this module gets
a new function on the day that script reports a construct that needs one.

**Why the checks run before the import rather than around it.** On an
interpreter older than `EXTENSION_FLOOR` the extension does not raise. The
Mojo runtime resolves CPython entry points by name out of libpython at load
time, and a name it cannot find ends the process:

    ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:762:18:
    symbol not found: Py_NewRef

That is measured, on CPython 3.9, and it is recorded in section 10 of
`docs/PYTHON_SUPPORT.md`. An abort cannot be caught, so a `try` around the
import would never run its handler, and pending stdout is lost with it. The
only place a check can do any good is in front.

**Where it is called from.** `python/mojoboost/__init__.py` calls
`import_extension()` and binds the result, ahead of every other import in
the package. That is the only call site it needs and the only one it should
have: importing any submodule of `mojoboost` runs the package's `__init__`
first, so a check there runs before any other module can name the extension.
The modules that later say `from . import _mojoboost` are reading a module
that is already in `sys.modules` by then, which is why they do not repeat
the check and must not start.

`EXTENSION_FLOOR` is a measured property of the compiled artifact and is not
the same fact as `requires-python` in `python/pyproject.toml`, which is what
the project chooses to declare and may be higher. `docs/PYTHON_SUPPORT.md` is
the authority on both and on the evidence behind them.
"""

import sys
import sysconfig

#: The lowest CPython the compiled extension runs on, as `(major, minor)`.
#: Measured, not inferred: 3.10 through 3.14 import and pass the Python API
#: suite, and 3.9 aborts on `Py_NewRef`, which CPython added in 3.10. Three
#: other entry points in the extension's table are also 3.10 additions, and
#: `mojo 1.0.0` independently requires `python >=3.10`, so nothing below this
#: is reachable by any route. See docs/PYTHON_SUPPORT.md sections 9 and 10.
EXTENSION_FLOOR = (3, 10)

#: Why the free-threaded build is excluded, quoted in the error. Every
#: `max 26.5.0` variant depends on `python-gil`, so the environment cannot be
#: solved for a free-threaded interpreter and no extension is built for one.
#: Unlike EXTENSION_FLOOR this has not been measured, because there is no
#: artifact to measure.
_NO_GIL_REASON = (
    "the pinned MAX toolchain depends on python-gil, so no mojoboost "
    "extension is built for a free-threaded interpreter"
)


def gil_enabled():
    """True when this interpreter was built with a GIL.

    `Py_GIL_DISABLED` is the configure-time fact, which is the question here.
    `sys._is_gil_enabled()` answers a narrower one, whether the GIL happens
    to be on at this moment, and an extension without free-threaded support
    causing it to be switched back on is part of what this gets ahead of.
    """
    return not bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def unsupported_interpreter():
    """The reason this interpreter cannot load the extension, or None.

    One string, phrased to be read at the end of an ImportError, naming the
    interpreter it was given rather than the one it wanted.
    """
    if not gil_enabled():
        return (
            "mojoboost does not support the free-threaded build of CPython "
            "%d.%d: %s"
            % (sys.version_info[0], sys.version_info[1], _NO_GIL_REASON)
        )
    if sys.version_info[:2] < EXTENSION_FLOOR:
        return (
            "mojoboost needs CPython %d.%d or newer and this is %d.%d. The "
            "compiled extension resolves CPython entry points that %d.%d "
            "does not have"
            % (
                EXTENSION_FLOOR[0],
                EXTENSION_FLOOR[1],
                sys.version_info[0],
                sys.version_info[1],
                sys.version_info[0],
                sys.version_info[1],
            )
        )
    return None


def import_extension():
    """Import and return `mojoboost._mojoboost`.

    Raises `ImportError` before touching the extension when the interpreter
    is one it cannot load, because the alternative is not an exception but a
    process abort naming a CPython C symbol. On a supported interpreter the
    extension's own failure, if any, propagates untouched: a missing build
    and a bad interpreter are different problems and should not report the
    same way.
    """
    reason = unsupported_interpreter()
    if reason is not None:
        raise ImportError(reason)
    from . import _mojoboost

    return _mojoboost
