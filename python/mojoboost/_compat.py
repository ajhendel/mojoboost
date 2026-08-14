"""The interpreter facts the wrapper checks before it loads the extension.

Internal, on the terms of section 2 of `docs/COMPATIBILITY_POLICY.md`, which
puts every underscore-prefixed module in `python/mojoboost/` outside the
public surface in its entirety.

This module holds two checks and nothing else. It is deliberately not a
compatibility layer: the audit in `docs/PYTHON_SUPPORT.md` found no
version-gated syntax and no version-gated standard library call anywhere in
`python/mojoboost/`, so there is nothing to shim and a shim written now would
be abstraction ahead of a need. `tools/audit_python_compat.py` re-checks that
finding, and this module gets a new function on the day that script reports a
construct that needs one.

What is here is the two ways an interpreter can be wrong in a way the user
cannot diagnose from the failure they would otherwise see.

**Below the floor.** A wheel carries an interpreter tag and pip refuses a
mismatch before `requires-python` is ever consulted, so an installed wheel
cannot land on the wrong interpreter. A source checkout can: the supported
way to use mojoboost today is `pixi install` plus `bindings/build.sh`
(sections 10.2 and 10.3 of the compatibility policy), and nothing in that
path checks the interpreter. What the user sees there is whatever the Mojo
runtime says about a missing CPython entry point, which names a C symbol and
not a version.

**Free-threaded.** Every `max 26.5.0` variant depends on `python-gil`, so no
extension for a free-threaded interpreter exists or can be built. CPython
does not refuse to load an extension that declares no free-threaded support;
it re-enables the GIL and emits a `RuntimeWarning`. That is a worse outcome
than a clear refusal, because the program keeps running with the thing the
user chose the interpreter for silently switched off.

`PYTHON_FLOOR` below and `requires-python` in `python/pyproject.toml` are the
same fact written twice and they move in the same commit. Neither is the
authority. `docs/PYTHON_SUPPORT.md` is, and it records what evidence would be
needed to move them.
"""

import sys
import sysconfig

#: The lowest CPython this release supports, as `(major, minor)`. Must equal
#: the lower bound of `requires-python` in `python/pyproject.toml`.
PYTHON_FLOOR = (3, 14)

#: Why the free-threaded build is excluded, quoted in the error.
_NO_GIL_REASON = (
    "the pinned MAX toolchain depends on python-gil, so no mojoboost "
    "extension is built for a free-threaded interpreter"
)


def gil_enabled():
    """True when this interpreter has a GIL.

    `Py_GIL_DISABLED` is the configure-time fact and it is what the build of
    the interpreter is, which is the question here. `sys._is_gil_enabled()`
    answers a different and narrower question, whether the GIL happens to be
    on right now, and an extension without free-threaded support turning it
    back on is exactly the case this function exists to get ahead of.
    """
    return not bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def unsupported_interpreter():
    """The reason this interpreter cannot run mojoboost, or None.

    One string, phrased to be read at the end of an ImportError, naming the
    interpreter it was given rather than the one it wanted.
    """
    if not gil_enabled():
        return (
            "mojoboost does not support the free-threaded build of CPython "
            "%d.%d: %s"
            % (sys.version_info[0], sys.version_info[1], _NO_GIL_REASON)
        )
    if sys.version_info[:2] < PYTHON_FLOOR:
        return (
            "mojoboost requires CPython %d.%d or newer and this is %d.%d"
            % (
                PYTHON_FLOOR[0],
                PYTHON_FLOOR[1],
                sys.version_info[0],
                sys.version_info[1],
            )
        )
    return None


def import_extension():
    """Import and return `mojoboost._mojoboost`.

    The extension's own failure is never swallowed. When the interpreter is
    one mojoboost does not support, that fact is added to the message,
    because a dynamic loader error naming a CPython C symbol does not tell
    the reader which interpreter to use instead.
    """
    try:
        from . import _mojoboost
    except ImportError as exc:
        reason = unsupported_interpreter()
        if reason is None:
            raise
        raise ImportError("%s (%s)" % (exc, reason))
    return _mojoboost
