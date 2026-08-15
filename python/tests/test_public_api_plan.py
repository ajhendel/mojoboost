"""`_public_api_plan` against the package it describes.

`python/mojotrees/_public_api_plan.py` is the record of what the package
exports, which submodules resolve lazily, and which public-looking names
are deliberately absent. Its own docstring says a test can compare it
against the real `__all__`; until this file existed, none did, so every
row in it was a claim that drifted the moment `__init__.py` changed.

What is checked here is only the part of that module that is a fact about
the code: the export list, the lazy tables, the collision decision, and
the "not exported" names. The `why` prose is not checkable and is not
touched.

Laziness itself is checked in a subprocess. `__getattr__` caches what it
resolves into the package globals, and other files in this suite import
every submodule at collection time, so `vars(mojotrees)` in this process
says nothing about what a plain `import mojotrees` reaches.
"""

import importlib
import os
import subprocess
import sys

import pytest

import mojotrees
from mojotrees import _public_api_plan as plan


def test_current_top_level_matches_all():
    """The recorded surface is the real one."""
    assert set(plan.CURRENT_TOP_LEVEL) == set(mojotrees.__all__)


def test_current_top_level_is_sorted():
    """The record is sorted even though `__all__` is grouped by topic,
    so a diff of this tuple shows the name that changed."""
    assert list(plan.CURRENT_TOP_LEVEL) == sorted(plan.CURRENT_TOP_LEVEL)


def test_additions_are_exported():
    """Every name the plan says was added is in `__all__` and resolves."""
    for entry in plan.TOP_LEVEL_ADDITIONS:
        name = entry["name"]
        assert name in mojotrees.__all__, name
        assert getattr(mojotrees, name) is not None, name


def test_lazy_submodules_match_the_package():
    """The plan's lazy table is the one `__init__.py` runs."""
    recorded = tuple(entry["name"] for entry in plan.LAZY_SUBMODULES)
    assert recorded == tuple(mojotrees._LAZY_SUBMODULES)


def test_lazy_snippet_matches_the_running_code():
    """The snippet is a copy of live code, so it is checked as one.

    Only the two tables are compared. The snippet's `__getattr__` body
    cannot be executed against the real package without shadowing it.
    """
    namespace = {}
    exec(plan.LAZY_SUBMODULE_SNIPPET, namespace)  # noqa: S102
    assert tuple(namespace["_LAZY_SUBMODULES"]) == tuple(
        mojotrees._LAZY_SUBMODULES
    )
    assert namespace["_LAZY_ATTRS"] == mojotrees._LAZY_ATTRS


@pytest.mark.parametrize("name", sorted(mojotrees._LAZY_SUBMODULES))
def test_every_lazy_submodule_resolves(name):
    """`mojotrees.<name>` imports, and is the submodule of that name."""
    module = getattr(mojotrees, name)
    assert module.__name__ == "mojotrees." + name


@pytest.mark.parametrize(
    "entry", plan.NOT_EXPORTED, ids=lambda e: e["name"]
)
def test_not_exported_names_exist_and_stay_out(entry):
    """A name listed as deliberately absent is absent, and is real.

    Both halves matter: a name that left the submodule makes the row a
    stale answer to "why is it missing", and a name that reached
    `__all__` makes the row wrong about the decision.
    """
    module_path, _, attribute = entry["name"].rpartition(".")
    module = importlib.import_module(module_path)
    assert hasattr(module, attribute), entry["name"]
    assert attribute not in mojotrees.__all__, entry["name"]


def test_cv_collision_resolves_to_the_function():
    """The decision in `NAME_COLLISIONS`, as the attribute answers it."""
    collision = plan.NAME_COLLISIONS[0]
    assert collision["attribute"] == "mojotrees.cv"
    assert collision["decision"] == "the function"
    assert callable(mojotrees.cv)
    assert not hasattr(mojotrees.cv, "__path__")


def _probe(source):
    """Run `source` in a fresh interpreter that sees this working copy."""
    env = dict(os.environ)
    python_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(mojotrees.__file__))
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [python_dir] + ([env["PYTHONPATH"]] if "PYTHONPATH" in env else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def test_import_mojotrees_reaches_no_lazy_submodule():
    """The rule the lazy table exists for: `import mojotrees` must not
    import dask, and must not pay for the others either."""
    reached = _probe(
        "import sys, mojotrees\n"
        "print(' '.join(sorted(\n"
        "    n for n in sys.modules if n.startswith('mojotrees.')\n"
        ")))\n"
    )
    for name in mojotrees._LAZY_SUBMODULES:
        assert "mojotrees." + name not in reached
    assert "dask" not in sys.modules or "dask" not in reached


def test_eager_and_lazy_additions_are_as_recorded():
    """`eager` in the plan is what a fresh import actually did."""
    present = _probe(
        "import mojotrees\n"
        "print(' '.join(sorted(vars(mojotrees))))\n"
    )
    for entry in plan.TOP_LEVEL_ADDITIONS:
        name = entry["name"]
        if entry["eager"]:
            assert name in present, name
        else:
            assert name not in present, name
