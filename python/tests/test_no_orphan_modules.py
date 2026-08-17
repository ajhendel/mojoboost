"""The gate that makes "built and never connected" loud instead of quiet.

Written 2026-08-16, when the round's test budget was closed before it could
be executed, and first run by pytest on 2026-08-17. The `ORPHANS` list below
was computed by the same walk this file performs, run once as an analysis
script over the tree at commit `907b2a1`. Its first real run reported eight
of those thirteen rows as stale, which is the direction this file calls good
news; the rows are gone and what remains is five.

WHY THIS EXISTS. Over one round this repository merged, tested and
documented six mechanisms that no shipped fit could reach: CatBoost mode,
oblivious trees, `float64` derivatives on the sparse and distributed paths,
`float64` on the GPU, the CatBoost parameter group, and the ranking,
survival and multi-target objectives. Every one of them had a passing native
suite. What none of them had was anything that noticed the module was
imported by nothing.

That is a graph question and it is cheap to answer. This file reads the
`from .x import` / `from mojotrees.x import` edges out of `src/mojotrees`,
seeds the walk from the three things that actually ship -- the package
`__init__.mojo`, `bindings/`, `capi/` and `cli/` -- and reports every module
no shipped entry point can transitively reach. It compiles nothing, imports
no extension, and trains no model, so it costs a directory walk.

HOW TO USE IT WHEN IT FAILS.

*A new name appeared.* You merged a module that nothing imports. That is the
defect. Wire it, or add it here with a one-line reason so the debt is
written down rather than discovered a round later.

*A name disappeared.* Someone connected it. Delete the row. Do not delete
the row for any other reason: the list shrinking is the only good news this
file can report.

A module in this list is not necessarily wrong -- `ctr.mojo` was verified
against CatBoost source line by line and is almost certainly correct. It is
*unreachable*, which is a different and much cheaper thing to check.
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PKG = _ROOT / "src" / "mojotrees"
_ENTRY_DIRS = ("bindings", "capi", "cli")

_IMPORT_PATTERNS = (
    re.compile(r"from\s+\.(\w+)\s+import"),
    re.compile(r"from\s+mojotrees\.(\w+)\s+import"),
    re.compile(r"import\s+mojotrees\.(\w+)"),
)

#: Modules that ship in `src/mojotrees` and that no shipped entry point can
#: reach, with why. First measured at commit 907b2a1: 13 of 107 modules, 12%
#: of the package. Eight of those thirteen were wired on 2026-08-16 and
#: their rows are gone, which leaves 5 of 109. Each row is a debt, not a
#: permission.
ORPHANS = {
    # The CatBoost objective families that are still only half wired. Both
    # carry complete trainers -- `train_catboost_ranker`, `train_cox`,
    # `train_survival_aft` -- and their objective codes are reserved in
    # objective_registry.mojo, which is the half that had to be right first
    # because a code is a number in a serialized model. What is missing is a
    # binding entry point that bins the matrix and calls them. The third
    # family, multi_target, got exactly that entry point in
    # bindings/catboost_reach_bindings.mojo and left this list.
    "catboost_ranking": "no binding calls train_catboost_ranker",
    "survival": "no binding calls train_cox or train_survival_aft",
    # The rest, each found by this walk rather than by a lane reporting it.
    "langevin": "no trainer takes its noise parameters",
    "backend": "superseded by device_policy and the gpu_* policy modules",
    "gpu_vendor_policy": "no GPU entry point consults it",
}


def _imports(text):
    out = set()
    for pattern in _IMPORT_PATTERNS:
        out.update(pattern.findall(text))
    return out


def _unreachable():
    """Every module in `src/mojotrees` no shipped entry point can reach.

    The walk is transitive on purpose. When `ctr.mojo` was still an orphan a
    direct-importer check would have called it reachable, because
    `ctr_combinations.mojo` imported it and `ctr_combinations.mojo` was
    imported by nothing: the whole subtree was dead and a direct check would
    have reported half of it. That subtree is wired now; the argument is not
    about those two files.
    """
    modules = {
        p.stem: _imports(p.read_text())
        for p in _PKG.glob("*.mojo")
        if p.stem != "__init__"
    }
    seeds = _imports((_PKG / "__init__.mojo").read_text())
    for name in _ENTRY_DIRS:
        directory = _ROOT / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.mojo"):
            seeds |= _imports(path.read_text())
    seen = set()
    stack = [m for m in seeds if m in modules]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(
            dep
            for dep in modules[current]
            if dep in modules and dep not in seen
        )
    return {m for m in modules if m not in seen}


def test_no_module_becomes_unreachable_without_being_written_down():
    """The gate. A module that nothing can reach is either wired or listed.

    The failure message is the whole point, so it says which direction the
    list moved and what to do about each.
    """
    found = _unreachable()
    listed = set(ORPHANS)
    new = sorted(found - listed)
    fixed = sorted(listed - found)
    assert not new, (
        "these modules ship in src/mojotrees and no entry point can reach "
        "them: " + ", ".join(new) + ". This is the sixth-in-a-row defect "
        "this file exists to catch. Wire the module, or add it to ORPHANS "
        "with the one line saying what is missing."
    )
    assert not fixed, (
        "these modules are now reachable and their ORPHANS rows are stale: "
        + ", ".join(fixed)
        + ". Delete the rows. The list shrinking is the only good news this "
        "file reports, so it should not be reported as a failure twice."
    )


def test_the_orphan_list_is_a_twelfth_of_the_package():
    """A standing number, not a threshold to tune.

    13 of 107 modules at commit 907b2a1, 5 of 109 once the reachability
    lanes landed. The ceiling stays at the number it was registered with,
    because it is a ceiling and not a record of the current share. It
    asserts the share does not grow, which is a weaker claim than the test
    above and a useful one: it fails on a pattern of small additions that
    each individually looked acceptable, which is how the list got to 13.
    """
    total = len([p for p in _PKG.glob("*.mojo") if p.stem != "__init__"])
    assert len(_unreachable()) / total <= 13 / 107


@pytest.mark.parametrize("name", sorted(ORPHANS))
def test_every_listed_orphan_still_exists(name):
    """A row that names a deleted file is a row nobody has read in a while."""
    assert (_PKG / f"{name}.mojo").is_file()


def test_orphan_list_agrees_with_the_connectivity_audit():
    """`ORPHANS` and `tools/connectivity_audit.py` must name the same modules.

    Both landed within an hour of each other, independently, against the same
    problem, and they agreed on all thirteen -- which is reassuring exactly
    once. After that they are two hand-maintained copies of one judgment, and
    `docs/INTEGRATION_INVENTORY.md` has a standing section about what happens
    to those: one gets updated, the other does not, and the question quietly
    has two answers.

    This does not merge them, because they are not the same artifact. The
    `CLASSIFICATION` table carries a severity and an owner for the tooling and
    the CI gate; `ORPHANS` carries the sentence a reader needs. What this
    forbids is the two disagreeing about the *facts*, which is the part
    neither of them gets to have an opinion about. Whichever one is stale, the
    failure names it.

    It shells out rather than importing, because `tools/connectivity_audit.py`
    is a script and `--json` is the interface it documents.
    """
    proc = subprocess.run(
        [sys.executable, "tools/connectivity_audit.py", "--json"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    audited = {
        f["subject"] for f in json.loads(proc.stdout).get("orphans", [])
    }
    assert audited == set(ORPHANS), (
        "ORPHANS and tools/connectivity_audit.py disagree. Only in ORPHANS: "
        + (", ".join(sorted(set(ORPHANS) - audited)) or "none")
        + ". Only in the audit: "
        + (", ".join(sorted(audited - set(ORPHANS))) or "none")
        + ". Update whichever is stale; docs/INTEGRATION_INVENTORY.md is "
        "rendered from the audit and moves with it."
    )
