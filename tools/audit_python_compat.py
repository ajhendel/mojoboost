#!/usr/bin/env python3
"""Audit which CPython versions mojoboost could be built for and run on.

    python3 tools/audit_python_compat.py

Standard library only. It imports nothing from `mojoboost`, never loads the
compiled extension, and builds nothing, so it runs on a bare checkout and,
more to the point, it runs under an interpreter that mojoboost does not yet
support. That is the whole reason it exists: the question this script serves
is "could this package run on 3.12", and a script that has to import the
package to answer it cannot be run on 3.12.

Written to parse under CPython 3.8 and later so that every candidate
interpreter in `docs/PYTHON_SUPPORT.md` can run it. Do not use syntax newer
than 3.8 here, and do not import `tomllib`.

Five independent sections, each printing its own verdict:

    source       AST feature scan of python/mojoboost, which gives a lower
                 bound on the interpreter the pure-Python half needs
    metadata     requires-python and the Python classifiers in
                 python/pyproject.toml, checked against that lower bound
    toolchain    what pixi.lock records about the interpreter the pinned
                 Mojo and MAX packages were solved against
    extension    a byte scan of a built _mojoboost.so for the CPython entry
                 point names the Mojo runtime resolves, and for the ones it
                 says it resolves conditionally
    interpreter  facts about the interpreter running this script

Exit status is 0 when every section passes and 1 otherwise. A section that
has nothing to inspect (no built extension, for instance) reports that and
does not fail, because the absence of a build artifact is not a defect in
the source tree.

What this script cannot do, stated so nobody reads more into its output than
is there:

  - The source scan is a denylist of known version-gated constructs. It
    proves a floor is at least N. It never proves a floor is exactly N.
  - The extension scan reads bytes. It finds the names of CPython entry
    points in the binary; it cannot tell a name that is linked against from
    one that is resolved at runtime, and it cannot tell an eagerly resolved
    name from a lazily resolved one. It prints the shell commands that can.
  - Nothing here builds, installs, or imports anything. A green run is not
    evidence that any interpreter works. `docs/PYTHON_SUPPORT.md` says what
    evidence would be.
"""

from __future__ import annotations

import ast
import re
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "python" / "mojoboost"
PYPROJECT = ROOT / "python" / "pyproject.toml"
LOCK = ROOT / "pixi.lock"
EXTENSION = PKG / "_mojoboost.so"

# ---------------------------------------------------------------------------
# Version-gated constructs
#
# Every entry is a construct whose first CPython release is a documented fact,
# not a guess. Adding a row is cheap; guessing at one makes the whole report
# untrustworthy, so a construct nobody is sure about stays out.
# ---------------------------------------------------------------------------

# ast node class name -> (version, description). Looked up with getattr so
# this file still parses and runs on an interpreter whose ast module has no
# such node.
NODE_FLOORS = [
    ("NamedExpr", (3, 8), "walrus operator, PEP 572"),
    ("Match", (3, 10), "match statement, PEP 634"),
    ("TryStar", (3, 11), "except*, PEP 654"),
    ("TypeAlias", (3, 12), "type alias statement, PEP 695"),
]

# Whole modules that did not exist before a given release.
MODULE_FLOORS = {
    "dataclasses": (3, 7),
    "contextvars": (3, 7),
    "importlib.metadata": (3, 8),
    "graphlib": (3, 9),
    "zoneinfo": (3, 9),
    "tomllib": (3, 11),
    "annotationlib": (3, 14),
    "compression": (3, 14),
    "concurrent.interpreters": (3, 14),
}

# (module, attribute) pairs added after the module itself.
ATTR_FLOORS = {
    ("math", "lcm"): (3, 9),
    ("math", "nextafter"): (3, 9),
    ("math", "ulp"): (3, 9),
    ("functools", "cache"): (3, 9),
    ("itertools", "pairwise"): (3, 10),
    ("itertools", "batched"): (3, 12),
    ("typing", "Self"): (3, 11),
    ("typing", "assert_type"): (3, 11),
    ("typing", "override"): (3, 12),
    ("warnings", "deprecated"): (3, 13),
    ("sys", "_is_gil_enabled"): (3, 13),
}

# Bare method names, which cannot be attributed to a receiver type without
# inference. Both of these are unambiguous enough to be worth the row.
METHOD_FLOORS = {
    "removeprefix": (3, 9),
    "removesuffix": (3, 9),
}

# Builtins that only became subscriptable at runtime in 3.9, PEP 585. Under
# `from __future__ import annotations` a subscript inside an annotation is
# never evaluated, so the floor does not apply there and the scan says so.
BUILTIN_GENERICS = {
    "list", "dict", "tuple", "set", "frozenset", "type",
}

# ---------------------------------------------------------------------------
# CPython entry points, by the release that first shipped them
#
# Only names first exported in 3.10 or later are listed. Anything older is not
# a discriminator for any interpreter this project would consider, so its
# absence from this table is not a claim that it has no floor.
# ---------------------------------------------------------------------------
SYMBOL_FLOORS = {
    "Py_Is": (3, 10),
    "Py_IsNone": (3, 10),
    "Py_IsTrue": (3, 10),
    "Py_IsFalse": (3, 10),
    "Py_NewRef": (3, 10),
    "Py_XNewRef": (3, 10),
    "PyModule_AddObjectRef": (3, 10),
    "PyIter_Send": (3, 10),
    "PyType_GetName": (3, 11),
    "PyType_GetQualName": (3, 11),
    "PyFrame_GetLasti": (3, 11),
    "PyErr_GetRaisedException": (3, 12),
    "PyErr_SetRaisedException": (3, 12),
    "PyErr_DisplayException": (3, 12),
    "PyType_GetDict": (3, 12),
    "Py_GetConstant": (3, 13),
    "Py_GetConstantBorrowed": (3, 13),
    "PyLong_AsInt": (3, 13),
    "PyType_GetFullyQualifiedName": (3, 13),
    "PyType_GetModuleName": (3, 13),
}

# The Mojo runtime carries a diagnostic string for every entry point it looks
# up conditionally. Finding the string in the binary is what tells this script
# that a symbol's absence is handled rather than fatal, which is the whole
# difference between a hard floor and a soft one. The pattern is read out of
# the binary rather than hardcoded, so a toolchain that guards one more symbol
# moves this report without anybody editing it.
GUARD_STRING = re.compile(
    rb"([A-Za-z_][A-Za-z0-9_]*) is not available in this Python version"
)
SYMBOL_TOKEN = re.compile(rb"_?Py[A-Za-z0-9_]{2,62}")

MIN_TOOL_PYTHON = (3, 8)


def vstr(v):
    return "%d.%d" % v


class Report(object):
    """Accumulates the verdicts. Sections print as they go, because a release
    decision wants the whole list rather than the first failure."""

    def __init__(self):
        self.failures = []
        self.notes = []

    def ok(self, section, message):
        print("  ok    %s: %s" % (section, message))

    def note(self, section, message):
        self.notes.append((section, message))
        print("  note  %s: %s" % (section, message))

    def fail(self, section, message):
        self.failures.append((section, message))
        print("  FAIL  %s: %s" % (section, message))

    def info(self, message):
        print("        %s" % message)


# ---------------------------------------------------------------------------
# Section 1: the source
# ---------------------------------------------------------------------------


class FeatureScan(ast.NodeVisitor):
    """Collects the version-gated constructs in one module.

    Annotations are tracked separately because `from __future__ import
    annotations` makes an annotation a string that is never evaluated, so a
    PEP 585 subscript or a PEP 604 union inside one imposes no runtime floor.
    Without the future import it imposes 3.9 or 3.10 respectively.
    """

    def __init__(self, path, lazy_annotations):
        self.path = path
        self.lazy = lazy_annotations
        self.found = []  # (version, description, lineno)
        self._in_annotation = 0

    def record(self, version, what, node):
        self.found.append((version, what, getattr(node, "lineno", 0)))

    # -- generic node kinds --------------------------------------------------

    def generic_visit(self, node):
        for name, version, what in NODE_FLOORS:
            cls = getattr(ast, name, None)
            if cls is not None and isinstance(node, cls):
                self.record(version, what, node)
        ast.NodeVisitor.generic_visit(self, node)

    # -- imports -------------------------------------------------------------

    def visit_Import(self, node):
        for alias in node.names:
            self._module(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            self._module(node.module, node)
            for alias in node.names:
                self._module(node.module + "." + alias.name, node)
                floor = ATTR_FLOORS.get((node.module, alias.name))
                if floor:
                    self.record(
                        floor, "%s.%s" % (node.module, alias.name), node
                    )
        self.generic_visit(node)

    def _module(self, dotted, node):
        floor = MODULE_FLOORS.get(dotted)
        if floor:
            self.record(floor, "module %s" % dotted, node)

    # -- attributes and calls ------------------------------------------------

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            floor = ATTR_FLOORS.get((node.value.id, node.attr))
            if floor:
                self.record(
                    floor, "%s.%s" % (node.value.id, node.attr), node
                )
        floor = METHOD_FLOORS.get(node.attr)
        if floor:
            self.record(floor, ".%s()" % node.attr, node)
        self.generic_visit(node)

    def visit_Call(self, node):
        # zip(..., strict=...) is 3.10. The only keyword worth a special case,
        # because it is the one that fails silently on an older interpreter by
        # being swallowed as an unexpected argument error rather than a
        # SyntaxError.
        if isinstance(node.func, ast.Name) and node.func.id == "zip":
            for kw in node.keywords:
                if kw.arg == "strict":
                    self.record((3, 10), "zip(strict=...)", node)
        self.generic_visit(node)

    # -- functions, classes, annotations -------------------------------------

    def visit_FunctionDef(self, node):
        self._function(node)

    def visit_AsyncFunctionDef(self, node):
        self._function(node)

    def _function(self, node):
        if getattr(node, "type_params", None):
            self.record((3, 12), "generic function, PEP 695", node)
        if getattr(node.args, "posonlyargs", None):
            self.record((3, 8), "positional-only parameters, PEP 570", node)
        for arg in self._all_args(node.args):
            if arg.annotation is not None:
                self._annotation(arg.annotation)
        if node.returns is not None:
            self._annotation(node.returns)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if getattr(node, "type_params", None):
            self.record((3, 12), "generic class, PEP 695", node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._annotation(node.annotation)
        self.generic_visit(node)

    @staticmethod
    def _all_args(args):
        out = list(getattr(args, "posonlyargs", None) or [])
        out += list(args.args) + list(args.kwonlyargs)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                out.append(extra)
        return out

    def _annotation(self, node):
        """Version floors that only apply when annotations are evaluated."""
        if self.lazy:
            return
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                self.record((3, 10), "X | Y annotation, PEP 604", sub)
            if isinstance(sub, ast.Subscript) and isinstance(
                sub.value, ast.Name
            ):
                if sub.value.id in BUILTIN_GENERICS:
                    self.record(
                        (3, 9),
                        "%s[...] annotation, PEP 585" % sub.value.id,
                        sub,
                    )


def _has_future_annotations(tree):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    return True
    return False


def scan_source(res):
    print("\nsource: version-gated constructs in python/mojoboost")
    if not PKG.is_dir():
        res.fail("source", "%s does not exist" % PKG.relative_to(ROOT))
        return None

    paths = sorted(p for p in PKG.glob("*.py"))
    if not paths:
        res.fail("source", "no modules found in %s" % PKG.relative_to(ROOT))
        return None

    floor = (3, 0)
    unparsed = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            unparsed += 1
            res.fail(
                "source",
                "%s does not parse under this interpreter (%s): line %s, %s"
                % (
                    path.name,
                    vstr(sys.version_info[:2]),
                    exc.lineno,
                    exc.msg,
                ),
            )
            continue
        scan = FeatureScan(path, _has_future_annotations(tree))
        scan.visit(tree)
        if scan.found:
            highest = max(scan.found)[0]
            floor = max(floor, highest)
            for version, what, lineno in sorted(set(scan.found), reverse=True):
                res.info(
                    "%s:%d needs %s (%s)"
                    % (path.name, lineno, vstr(version), what)
                )

    if unparsed:
        return None
    res.ok(
        "source",
        "%d modules scanned, lower bound %s" % (len(paths), vstr(floor)),
    )
    res.info(
        "a lower bound only: this is a denylist of known constructs, not a "
        "proof that %s is enough" % vstr(floor)
    )
    return floor


# ---------------------------------------------------------------------------
# Section 2: the metadata
# ---------------------------------------------------------------------------

REQUIRES = re.compile(r'^\s*requires-python\s*=\s*"([^"]+)"', re.M)
CLASSIFIER = re.compile(r'"Programming Language :: Python :: (\d+)\.(\d+)"')
LOWER_BOUND = re.compile(r">=\s*(\d+)\.(\d+)")
UPPER_BOUND = re.compile(r"<\s*(\d+)\.(\d+)")


def scan_metadata(res, source_floor):
    print("\nmetadata: python/pyproject.toml")
    if not PYPROJECT.is_file():
        res.fail("metadata", "%s does not exist" % PYPROJECT.relative_to(ROOT))
        return None
    text = PYPROJECT.read_text(encoding="utf-8")

    match = REQUIRES.search(text)
    if not match:
        res.fail("metadata", "no requires-python declared")
        return None
    spec = match.group(1)
    bound = LOWER_BOUND.search(spec)
    if not bound:
        res.fail("metadata", "requires-python %r has no lower bound" % spec)
        return None
    declared = (int(bound.group(1)), int(bound.group(2)))
    cap = UPPER_BOUND.search(spec)
    res.ok("metadata", "requires-python = %r, floor %s" % (spec, vstr(declared)))
    if cap:
        res.note(
            "metadata",
            "requires-python is capped at %s. A cap has to be edited on every "
            "release the toolchain permits" % vstr((int(cap.group(1)), int(cap.group(2)))),
        )

    claimed = sorted(
        set(
            (int(a), int(b))
            for a, b in CLASSIFIER.findall(text)
            if int(b) > 0
        )
    )
    if claimed:
        res.ok(
            "metadata",
            "classifiers claim %s" % ", ".join(vstr(v) for v in claimed),
        )
        for version in claimed:
            if version < declared:
                res.fail(
                    "metadata",
                    "classifier claims %s but requires-python excludes it"
                    % vstr(version),
                )
    else:
        res.note("metadata", "no per-minor-version Python classifier")

    if source_floor is not None and source_floor > declared:
        res.fail(
            "metadata",
            "the source needs at least %s but requires-python says %s"
            % (vstr(source_floor), vstr(declared)),
        )
    elif source_floor is not None and source_floor < declared:
        res.note(
            "metadata",
            "requires-python is %s while the pure-Python source needs only "
            "%s. The gap is the extension and the toolchain, not the "
            "language. See docs/PYTHON_SUPPORT.md"
            % (vstr(declared), vstr(source_floor)),
        )
    return declared


# ---------------------------------------------------------------------------
# Section 3: the toolchain
# ---------------------------------------------------------------------------

CONDA_LINE = re.compile(
    r"^- conda: .*/([a-z0-9_.-]+)-([0-9][^-]*)-([^/]+)\.conda$"
)
PY_EXACT = re.compile(r"^\s*- python (\d+)\.(\d+)\.\*\s*$")
PY_FLOOR = re.compile(r"^\s*- python >=\s*(\d+)\.(\d+)")
PY_GIL = re.compile(r"^\s*- python-gil\s*$")


def scan_toolchain(res, declared):
    print("\ntoolchain: pixi.lock")
    if not LOCK.is_file():
        res.fail("toolchain", "%s does not exist" % LOCK.name)
        return
    packages = {}
    name = None
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        head = CONDA_LINE.match(line)
        if head:
            name = head.group(1)
            packages.setdefault(
                name,
                {
                    "version": head.group(2),
                    "build": head.group(3),
                    "python": set(),
                    "floor": set(),
                    "gil": False,
                },
            )
            continue
        if name is None:
            continue
        exact = PY_EXACT.match(line)
        if exact:
            packages[name]["python"].add(
                (int(exact.group(1)), int(exact.group(2)))
            )
        floor = PY_FLOOR.match(line)
        if floor:
            packages[name]["floor"].add(
                (int(floor.group(1)), int(floor.group(2)))
            )
        if PY_GIL.match(line):
            packages[name]["gil"] = True

    for key in ("mojo", "mojo-compiler", "mojo-python", "max", "max-core"):
        info = packages.get(key)
        if not info:
            res.note("toolchain", "%s not found in the lock" % key)
            continue
        bits = ["%s %s (%s)" % (key, info["version"], info["build"])]
        if info["python"]:
            bits.append(
                "pins python "
                + ", ".join(vstr(v) + ".*" for v in sorted(info["python"]))
            )
        if info["floor"]:
            bits.append(
                "needs python >=" + vstr(min(info["floor"]))
            )
        if not info["python"] and not info["floor"]:
            bits.append("no python dependency")
        if info["gil"]:
            bits.append("requires python-gil")
        res.ok("toolchain", "; ".join(bits))

    solved = set()
    for info in packages.values():
        solved |= info["python"]
    floors = set()
    for info in packages.values():
        floors |= info["floor"]

    if solved and declared is not None:
        if max(solved) != declared:
            res.note(
                "toolchain",
                "the lock solved against python %s but requires-python says "
                "%s" % (vstr(max(solved)), vstr(declared)),
            )
    if floors:
        res.note(
            "toolchain",
            "the lowest python any pinned toolchain package will accept is "
            "%s. An exact pin above that is the variant pixi chose, not a "
            "constraint the toolchain imposes" % vstr(min(floors)),
        )
    res.info(
        "pixi.lock records the solved variant only. To enumerate the "
        "variants the channel publishes, run this by hand (it queries the "
        "network, so this script does not):"
    )
    res.info("    pixi search --channel https://conda.modular.com/max max")


# ---------------------------------------------------------------------------
# Section 4: the built extension
# ---------------------------------------------------------------------------


def scan_extension(res, path, declared):
    print("\nextension: %s" % path)
    if not path.is_file():
        res.note(
            "extension",
            "not built, nothing to scan. Build it with bindings/build.sh and "
            "run this script again",
        )
        return
    blob = path.read_bytes()

    guarded = set(
        m.group(1).decode("ascii") for m in GUARD_STRING.finditer(blob)
    )
    names = set(
        m.group(0).decode("ascii") for m in SYMBOL_TOKEN.finditer(blob)
    )
    entry_points = sorted(n for n in names if n in SYMBOL_FLOORS)

    res.ok(
        "extension",
        "%d bytes, %d CPython entry point names found, %d of them with a "
        "floor above 3.9" % (len(blob), len(names), len(entry_points)),
    )
    if guarded:
        res.ok(
            "extension",
            "resolved conditionally by the runtime: %s"
            % ", ".join(sorted(guarded)),
        )
    else:
        res.note(
            "extension",
            "no conditional-resolution diagnostic found. Either the runtime "
            "guards nothing or it words the message differently",
        )

    hard = (3, 0)
    for symbol in entry_points:
        version = SYMBOL_FLOORS[symbol]
        state = "guarded" if symbol in guarded else "unguarded"
        res.info("%s needs CPython %s (%s)" % (symbol, vstr(version), state))
        if symbol not in guarded:
            hard = max(hard, version)

    if hard > (3, 0):
        res.ok(
            "extension",
            "highest unguarded entry point needs CPython %s" % vstr(hard),
        )
        if declared is not None and hard > declared:
            res.fail(
                "extension",
                "the extension needs at least %s but requires-python says %s"
                % (vstr(hard), vstr(declared)),
            )
        elif declared is not None and hard < declared:
            res.note(
                "extension",
                "the extension's own floor is %s, below the declared %s. "
                "Whether that gap is real depends on whether these names are "
                "resolved eagerly at module init or lazily on first use, "
                "which a byte scan cannot tell. docs/PYTHON_SUPPORT.md names "
                "the experiment that can" % (vstr(hard), vstr(declared)),
            )

    if b"libpython" in blob:
        res.note(
            "extension",
            "the binary mentions libpython by name, which is what a runtime "
            "that dlopens the interpreter looks like",
        )
    res.info(
        "a byte scan finds names. To tell a linked dependency from a "
        "runtime lookup, run one of these by hand:"
    )
    res.info("    otool -L %s && nm -u %s   # macOS" % (path, path))
    res.info(
        "    readelf -d %s && nm -D --undefined-only %s   # Linux" % (path, path)
    )


# ---------------------------------------------------------------------------
# Section 5: the running interpreter
# ---------------------------------------------------------------------------


def scan_interpreter(res, declared):
    print("\ninterpreter: the one running this script")
    version = sys.version_info[:2]
    res.ok("interpreter", "CPython %s (%s)" % (vstr(version), sys.executable))
    res.info("implementation: %s" % sys.implementation.name)
    res.info("platform tag:   %s" % sysconfig.get_platform())
    res.info("EXT_SUFFIX:     %s" % sysconfig.get_config_var("EXT_SUFFIX"))
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    res.info("free-threaded:  %s" % free_threaded)
    if free_threaded:
        res.note(
            "interpreter",
            "this is a free-threaded build. Every max 26.5.0 variant depends "
            "on python-gil, so no mojoboost extension exists for it",
        )
    if declared is not None and version < declared:
        res.note(
            "interpreter",
            "this interpreter is below the declared floor of %s. That is a "
            "supported way to run this script and says nothing about the "
            "package" % vstr(declared),
        )
    if sys.implementation.name != "cpython":
        res.note(
            "interpreter",
            "not CPython. The Mojo runtime resolves CPython C entry points "
            "by name, so no other implementation is a candidate",
        )


# ---------------------------------------------------------------------------


def main(argv):
    if sys.version_info[:2] < MIN_TOOL_PYTHON:
        sys.stderr.write(
            "this script needs CPython %s or newer\n" % vstr(MIN_TOOL_PYTHON)
        )
        return 1

    extension = EXTENSION
    args = list(argv[1:])
    if args:
        if args[0] in ("-h", "--help"):
            print(__doc__)
            return 0
        extension = Path(args[0])

    print("auditing python compatibility for %s" % ROOT)
    res = Report()
    source_floor = scan_source(res)
    declared = scan_metadata(res, source_floor)
    scan_toolchain(res, declared)
    scan_extension(res, extension, declared)
    scan_interpreter(res, declared)

    print("")
    if res.failures:
        print("%d failure(s):" % len(res.failures))
        for section, message in res.failures:
            print("  - %s: %s" % (section, message))
        return 1
    print("no contradictions found in %d note(s)" % len(res.notes))
    print(
        "this is a consistency check, not evidence that any interpreter "
        "works. See docs/PYTHON_SUPPORT.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
