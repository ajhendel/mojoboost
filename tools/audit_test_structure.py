#!/usr/bin/env python3
"""Check that every `tests/test_*.mojo` is shaped like a file that can run.

Why this exists, with the case that caused it.
`tests/test_objective_reserved_codes.mojo` was written on 2026-08-16,
imported `from testing import ...` rather than `from std.testing import ...`,
and had no `def main()`. It therefore did not parse and, had it parsed, would
have run nothing. It sat in `tests/`, it was named like a test, and
`tools/run_tests.sh` selected it by glob, so every count of "how many tests
does this repository have" included it for as long as it existed. It was the
file that pinned the seven reserved objective codes -- the integers that ARE
the model file format, and the ones two lanes independently assigned to the
same values in a single round -- and not one of its assertions had ever
touched the code.

A test file that cannot run is worse than a missing one, because a missing
one is visible. This gate makes that shape impossible to commit.

WHAT THIS CHECKS, all of it by reading the file as text:

  T1  the file defines a top-level `def main(`;
  T2  `main` actually runs the tests -- its body reaches `TestSuite` or
      `discover_tests`, or calls the file's own `test_*` functions.  A
      `def main(): pass` satisfies T1 and runs nothing;
  T3  the file defines at least one top-level `def test_*`.  Under
      `discover_tests[__functions_in_module()]` a file with none is a green
      run of zero tests;
  T4  every `def test_*` is at top level.  An indented one is invisible to
      `__functions_in_module()` and silently never runs;
  T5  testing symbols are imported from `std.testing`, not from a bare
      `testing`.  This is the import that did not resolve.

WHAT THIS DOES NOT CHECK, stated plainly because a gate that is believed to
prove more than it proves is worse than no gate:

  * **It does not prove the file compiles.**  It never invokes `mojo`.  A
    file can pass all five checks above and still fail to parse -- a typo in
    a type, an import of a symbol the package does not export, a syntax
    error anywhere below the lines this reads.  The suite is what proves a
    file compiles; `bash tools/run_tests.sh` is that check and this is not a
    substitute for it.
  * It does not check that any assertion is true, that any assertion
    discriminates, or that a `test_*` function asserts anything at all.
  * It reads text with regular expressions rather than parsing Mojo, so it
    is fooled by a `def main(` inside a string or a comment.  That is an
    accepted limit: the failure it is built to catch is an omission, and you
    cannot omit something by mentioning it in a comment.

Compiling all 154 test files takes minutes and is exactly what the suite
does, so the compile check is deliberately NOT the gated part.  The gated
part is structural, costs milliseconds, and catches the one defect that the
suite itself could not report -- because the file that could not parse
aborted the run rather than being counted as a failure.

SCOPE.  Files named `tests/test_*.mojo`, and only those.  `tests/support.mojo`
is a helper module with no `def main()` by design and is correctly out of
scope: the gate keys on the `test_` prefix, not on the directory.

Standard library only.  Nothing here builds, runs Mojo, imports the
extension, touches the network, or writes inside the repository.  Exit status
is 0 when clean and 1 when a check fails.

Usage:
    python3 tools/audit_test_structure.py
    python3 tools/audit_test_structure.py --tests OTHER_DIR   # for testing
    python3 tools/audit_test_structure.py --list              # per-file table
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# A top-level `def NAME(`. Column zero is the whole point of the pattern:
# `__functions_in_module()` sees module-level functions and nothing else.
TOP_LEVEL_DEF = re.compile(r"^def[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(", re.M)

# A `def test_*` that is indented, i.e. nested inside something and therefore
# invisible to discovery.
NESTED_TEST_DEF = re.compile(r"^[ \t]+def[ \t]+(test_[A-Za-z0-9_]*)[ \t]*\(", re.M)

# `from testing import ...` / `import testing` -- the module path that does
# not resolve. `std.testing` is the one that does, and the negative lookahead
# is what keeps this from firing on it.
BARE_TESTING_IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+testing[ \t]+import\b|import[ \t]+testing\b)", re.M
)

# Evidence inside `main` that the tests are actually reached.
RUNS_THE_SUITE = ("TestSuite", "discover_tests")

problems: list[str] = []


def fail(path: Path, check: str, message: str) -> None:
    problems.append(f"{check}  {path.name}: {message}")


def main_body(text: str) -> str | None:
    """The source of the top-level `def main(...)`, or None if there is none.

    The body runs from the `def main` line to the next top-level `def`, or to
    the end of the file. Crude, and sufficient: the question is only whether
    the tests are reached from inside it.
    """
    match = re.search(r"^def[ \t]+main[ \t]*\(", text, re.M)
    if match is None:
        return None
    start = match.start()
    rest = text[match.end() :]
    following = re.search(r"^def[ \t]+", rest, re.M)
    end = match.end() + following.start() if following else len(text)
    return text[start:end]


def check_file(path: Path) -> None:
    try:
        text = path.read_text()
    except OSError as exc:  # pragma: no cover - filesystem dependent
        fail(path, "T0", f"cannot be read: {exc}")
        return

    top_level = TOP_LEVEL_DEF.findall(text)
    test_functions = [name for name in top_level if name.startswith("test_")]

    # T5 first: it is the one that stops the file from parsing at all, so it
    # is the most useful line to see when several fire together.
    if BARE_TESTING_IMPORT.search(text):
        fail(
            path,
            "T5",
            "imports from `testing`; the module is `std.testing`, and this "
            "import is why the file does not parse",
        )

    body = main_body(text)
    if body is None:
        fail(
            path,
            "T1",
            "has no top-level `def main(`, so nothing in it ever runs. Add:\n"
            "        def main() raises:\n"
            "            TestSuite.discover_tests[__functions_in_module()]().run()",
        )
    elif not any(token in body for token in RUNS_THE_SUITE) and not any(
        name in body for name in test_functions
    ):
        fail(
            path,
            "T2",
            "`def main` reaches neither TestSuite/discover_tests nor any of "
            f"its {len(test_functions)} test function(s), so it runs no tests",
        )

    if not test_functions:
        fail(
            path,
            "T3",
            "defines no top-level `def test_*`, so discovery finds nothing "
            "and the file reports a green run of zero tests",
        )

    for name in NESTED_TEST_DEF.findall(text):
        fail(
            path,
            "T4",
            f"`{name}` is indented, so `__functions_in_module()` cannot see "
            "it and it never runs. Move it to top level",
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="check test file structure")
    parser.add_argument("--tests", type=Path, default=TESTS)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print each file with its top-level test count",
    )
    args = parser.parse_args(argv)

    if not args.tests.is_dir():
        print(f"no tests directory at {args.tests}", file=sys.stderr)
        return 1

    paths = sorted(args.tests.glob("test_*.mojo"))
    if not paths:
        print(f"no test_*.mojo under {args.tests}", file=sys.stderr)
        return 1

    print(f"checking the structure of {len(paths)} test files in {args.tests.name}/")
    for path in paths:
        check_file(path)

    if args.list:
        for path in paths:
            names = TOP_LEVEL_DEF.findall(path.read_text())
            count = len([n for n in names if n.startswith("test_")])
            print(f"  {count:4d}  {path.name}")

    # Named rather than counted, so the exemption is visible instead of being
    # a file that quietly never appears.
    helpers = sorted(
        p.name for p in args.tests.glob("*.mojo") if not p.name.startswith("test_")
    )
    if helpers:
        print(f"  not checked (no test_ prefix, helpers): {', '.join(helpers)}")

    if problems:
        print()
        for message in problems:
            print(f"  {message}", file=sys.stderr)
        print(
            f"\n{len(problems)} structural problem(s) in {args.tests.name}/."
            "\nA file that cannot run is counted as a test by every glob in "
            "this repository.",
            file=sys.stderr,
        )
        return 1

    print("  ok: every file defines main, runs the suite, and has tests to run")
    print("  (structure only; this proves nothing about whether they compile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
