#!/usr/bin/env python3
"""Check pixi.toml against the files and tasks that actually exist.

Why this exists. `pixi.toml` is a gate artifact in the same sense as
`compatibility/api_snapshot.json` and `docs/LIGHTGBM_PARITY.md`: it makes a
claim about the repository, and nothing used to check that claim. A task can
name `bench/bench_removed.mojo` long after the file went, a `depends-on` can
name a task nobody defines, and an `[environments]` entry can name a feature
that is not declared. Each of those fails at `pixi run` time, on somebody
else's machine, for a reason that has nothing to do with what they changed.

What it checks, all of it by reading files and none of it by running them:

  P1  every repository path a task command names exists;
  P2  every `-I` include directory a task names exists;
  P3  every suite name handed to `tools/run_tests.sh` has a `tests/` file;
  P4  every `depends-on` entry names a task defined somewhere in the file;
  P5  every `[environments]` member names a declared `[feature.*]`.

Build outputs are exempt, because a clean checkout has not produced them yet:
anything under `build/` and anything following `-o`.

Notes are printed and do not fail. The only one today is a benchmark entry
point that no task runs, which is drift worth seeing but not worth blocking a
commit over, since a benchmark can legitimately be reachable only by hand.

Standard library only, Python 3.11 or later for `tomllib`. Nothing here
builds, runs, imports the extension, touches the network, or writes inside
the repository. Exit status is 0 when clean and 1 when a check fails.

Usage:
    python3 tools/check_pixi_tasks.py
    python3 tools/check_pixi_tasks.py --pixi OTHER.toml   # for testing

`--pixi` reads a different manifest and still resolves the paths inside it
against this repository, which is how the checks themselves get exercised
without editing the real pixi.toml.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

# Before importing tomllib, so an old interpreter gets this sentence instead
# of a bare ImportError from the middle of the import block.
if sys.version_info < (3, 11):  # pragma: no cover - environment dependent
    sys.exit(
        "tools/check_pixi_tasks.py needs Python 3.11 or later for tomllib; "
        f"this is {sys.version.split()[0]}"
    )

import tomllib  # noqa: E402 - deliberately after the version guard

ROOT = Path(__file__).resolve().parent.parent
PIXI = ROOT / "pixi.toml"

# A token is treated as a repository path when it carries one of these
# suffixes or contains a separator. Bare words like `mojo`, `pytest`, and
# `-q` are commands and flags, not paths.
PATH_SUFFIXES = (".mojo", ".py", ".sh", ".toml", ".json", ".md")

# Produced by a task rather than checked into the tree.
GENERATED_PREFIXES = ("build/",)

SUITE_NAME = re.compile(r"^test_[A-Za-z0-9_]+$")
RUN_TESTS = "tools/run_tests.sh"
RUN_TESTS_MODES = {"all", "cpu", "gpu", "list"}

problems: list[str] = []
notes: list[str] = []


def fail(message: str) -> None:
    problems.append(message)


def note(message: str) -> None:
    notes.append(message)


def load_tasks(doc: dict) -> dict[str, dict]:
    """{task name: {"cmd": str, "depends-on": [...], "where": section}}.

    Covers `[tasks]` and every `[feature.NAME.tasks]`, because a task in a
    feature is just as able to name a file that is gone.
    """
    tasks: dict[str, dict] = {}

    def absorb(table: dict, where: str) -> None:
        for name, body in table.items():
            if isinstance(body, str):
                entry = {"cmd": body, "depends-on": [], "where": where}
            elif isinstance(body, dict):
                entry = {
                    "cmd": body.get("cmd", ""),
                    "depends-on": list(body.get("depends-on", [])),
                    "where": where,
                }
            else:  # a list-form command
                entry = {"cmd": " ".join(map(str, body)), "depends-on": [], "where": where}
            if name in tasks:
                fail(
                    f"P4: task {name!r} is defined twice, in {tasks[name]['where']} "
                    f"and in {where}; the later one silently wins"
                )
            tasks[name] = entry

    absorb(doc.get("tasks", {}), "[tasks]")
    for feature, body in doc.get("feature", {}).items():
        absorb(body.get("tasks", {}), f"[feature.{feature}.tasks]")
    return tasks


def check_command_paths(name: str, cmd: str) -> None:
    """P1 and P2."""
    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        fail(f"P1: task {name!r} has an unparsable command: {exc}")
        return

    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in ("-o", "--output"):
            skip_next = True  # an output path need not exist yet
            continue
        if token == "-I":
            skip_next = True
            if index + 1 < len(tokens):
                include = tokens[index + 1]
                if include.startswith(GENERATED_PREFIXES) or include == "build":
                    continue
                if not (ROOT / include).is_dir():
                    fail(
                        f"P2: task {name!r} passes `-I {include}`, and no such "
                        f"directory exists"
                    )
            continue
        if token.startswith("-"):
            continue
        if token.startswith(GENERATED_PREFIXES):
            continue
        if "/" not in token and not token.endswith(PATH_SUFFIXES):
            continue
        candidate = ROOT / token
        if not candidate.exists():
            fail(f"P1: task {name!r} names {token}, and no such file exists")


def check_suite_names(name: str, cmd: str) -> None:
    """P3. `tools/run_tests.sh cpu test_capi` must find tests/test_capi.mojo."""
    if RUN_TESTS not in cmd:
        return
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return  # already reported by P1
    for token in tokens:
        if token in RUN_TESTS_MODES or not SUITE_NAME.match(token):
            continue
        if not (ROOT / "tests" / f"{token}.mojo").is_file():
            fail(
                f"P3: task {name!r} runs suite {token!r}, and "
                f"tests/{token}.mojo does not exist"
            )


def check_depends(tasks: dict[str, dict]) -> None:
    """P4."""
    for name, entry in tasks.items():
        for dependency in entry["depends-on"]:
            target = dependency if isinstance(dependency, str) else dependency.get("task", "")
            if target not in tasks:
                fail(
                    f"P4: task {name!r} depends on {target!r}, and pixi.toml "
                    f"defines no such task"
                )


def check_environments(doc: dict) -> None:
    """P5."""
    declared = set(doc.get("feature", {}))
    for environment, members in doc.get("environments", {}).items():
        if isinstance(members, dict):
            members = members.get("features", [])
        if isinstance(members, str):
            members = [members]
        for feature in members:
            if feature not in declared:
                fail(
                    f"P5: environment {environment!r} names feature "
                    f"{feature!r}, and no [feature.{feature}] section declares it"
                )


def check_unwired_benchmarks(tasks: dict[str, dict]) -> None:
    """A note, not a failure. A benchmark entry point no task names is the
    same class of drift the test runner's glob removed for suites, but a
    benchmark can be reachable only by hand on purpose."""
    named = " ".join(entry["cmd"] for entry in tasks.values())
    for path in sorted(ROOT.glob("bench/**/*.mojo")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in named:
            continue
        try:
            if "def main()" not in path.read_text(errors="replace"):
                continue  # a helper module, not an entry point
        except OSError:
            continue
        note(f"{relative} defines main() and no pixi task runs it")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixi", type=Path, default=PIXI)
    args = parser.parse_args(argv)

    manifest = args.pixi
    print(f"checking {manifest.name} against the tree")
    if not manifest.is_file():
        print(f"  no manifest at {manifest}", file=sys.stderr)
        return 1
    try:
        doc = tomllib.loads(manifest.read_text())
    except tomllib.TOMLDecodeError as exc:
        print(f"  {manifest.name} is not valid TOML: {exc}", file=sys.stderr)
        return 1

    tasks = load_tasks(doc)
    for name, entry in tasks.items():
        check_command_paths(name, entry["cmd"])
        check_suite_names(name, entry["cmd"])
    check_depends(tasks)
    check_environments(doc)
    check_unwired_benchmarks(tasks)

    print(f"  {len(tasks)} tasks across {1 + len(doc.get('feature', {}))} task tables")
    for message in notes:
        print(f"  note: {message}")
    if problems:
        print()
        for message in problems:
            print(f"  {message}", file=sys.stderr)
        print(f"\n{len(problems)} problem(s) in {manifest.name}", file=sys.stderr)
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
