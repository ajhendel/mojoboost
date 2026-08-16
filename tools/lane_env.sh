#!/usr/bin/env bash
#
# Point this worktree at the MAIN checkout's pixi environment.
#
#   source tools/lane_env.sh
#
# Why this exists.  A lane works in a `git worktree`, and a worktree is a
# separate directory containing `pixi.toml`, so `pixi run` treats it as a
# separate project and installs a complete second copy of the environment
# into `<worktree>/.pixi`.  Measured 2026-08-16: 46 of 56 lane worktrees had
# done exactly that, roughly 1.1 GB each, **49 GB in total**, for an
# environment byte-identical to the one already sitting in the main checkout.
#
# The install itself is the expensive part, and it is paid before the lane
# compiles a single line.  Worse, `MODULAR_HOME` follows the environment, so
# each of those 46 worktrees also got its own empty Mojo compile cache: they
# measured between 1.1 MB and 243 MB against the main checkout's 8.7 GB.  No
# lane could see anything another lane had compiled, and none could see the
# main checkout's cache either.  Every lane started cold and stayed cold.
#
# What this does.  Exports the three variables `pixi run` would have exported
# -- `PATH`, `MODULAR_HOME`, `CONDA_PREFIX` -- pointing at the main
# checkout's environment.  Nothing is copied and nothing is installed.  The
# environment holds the toolchain and the Python packages and knows nothing
# about which checkout is calling it; your `src/` is still yours, reached
# through `-I src` from the directory you are standing in.
#
# The shared compile cache is safe to share.  Entries are content-keyed, and
# `mojo` is built to have several processes writing to one cache; that is the
# case CI exercises every run.  It is also the entire point: a lane that
# compiles a module another lane already compiled gets it for free.
#
# When this DECLINES, and why it is a decline rather than a failure.  If the
# worktree's `pixi.toml` differs from the main checkout's, the manifest this
# environment was solved for is not the manifest you are working against, and
# silently handing you the wrong toolchain is the kind of thing this project
# throws results away over.  It says so and leaves your environment alone, so
# the ordinary `pixi run` path still works and you get your own environment,
# slowly and correctly.
#
# Sourcing this from the main checkout is a no-op that succeeds, so a script
# can source it unconditionally without asking where it is.
#
# SCOPE, stated rather than discovered later.  This shares the `default`
# environment, which is the one that carries `mojo` and is the one a lane
# needs for every compile and every native test.  It does NOT cover the
# other environments in `pixi.toml` -- `pytest`, `bench`, `pkg` -- because
# `PATH` and `MODULAR_HOME` can each name one, and picking `default` is what
# makes `tools/run_tests.sh` work.  A lane that runs `pixi run -e pytest ...`
# in its worktree will still install that environment locally.  That is a
# much smaller environment than `default` and lanes rarely reach for it, so
# it is left alone rather than solved with a second mechanism.

# Not `set -e`: this file is sourced, and killing the caller's shell because
# a lookup failed is worse than declining.

_lane_env_warn() { echo "lane_env: $*" >&2; }

_lane_env_main() {
  local common main here

  if ! command -v git >/dev/null 2>&1; then
    _lane_env_warn "git is not on PATH, leaving the environment alone"
    return 1
  fi

  # `--git-common-dir` is the main checkout's `.git` from anywhere in any
  # linked worktree, which is exactly the question being asked.  In the main
  # checkout it is that checkout's own `.git`, so the two cases need no
  # separate handling.
  common="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || {
    _lane_env_warn "not inside a git repository, leaving the environment alone"
    return 1
  }
  main="$(dirname "$common")"
  here="$(git rev-parse --show-toplevel 2>/dev/null)"

  if [ "$main" = "$here" ]; then
    return 0  # already in the main checkout; pixi run does the right thing
  fi

  if [ ! -x "$main/.pixi/envs/default/bin/mojo" ]; then
    _lane_env_warn "the main checkout at $main has no pixi environment yet;"
    _lane_env_warn "run 'pixi run mojo --version' there once, then re-source this"
    return 1
  fi

  # The manifest guard.  Compared by content rather than mtime, since a
  # worktree checkout rewrites mtimes without changing anything.
  # Compare only the DEPENDENCY-BEARING tables, not the whole manifest.
  #
  # This used to be `cmp -s` over the entire file, and it cost a lane its
  # shared environment for adding one `[tasks]` line: a task entry with no
  # dependency change is byte-different and looks exactly like a different
  # solve. That lane exported the three variables by hand instead, which is
  # the right call and should not have been necessary, and a neighbouring
  # lane installed its own 1.1 GB copy rather than work around it, which is
  # the exact duplication this file exists to prevent. A guard that fires on
  # changes it does not care about teaches people to bypass it.
  #
  # What actually invalidates a shared solve is a dependency table. Tasks,
  # comments and metadata do not. `[*dependencies]` catches `dependencies`,
  # `pypi-dependencies`, `host-dependencies`, `build-dependencies` and every
  # `[feature.X.*dependencies]`, which is the whole set pixi solves from.
  _lane_env_deps() {
    awk '
      /^\[/ { keep = ($0 ~ /dependencies\]$/) }
      keep { print }
    ' "$1"
  }
  # SUBSET is safe; superset is not.
  #
  # The question is not "are the manifests equal" but "does this worktree
  # need anything the shared environment does not have". A worktree whose
  # dependency tables are a SUBSET of the main checkout's asks for nothing
  # that is missing, so sharing is correct and declining is pure cost.
  #
  # This matters because the main checkout drifts forward. Adding `pandas`
  # to `[feature.bench.dependencies]` made every worktree branched before
  # that commit look "different" and declined all of them, on both
  # campaigns, for a dependency the shared environment already had. The
  # equality test was still too strict after being narrowed once, and the
  # symptom was lanes silently exporting the three variables by hand or
  # installing their own 1.1 GB copy, which is the failure this file exists
  # to prevent, arriving through the guard rather than around it.
  #
  # Lines only in the worktree are what disqualify it. Lines only in the
  # main checkout are the environment being ahead, which is fine.
  _lane_env_extra="$(
    comm -23 <(_lane_env_deps "$here/pixi.toml" | sort -u) \
             <(_lane_env_deps "$main/pixi.toml" | sort -u) 2>/dev/null
  )"
  if [ -r "$here/pixi.toml" ] && [ -r "$main/pixi.toml" ] &&
     [ -n "$_lane_env_extra" ]; then
    _lane_env_warn "this worktree needs dependencies $main does not have:"
    _lane_env_warn "$_lane_env_extra"
    _lane_env_warn "DECLINING to share its environment: it was solved without"
    _lane_env_warn "them. Use 'pixi run' and take your own."
    return 1
  fi

  export CONDA_PREFIX="$main/.pixi/envs/default"
  export MODULAR_HOME="$CONDA_PREFIX/share/max"
  case ":$PATH:" in
    *":$CONDA_PREFIX/bin:"*) ;;
    *) export PATH="$CONDA_PREFIX/bin:$PATH" ;;
  esac
  return 0
}

_lane_env_main
_lane_env_status=$?
unset -f _lane_env_main _lane_env_warn
return $_lane_env_status 2>/dev/null || true
