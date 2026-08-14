#!/bin/sh
# Serialize heavy builds/tests across parallel sessions.
# macOS has no flock(1), so this blocks on a Python fcntl lock without polling.
# Usage: tools/with_build_lock.sh <command> [args...]
exec /usr/bin/python3 -c '
import fcntl, subprocess, sys
f = open("/tmp/mojoboost-build.lock", "w")
fcntl.flock(f, fcntl.LOCK_EX)
sys.exit(subprocess.call(sys.argv[1:]))
' "$@"
