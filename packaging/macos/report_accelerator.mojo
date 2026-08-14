"""Print this build's compile-time accelerator answer, and nothing else.

    pixi run mojo run -I src packaging/macos/report_accelerator.mojo

Prints exactly `true` or `false` on one line, for
packaging/macos/provenance.sh to record as `has_accelerator_at_build`.

NOT EXECUTED. It has never been compiled or run.

Why this exists. `has_accelerator()` is resolved at compile time, so
availability is a property of the build rather than of the machine running
it (src/mojoboost/device.mojo). A wheel compiled where an accelerator was
visible reports one as available wherever it is installed, and a `gpu`
request on such a build fails when the device is opened rather than when it
is resolved. That is the one field in the provenance sidecar that changes
what the artifact does, so it is measured rather than assumed.

This duplicates a four-line program that handoffs/task18_platform.md
proposed as `tools/report_accelerator.mojo`, which does not exist. If that
file lands, delete this one and change the single path in provenance.sh.
Two reporters that could disagree is worse than either.
"""

from std.sys import has_accelerator


def main() raises:
    comptime if has_accelerator():
        print("true")
    else:
        print("false")
