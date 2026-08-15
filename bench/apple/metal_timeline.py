#!/usr/bin/env python3
"""Reduce an Instruments Metal System Trace to the numbers a GPU lane argues about.

This is the reader half of `bench/apple/metal_capture.sh`. The capture half
records a `.trace` bundle; this half exports the tables out of it with
`xctrace export` and turns twenty-odd thousand rows into per-dispatch and
per-round statistics. It is separate from the capture so that a trace taken
once can be re-read as many times as a question changes, and so that a trace
taken on someone else's machine can be read on this one.

WHAT IT MEASURES

Every number this script prints comes from a timestamp Instruments recorded.
Nothing here is fitted, modeled, or extrapolated. The four tables it reads are:

  metal-gpu-intervals
      One row per encoder that ran on the GPU, with the GPU's own start
      timestamp and duration. This is the only table that says what the GPU
      hardware was doing, and it is the basis of every busy/idle number below.

  metal-application-command-buffer-submissions
      One row per `commit`, on the CPU, with the duration of the commit call
      itself. This is the enqueue cost.

  metal-command-buffer-completed
      One row per completion notification delivered back to the process.

  metal-gpu-state-intervals
      The GPU hardware's own Active/Idle record, for the whole machine rather
      than for one process. It is not filtered by pid and so includes the
      window server and anything else drawing, which is why it is used only as
      a cross-check on the per-process union and never as the headline.

WHAT IT CANNOT MEASURE

Occupancy, ALU utilization, memory bandwidth, cache hit rate, and register
pressure are all absent, because the Apple M4 in this machine exposes exactly
one GPU counter to Instruments ("RT Unit Active", the raytracing unit) and
that counter is useless to a histogram kernel. Asking `xctrace` for the
`Metal GPU Counters` instrument makes it worse rather than better: the request
is refused with "Selected counter profile is not supported on target device"
and the one counter that would otherwise have been recorded is dropped too.
There is therefore no way, from this tool, to say whether a kernel is
latency-bound or bandwidth-bound directly. Duration against problem size is
the only lever left, and it is inference, not observation.

Kernel identity is also absent. Metal System Trace labels every dispatch
`Compute Command 0` and every copy `Blit Command 0`, because MAX does not
push debug groups or set labels on its encoders. Which kernel ran is therefore
recovered by position and duration, not by name, and every such attribution in
the output below is marked as inferred.

USAGE

    python3 bench/apple/metal_timeline.py <trace-path> [--process NAME]
    python3 bench/apple/metal_timeline.py <trace-path> --rounds 100

`--rounds` is the number of boosting rounds the run performed. It is used only
to divide per-run totals into per-round ones, and to cut the dispatch stream
into rounds at the N longest kernels, which assumes exactly one root histogram
per round. Pass 0 to skip all per-round output.
"""

import argparse
import collections
import os
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

TABLES = (
    "metal-gpu-intervals",
    "metal-application-command-buffer-submissions",
    "metal-command-buffer-completed",
    "metal-gpu-state-intervals",
    "gpu-performance-state-intervals",
    "gpu-counter-info",
)


# ---------------------------------------------------------------------------
# xctrace export, and the interned XML it produces.
# ---------------------------------------------------------------------------


def export(trace, schema, cache_dir):
    """Export one schema out of the trace, caching the XML next to the trace.

    `xctrace export` takes tens of seconds on a table of this size, and a
    reader that re-exports on every question is a reader nobody runs twice.
    """
    path = os.path.join(cache_dir, schema + ".xml")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    xpath = "/trace-toc/run[@number='1']/data/table[@schema='%s']" % schema
    with open(path, "wb") as fh:
        rc = subprocess.call(
            ["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath],
            stdout=fh,
            stderr=subprocess.DEVNULL,
        )
    if rc != 0:
        raise SystemExit("xctrace export failed for schema %s (rc=%d)" % (schema, rc))
    return path


def load(path):
    """Parse an exported table into (column mnemonics, rows, resolver).

    The export interns repeated values: an element carries `id="N"` the first
    time a value appears anywhere in the document and `ref="N"` on every later
    appearance. A row is a flat sequence of direct child elements, one per
    schema column, in the order the schema declares them. Both facts are
    undocumented by Apple and both are load-bearing here.
    """
    root = ET.parse(path).getroot()
    idmap = {}
    for el in root.iter():
        i = el.get("id")
        if i is not None:
            idmap[i] = el

    def resolve(el):
        r = el.get("ref")
        return idmap.get(r, el) if r is not None else el

    schema = root.find(".//schema")
    if schema is None:
        return [], [], resolve
    cols = [c.findtext("mnemonic") for c in schema.findall("col")]
    rows = [[resolve(child) for child in list(row)] for row in root.iter("row")]
    return cols, rows, resolve


def txt(el):
    if el is None:
        return ""
    if el.text and el.text.strip():
        return el.text.strip()
    return el.get("fmt", "") or ""


def num(el):
    s = txt(el).replace(",", "")
    try:
        return int(s)
    except ValueError:
        return 0


def pid_of(el, resolve):
    if el is None:
        return None
    for d in el.iter("pid"):
        return num(resolve(d))
    return None


def label_of(el, resolve):
    if el is None:
        return ""
    s = el.find("string")
    if s is not None:
        return txt(resolve(s))
    return txt(el)


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------


def pctl(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


def stat_line(name, xs):
    if not xs:
        print("  %-36s none" % name)
        return
    print(
        "  %-36s n=%6d  total=%9.2f ms  mean=%8.2f us  med=%8.2f us  p90=%8.2f us"
        % (
            name,
            len(xs),
            sum(xs) / 1e6,
            statistics.mean(xs) / 1e3,
            statistics.median(xs) / 1e3,
            pctl(xs, 0.90) / 1e3,
        )
    )


def head(title):
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------


def target_pid(trace, want):
    """The pid Instruments launched, read out of the trace's own table of
    contents rather than guessed from the process list."""
    out = subprocess.check_output(
        ["xcrun", "xctrace", "export", "--input", trace, "--toc"],
        stderr=subprocess.DEVNULL,
    ).decode("utf-8", "replace")
    m = re.search(r'<process arguments="([^"]*)"[^>]*name="([^"]+)" pid="(\d+)"', out)
    if m and (want is None or m.group(2) == want):
        return int(m.group(3)), m.group(2), m.group(1)
    for m in re.finditer(r'<process name="([^"]+)" pid="(\d+)"', out):
        if want and m.group(1) == want:
            return int(m.group(2)), m.group(1), ""
    raise SystemExit("could not find the launched process in the trace table of contents")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--process", default=None, help="target process name")
    ap.add_argument("--rounds", type=int, default=0, help="boosting rounds in the run")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    trace = os.path.abspath(args.trace)
    if not os.path.exists(trace):
        raise SystemExit("no such trace: %s" % trace)
    cache = args.cache_dir or (trace.rstrip("/") + ".export")
    os.makedirs(cache, exist_ok=True)

    pid, pname, pargs = target_pid(trace, args.process)
    print("trace     %s" % trace)
    print("process   %s (pid %d) %s" % (pname, pid, pargs))
    print("export    %s" % cache)

    paths = {t: export(trace, t, cache) for t in TABLES}

    # ---- Which GPU counters exist at all -------------------------------
    cols, rows, resolve = load(paths["gpu-counter-info"])
    names = [txt(r[2]) for r in rows]
    head("GPU counters available on this device")
    if names:
        for n in names:
            print("  %s" % n)
    else:
        print("  none")
    print("  (occupancy, ALU utilization, and bandwidth are not among them,")
    print("   so no kernel here can be called latency- or bandwidth-bound from a counter)")

    # ---- GPU execution intervals ---------------------------------------
    cols, rows, resolve = load(paths["metal-gpu-intervals"])
    ev = []
    for r in rows:
        if pid_of(r[10], resolve) != pid:
            continue
        lab = label_of(r[6], resolve)
        ev.append(
            (num(r[0]), num(r[1]), num(r[4]), "Blit" if "Blit" in lab else "Compute", txt(r[15]))
        )
    if not ev:
        raise SystemExit("no GPU intervals for pid %d; was the run actually on the GPU?" % pid)
    ev.sort()
    t0 = ev[0][0]
    t1 = max(s + d for s, d, _, _, _ in ev)
    span = t1 - t0

    # Union of busy time. Merging matters: two encoders can overlap on the
    # GPU, and summing durations would then claim more busy time than the
    # clock allows.
    merged = []
    cs, ce = ev[0][0], ev[0][0] + ev[0][1]
    for s, d, _, _, _ in ev[1:]:
        if s <= ce:
            ce = max(ce, s + d)
        else:
            merged.append((cs, ce))
            cs, ce = s, s + d
    merged.append((cs, ce))
    busy = sum(e - s for s, e in merged)

    comp = [(s, d) for s, d, _, k, _ in ev if k == "Compute"]
    blit = [(s, d) for s, d, _, k, _ in ev if k == "Blit"]

    head("Question 1: is the GPU busy between kernels")
    print("  dispatches                    %d  (%d compute, %d blit)"
          % (len(ev), len(comp), len(blit)))
    print("  span, first GPU start to last GPU end   %9.2f ms" % (span / 1e6))
    print("  GPU busy (union of intervals)           %9.2f ms  (%.2f%% of span)"
          % (busy / 1e6, 100.0 * busy / span))
    print("  GPU idle inside the span                %9.2f ms  (%.2f%% of span)"
          % ((span - busy) / 1e6, 100.0 * (span - busy) / span))
    print("  compute GPU time                        %9.2f ms  (%.2f%% of span)"
          % (sum(d for _, d in comp) / 1e6, 100.0 * sum(d for _, d in comp) / span))
    print("  blit GPU time (bytes actually moving)   %9.2f ms  (%.2f%% of span)"
          % (sum(d for _, d in blit) / 1e6, 100.0 * sum(d for _, d in blit) / span))

    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    if gaps:
        print()
        stat_line("gaps between busy runs", gaps)
        print("  gap size distribution:")
        edges = [(0, 1e3), (1e3, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e6), (1e6, 1e15)]
        labels = ["<1us", "1-5us", "5-20us", "20-100us", "0.1-1ms", ">1ms"]
        for (lo, hi), nm in zip(edges, labels):
            sel = [g for g in gaps if lo <= g < hi]
            if sel:
                print("    %-9s n=%6d  total=%9.2f ms  (%.1f%% of all idle)"
                      % (nm, len(sel), sum(sel) / 1e6, 100.0 * sum(sel) / (span - busy)))

    # Cross-check against the hardware's own Active/Idle record, which is not
    # filtered by process and so should be slightly larger.
    cols, rows, resolve = load(paths["metal-gpu-state-intervals"])
    st = collections.Counter()
    for r in rows:
        st[txt(r[2])] += num(r[1])
    if st:
        print()
        print("  cross-check, GPU hardware Active/Idle for the whole machine over the whole trace:")
        for k, v in st.most_common():
            print("    %-10s %9.2f ms" % (k, v / 1e6))
        print("    (this counts every process, so Active here should exceed the")
        print("     per-process busy above by whatever else touched the GPU)")

    cols, rows, resolve = load(paths["gpu-performance-state-intervals"])
    ps = collections.Counter()
    for r in rows:
        ps[txt(r[2])] += num(r[1])
    if ps:
        print()
        print("  GPU performance state over the whole trace (rules out idleness being a clock artifact):")
        for k, v in ps.most_common():
            print("    %-10s %9.2f ms" % (k, v / 1e6))

    # ---- Kernel size distribution --------------------------------------
    head("Question 2: what the kernels look like (no counters, so shape only)")
    durs = [d for _, d in comp]
    print("  compute kernel duration percentiles, microseconds:")
    print("    p10 %8.2f   p25 %8.2f   p50 %8.2f   p75 %8.2f   p90 %8.2f   p99 %8.2f   max %8.2f"
          % tuple(pctl(durs, q) / 1e3 for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)))
    tot = sum(durs)
    print("  where compute GPU time goes, by kernel duration class:")
    edges = [0, 2e3, 5e3, 10e3, 25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 1e15]
    labels = ["<2us", "2-5us", "5-10us", "10-25us", "25-50us", "50-100us",
              "100-250us", "250-500us", "0.5-1ms", ">1ms"]
    for (lo, hi), nm in zip(zip(edges, edges[1:]), labels):
        sel = [d for d in durs if lo <= d < hi]
        if sel:
            print("    %-10s n=%6d  total=%8.2f ms  (%5.1f%% of compute GPU time)"
                  % (nm, len(sel), sum(sel) / 1e6, 100.0 * sum(sel) / tot))

    # ---- CPU side and the round trip -----------------------------------
    cols, rows, resolve = load(paths["metal-application-command-buffer-submissions"])
    commits = {}
    for r in rows:
        if pid_of(r[7], resolve) != pid:
            continue
        commits[txt(r[14])] = (num(r[0]), num(r[1]))

    gpu = {}
    for s, d, lat, k, cb in ev:
        if cb in gpu:
            ps_, pd_, pk_, pl_ = gpu[cb]
            gpu[cb] = (min(ps_, s), max(ps_ + pd_, s + d) - min(ps_, s), pk_, pl_)
        else:
            gpu[cb] = (s, d, k, lat)

    cols, rows, resolve = load(paths["metal-command-buffer-completed"])
    done = {txt(r[1]): num(r[0]) for r in rows}

    joined = sorted(set(commits) & set(gpu))
    recs = []
    for cb in joined:
        c_s, c_d = commits[cb]
        g_s, g_d, kind, lat = gpu[cb]
        recs.append((c_s, c_d, g_s, g_s + g_d, g_d, kind, done.get(cb)))
    recs.sort()

    head("Question 3: what one enqueue costs, and what one host wait costs")
    print("  command buffers joined across CPU, GPU, and completion tables: %d" % len(recs))
    print()
    stat_line("enqueue: the commit call on CPU", [r[1] for r in recs])
    stat_line("commit end -> GPU start", [r[2] - (r[0] + r[1]) for r in recs])
    stat_line("GPU busy per command buffer", [r[4] for r in recs])
    stat_line("GPU end -> completion signal", [r[6] - r[3] for r in recs if r[6]])
    stat_line("commit N -> commit N+1", [recs[i + 1][0] - recs[i][0] for i in range(len(recs) - 1)])

    stalls = []
    for i in range(len(recs) - 1):
        turn = recs[i + 1][0] - recs[i][3]
        if turn > 0:
            stalls.append((turn, recs[i][5], recs[i + 1][5]))
    print()
    print("  A serialization point is a command buffer whose successor was committed only")
    print("  after the GPU had finished it. Anything else was already in flight, so the")
    print("  host was pipelining and paid no wait for it.")
    print("    serialization points   %d of %d command buffers (%.1f%%)"
          % (len(stalls), len(recs), 100.0 * len(stalls) / len(recs)))
    print("    host time stalled      %.2f ms (%.1f%% of the span)"
          % (sum(t for t, _, _ in stalls) / 1e6,
             100.0 * sum(t for t, _, _ in stalls) / span))
    kinds = collections.Counter(k for _, _, _, _, _, k, _ in recs)
    waited = collections.Counter(a for _, a, _ in stalls)
    for k in sorted(kinds):
        print("    %-8s %6d total, %6d of them serialization points (%.1f%%)"
              % (k, kinds[k], waited[k], 100.0 * waited[k] / kinds[k]))

    # ---- Per round ------------------------------------------------------
    if args.rounds > 0:
        head("Per boosting round (%d rounds)" % args.rounds)
        # Cut the stream at the N longest kernels. This assumes exactly one
        # root histogram per round and that it is the longest dispatch in the
        # round, which the spacing check below either supports or refutes.
        # These two are plain division and are always valid.
        print("  serialization points per round  %.1f" % (len(stalls) / float(args.rounds)))
        print("  dispatches per round            %.1f" % (len(recs) / float(args.rounds)))
        print()

        # Everything past here depends on cutting the dispatch stream at the N
        # longest kernels, which is only meaningful if there really is one
        # dominant kernel per round. At small row counts the root histogram
        # stops dominating, the N longest kernels bunch together inside a few
        # rounds, and the cut produces per-round figures that are arithmetically
        # real and physically meaningless. The spacing test below catches that.
        top = sorted(sorted(comp, key=lambda x: -x[1])[: args.rounds])
        deltas = [top[i + 1][0] - top[i][0] for i in range(len(top) - 1)]
        expected = span / float(args.rounds)
        spacing = statistics.median(deltas)
        print("  the %d longest kernels are spaced   median %.2f ms apart" % (args.rounds, spacing / 1e6))
        print("  one per round would be             %.2f ms apart" % (expected / 1e6))
        print("  their own durations                median %.1f us (min %.1f, max %.1f)"
              % (statistics.median(d for _, d in top) / 1e3,
                 min(d for _, d in top) / 1e3, max(d for _, d in top) / 1e3))
        if spacing < 0.5 * expected:
            print()
            print("  The %d longest kernels are NOT one per round: they are spaced %.0f%% of a" % (args.rounds, 100.0 * spacing / expected))
            print("  round apart, so they bunch inside a few rounds instead of marking them.")
            print("  No per-round cut is reported, because one taken here would be a number")
            print("  about the wrong intervals. This is normal at small row counts, where the")
            print("  root histogram stops being the largest thing in its round.")
            print()
            print("Every number above is a recorded timestamp. Lines marked INFERRED are not.")
            return
        print("  INFERRED: one per round, and the root histogram, because the count and")
        print("  the spacing both match the round count. The trace does not name kernels.")
        rounds = []
        allev = sorted((s, d) for s, d, _, _, _ in ev)
        for i in range(len(top) - 1):
            a, b = top[i][0], top[i + 1][0]
            sel = [(s, d) for s, d in allev if a <= s < b]
            if sel:
                rounds.append((b - a, sum(d for _, d in sel), len(sel)))
        if rounds:
            print()
            print("  wall per round        median %8.2f ms" % (statistics.median(r[0] for r in rounds) / 1e6))
            print("  GPU busy per round    median %8.2f ms" % (statistics.median(r[1] for r in rounds) / 1e6))
            print("  dispatches per round  median %8.0f" % statistics.median(r[2] for r in rounds))
            occ = [100.0 * r[1] / r[0] for r in rounds]
            print("  GPU busy fraction     median %8.1f%%  (min %.1f%%, max %.1f%%)"
                  % (statistics.median(occ), min(occ), max(occ)))

    print()
    print("Every number above is a recorded timestamp. Lines marked INFERRED are not.")


if __name__ == "__main__":
    main()
