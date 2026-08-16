# mojotrees, for an agent starting here

Four files decide how work is done in this repository. Read them before
proposing anything; they are contracts rather than advice.

- **`bench/results/LANE_RULES.md`** — what a lane may do, what it may run, and
  the four categories that decide whether a change is built, switched, gated,
  or closed. Both the CPU and GPU campaigns brief from it.
- **`bench/results/PROFILE_PROTOCOL.md`** — the measurement rules, registered
  before the data they judge. M0 resolved / consistent / indistinguishable, the
  quiet-box precondition, and the provenance labels every number carries.
- **`bench/results/MACHINE_LOCK.md`** — two campaigns share one machine. The
  timing lock, the worktree rules, and why a suite counts as a compile.
- **The comparator: LightGBM at stock defaults plus `deterministic=true`,
  labelled `stock+det`,** defined in `bench/real_data/scenarios.py` and
  registered as section C9 of `PROFILE_PROTOCOL.md`. Every published number is
  end-to-end, binning plus training, against it. No other configuration is
  published.
