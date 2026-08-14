"""Download the datasets in sources.json, verify them, and pin them.

Fetching is separate from running on purpose. A benchmark that downloads
gigabytes as a side effect of being run is a benchmark that gets run with
whatever it happened to download, and the run that first fetched a dataset
is the run least able to tell whether it got the right one.

    python bench/real_data/fetch.py --list
    python bench/real_data/fetch.py adult covertype
    python bench/real_data/fetch.py adult --pin
    python bench/real_data/fetch.py --tier standard

The two modes:

- Without `--pin`, a download is verified against checksums.lock.json and
  a mismatch is an error. This is the mode every machine except the first
  one uses.
- With `--pin`, the digest of whatever arrived is written into the lock.
  This is a human deciding that this is the right data, and the lock file
  is a commit that records the decision. Run it once, read what it wrote,
  and commit it.

Nothing here writes a digest that was not computed from bytes on disk.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import urllib.error
import urllib.request

import loaders

CHUNK = 1 << 20
USER_AGENT = "mojoboost-bench/0.1 (+https://github.com/mojoboost-ml/mojoboost)"


def _sha256_and_size(path):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _download(url, destination):
    """Stream a URL to disk through a .part file, so an interrupted fetch
    never leaves something that looks complete."""
    partial = destination + ".part"
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.time()
    with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
        total = response.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                pct = 100.0 * done / total
                sys.stderr.write(f"\r  {done >> 20} MiB of {total >> 20} MiB ({pct:.1f}%)")
            else:
                sys.stderr.write(f"\r  {done >> 20} MiB")
            sys.stderr.flush()
    sys.stderr.write(f"\r  {done >> 20} MiB in {time.time() - started:.1f}s\n")
    shutil.move(partial, destination)
    return destination


def _member_digests(dataset_id, spec):
    """Digests of the extracted members the loaders actually read. An
    archive can be repacked without its contents changing, and this is what
    tells the two cases apart."""
    members = []
    archive = spec["archive"]
    if archive["format"] == "directory":
        members = [spec["split"]["train_member"], spec["split"]["test_member"]]
    else:
        members = [archive.get("member")] + list(archive.get("extra_members", []))
    out = {}
    for member in [m for m in members if m]:
        try:
            raw = loaders._member_bytes(dataset_id, member)
        except (KeyError, OSError, ValueError) as exc:
            out[member] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out[member] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    return out


def _write_lock(lock):
    with open(loaders.LOCK_PATH, "w") as handle:
        json.dump(lock, handle, indent=2, sort_keys=False)
        handle.write("\n")


def pin(dataset_id, spec, path):
    digest, size = _sha256_and_size(path) if os.path.isfile(path) else (None, None)
    lock = loaders.lock()
    previous = lock["pins"].get(dataset_id)
    if previous and previous.get("archive_sha256") not in (None, digest):
        lock.setdefault("superseded", {}).setdefault(dataset_id, []).append(previous)
        print(
            f"  the previous pin for {dataset_id} did not match and was moved "
            "to `superseded`. Results taken before today used the old bytes."
        )
    lock["pins"][dataset_id] = {
        "url": spec.get("url"),
        "version": spec.get("version"),
        "archive_sha256": digest,
        "archive_bytes": size,
        "members": _member_digests(dataset_id, spec),
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_by": os.environ.get("MOJOBOOST_BENCH_PINNER") or platform.node(),
    }
    _write_lock(lock)
    print(f"  pinned {dataset_id}: sha256 {digest}, {size} bytes")


def fetch_one(dataset_id, spec, do_pin, force):
    path = loaders.archive_path(dataset_id)
    manual = spec["acquisition"] == "manual"

    if manual:
        target = os.path.join(loaders.cache_dir(), spec["archive"]["member"])
        if not os.path.exists(target):
            print(
                f"  {dataset_id} is acquired manually. Download it from "
                f"{spec['url']}, put it at {target}, then run this again with "
                "--pin.",
            )
            return False
        if do_pin:
            pin(dataset_id, spec, target)
        return True

    if os.path.exists(path) and not force:
        print(f"  {dataset_id} already at {path}")
    else:
        print(f"  {dataset_id} from {spec['url']}")
        _download(spec["url"], path)

    if do_pin:
        pin(dataset_id, spec, path)
        return True

    ok, reason = loaders.check_pinned(dataset_id)
    if not ok:
        print(f"  NOT VERIFIED: {reason}", file=sys.stderr)
        return False
    print(f"  verified {dataset_id} against checksums.lock.json")
    return True


def main(argv=None):
    registry = loaders.sources()["datasets"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="dataset ids from sources.json")
    parser.add_argument("--list", action="store_true", help="show the registry and exit")
    parser.add_argument("--tier", choices=("standard", "large"), help="fetch a whole tier")
    parser.add_argument("--pin", action="store_true", help="record the observed digest")
    parser.add_argument("--force", action="store_true", help="re-download an existing file")
    args = parser.parse_args(argv)

    if args.list:
        pins = loaders.lock().get("pins", {})
        width = max(len(name) for name in registry)
        for name, spec in sorted(registry.items()):
            state = "pinned" if name in pins else "unpinned"
            print(
                f"{name:<{width}}  {spec['task']:<11} {spec['tier']:<8} "
                f"{spec['acquisition']:<8} {state}"
            )
        return 0

    wanted = list(args.datasets)
    if args.tier:
        wanted += [n for n, s in registry.items() if s["tier"] == args.tier]
    if not wanted:
        parser.error("name at least one dataset, or pass --tier or --list")

    unknown = [name for name in wanted if name not in registry]
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")

    failures = []
    for name in dict.fromkeys(wanted):
        print(f"{name}:")
        try:
            if not fetch_one(name, registry[name], args.pin, args.force):
                failures.append(name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(f"\nnot ready: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
