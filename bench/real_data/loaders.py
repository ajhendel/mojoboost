"""Reading the pinned datasets into arrays.

One rule runs through this module: whatever transformation the data needs,
it happens once, here, and both engines receive the result. Category codes,
missing markers, label mapping, and the train and test split are all
decided before either library sees a row. Letting each library do its own
encoding would produce a comparison of two encoders wearing the costume of
a comparison of two trainers.

Every loader checks the shape it got against `expected_shape` in
sources.json and fails loudly on a mismatch. A URL that quietly starts
serving something else is exactly the failure a benchmark suite is worst
at noticing.

Splits are deterministic and are described in the record. A hash split
assigns a row from its index alone, so it does not depend on row order or
on how many rows were read; a file split uses the split the dataset ships
with; a query split keeps every query whole on one side.
"""

import bz2
import gzip
import hashlib
import io
import json
import os
import zipfile

import numpy as np

import generators

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES_PATH = os.path.join(HERE, "sources.json")
LOCK_PATH = os.path.join(HERE, "checksums.lock.json")


class DataUnavailable(RuntimeError):
    """The dataset is not on disk, or is not pinned. The runner turns this
    into a recorded fallback to the generator, never into a silent one."""


def sources():
    with open(SOURCES_PATH) as handle:
        return json.load(handle)


def lock():
    with open(LOCK_PATH) as handle:
        return json.load(handle)


def cache_dir():
    env = sources()["cache_env"]
    override = os.environ.get(env)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(HERE, "data")


def archive_path(dataset_id):
    spec = sources()["datasets"][dataset_id]
    if spec["acquisition"] == "manual":
        return os.path.join(cache_dir(), spec["archive"]["member"])
    return os.path.join(cache_dir(), f"{dataset_id}{_suffix(spec)}")


def _suffix(spec):
    return {"zip": ".zip", "bz2": ".bz2", "gzip": ".gz", "directory": ""}[
        spec["archive"]["format"]
    ]


def check_pinned(dataset_id):
    """(pinned, reason). A dataset is pinned when the lock holds a digest
    for it and the file on disk matches that digest."""
    pins = lock().get("pins", {})
    if dataset_id not in pins:
        return False, (
            f"{dataset_id} has no entry in checksums.lock.json; run "
            f"`python bench/real_data/fetch.py --pin {dataset_id}` once and "
            "commit the lock"
        )
    path = archive_path(dataset_id)
    if not os.path.exists(path):
        return False, f"{path} is not on disk"
    observed = _sha256_file(path)
    expected = pins[dataset_id].get("archive_sha256")
    if observed != expected:
        return False, (
            f"{path} hashes to {observed} but the lock pins {expected}; the "
            "dataset changed underneath the pin"
        )
    return True, None


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# -- archive access -------------------------------------------------------


def _member_bytes(dataset_id, member=None):
    """The raw bytes of one member of a dataset's archive.

    `member` supports a single level of nesting with `outer.zip!inner/path`,
    which is what the bank marketing archive needs.
    """
    spec = sources()["datasets"][dataset_id]
    fmt = spec["archive"]["format"]
    path = archive_path(dataset_id)
    member = member or spec["archive"].get("member")

    if fmt == "bz2":
        with bz2.open(path, "rb") as handle:
            return handle.read()
    if fmt == "directory":
        with open(os.path.join(cache_dir(), member), "rb") as handle:
            return handle.read()
    if fmt != "zip":
        raise ValueError(f"unsupported archive format {fmt!r}")

    outer, _, inner = member.partition("!")
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(outer)
    if inner:
        with zipfile.ZipFile(io.BytesIO(raw)) as nested:
            raw = nested.read(inner)
    if spec["archive"].get("inner") == "gzip" or outer.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw


# -- parsers --------------------------------------------------------------


def _read_table(raw, delimiter, header):
    """Rows of string fields. pandas when it is installed, because these
    files run to hundreds of megabytes; the standard library otherwise."""
    try:
        import pandas as pd

        frame = pd.read_csv(
            io.BytesIO(raw),
            sep=delimiter,
            header=0 if header else None,
            skipinitialspace=True,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
        return list(frame.columns) if header else None, frame.to_numpy(dtype=object)
    except ImportError:
        import csv

        reader = csv.reader(io.StringIO(raw.decode("utf-8")), delimiter=delimiter)
        rows = [row for row in reader if row]
        names = rows.pop(0) if header else None
        width = max(len(row) for row in rows)
        table = np.empty((len(rows), width), dtype=object)
        for i, row in enumerate(rows):
            table[i, : len(row)] = [field.strip() for field in row]
        return names, table


def _encode_categories(column, missing_token=None):
    """Integer codes from sorted distinct values, missing as NaN.

    Sorting makes the encoding depend only on the set of values, so the
    same file encodes the same way on any machine and in any pandas
    version, and the mapping can be digested into the record.
    """
    values = np.asarray(column, dtype=object)
    present = np.array(
        [v is not None and v != "" and v != missing_token for v in values]
    )
    vocab = sorted({str(v) for v in values[present]})
    index = {name: float(i) for i, name in enumerate(vocab)}
    codes = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if present[i]:
            codes[i] = index[str(value)]
    return codes, vocab


def _vocab_digest(vocabs):
    payload = json.dumps(vocabs, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_libsvm(raw, n_features, label_map, one_based):
    """A LIBSVM file into a CSC matrix and a label vector.

    Parsed in one pass into index arrays, then handed to scipy, rather than
    assembling a matrix incrementally: on a 1.3 million column file the
    incremental version is the slowest part of the whole harness.
    """
    from scipy import sparse

    labels, rows, cols, vals = [], [], [], []
    for r, line in enumerate(raw.decode("utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        labels.append(label_map.get(head, head))
        for token in rest.split():
            if ":" not in token:
                continue
            index, _, value = token.partition(":")
            j = int(index) - (1 if one_based else 0)
            rows.append(r)
            cols.append(j)
            vals.append(float(value))
    n_rows = len(labels)
    matrix = sparse.csr_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(n_rows, n_features),
        dtype=np.float64,
    )
    matrix.sum_duplicates()
    return matrix.tocsc(), np.asarray(labels, dtype=np.float64)


def _parse_letor(raw, n_features):
    """A LETOR or MSLR file into (X, y, group). Rows are already grouped by
    query in these files, which is checked rather than assumed."""
    labels, qids, feats = [], [], []
    for line in raw.decode("utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        labels.append(float(parts[0]))
        qids.append(parts[1].split(":", 1)[1])
        row = np.zeros(n_features)
        for token in parts[2:]:
            index, _, value = token.partition(":")
            row[int(index) - 1] = float(value)
        feats.append(row)
    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    sizes, seen = [], None
    for qid in qids:
        if qid != seen:
            if seen is not None and qid in set(qids[: len(sizes)]):
                raise ValueError("query ids are interleaved; rows must be grouped")
            sizes.append(0)
            seen = qid
        sizes[-1] += 1
    return x, y, np.asarray(sizes, dtype=np.int64)


# -- per-dataset loaders --------------------------------------------------


def _check_shape(dataset_id, rows, columns):
    expected = sources()["datasets"][dataset_id]["expected_shape"]
    for name, got, want in (("rows", rows, expected["rows"]), ("columns", columns, expected["columns"])):
        if want is not None and got != want:
            raise DataUnavailable(
                f"{dataset_id} has {got} {name} but sources.json expects "
                f"{want}. The file is not what the registry describes; fix "
                "one of the two rather than benchmarking it."
            )


def _split_prefix(x, y, train_rows):
    return (
        {"X": x[:train_rows], "y": y[:train_rows]},
        {"X": x[train_rows:], "y": y[train_rows:]},
    )


def _split_hash(x, y, fraction, seed, sparse_matrix=False, extra=None):
    mask = generators._hash_split(len(y), fraction, seed)
    if sparse_matrix:
        csr = x.tocsr()
        train_x, test_x = csr[mask].tocsc(), csr[~mask].tocsc()
    else:
        train_x, test_x = x[mask], x[~mask]
    train = {"X": train_x, "y": y[mask]}
    test = {"X": test_x, "y": y[~mask]}
    for part in (train, test):
        part.update(extra or {})
    return train, test


def load_year_prediction_msd():
    raw = _member_bytes("year_prediction_msd")
    _, table = _read_table(raw, ",", header=False)
    data = table.astype(np.float64)
    _check_shape("year_prediction_msd", data.shape[0], data.shape[1])
    y, x = data[:, 0].copy(), np.ascontiguousarray(data[:, 1:])
    split = sources()["datasets"]["year_prediction_msd"]["split"]
    train, test = _split_prefix(x, y, split["train_rows"])
    return train, test, {"task": "regression", "split": split}


def load_covertype():
    raw = _member_bytes("covertype")
    _, table = _read_table(raw, ",", header=False)
    data = table.astype(np.float64)
    _check_shape("covertype", data.shape[0], data.shape[1])
    y = data[:, 54] - 1.0
    x = np.ascontiguousarray(data[:, :54])
    split = sources()["datasets"]["covertype"]["split"]
    train, test = _split_hash(
        x, y, split["train_fraction"], split["seed"], extra={"n_classes": 7}
    )
    return train, test, {"task": "multiclass", "split": split, "n_classes": 7}


def load_bank_marketing():
    raw = _member_bytes("bank_marketing")
    names, table = _read_table(raw, ";", header=True)
    _check_shape("bank_marketing", table.shape[0], table.shape[1])
    label_col = names.index("y")
    y = np.array(
        [1.0 if str(v).strip('"') == "yes" else 0.0 for v in table[:, label_col]]
    )
    keep = [i for i in range(table.shape[1]) if i != label_col]
    columns, cat_indices, vocabs = [], [], {}
    for out_index, col in enumerate(keep):
        raw_col = np.array([str(v).strip('"') for v in table[:, col]], dtype=object)
        try:
            columns.append(raw_col.astype(np.float64))
        except ValueError:
            codes, vocab = _encode_categories(raw_col)
            columns.append(codes)
            cat_indices.append(out_index)
            vocabs[names[col]] = vocab
    x = np.ascontiguousarray(np.column_stack(columns))
    split = sources()["datasets"]["bank_marketing"]["split"]
    train, test = _split_hash(
        x,
        y,
        split["train_fraction"],
        split["seed"],
        extra={"categorical_feature": cat_indices},
    )
    meta = {
        "task": "binary",
        "split": split,
        "categorical_feature": cat_indices,
        "category_vocab_sha256": _vocab_digest(vocabs),
        "positive_rate": float(y.mean()),
    }
    return train, test, meta


def load_adult():
    spec = sources()["datasets"]["adult"]
    fmt = spec["format"]
    parts = []
    for member, skip in (
        (spec["split"]["train_member"], 0),
        (spec["split"]["test_member"], fmt["test_skip_rows"]),
    ):
        _, table = _read_table(_member_bytes("adult", member), ",", header=False)
        parts.append(table[skip:])
    n_rows = sum(part.shape[0] for part in parts)
    _check_shape("adult", n_rows, parts[0].shape[1])

    table = np.vstack(parts)
    label_raw = np.array(
        [str(v).strip().rstrip(fmt["test_label_suffix"]) for v in table[:, fmt["label_column"]]]
    )
    y = (label_raw == fmt["positive_label"]).astype(np.float64)
    keep = [i for i in range(table.shape[1]) if i != fmt["label_column"]]
    columns, cat_indices, vocabs = [], [], {}
    for out_index, col in enumerate(keep):
        raw_col = np.array([str(v).strip() for v in table[:, col]], dtype=object)
        if col in fmt["categorical_columns"]:
            codes, vocab = _encode_categories(raw_col, fmt["missing_token"])
            columns.append(codes)
            cat_indices.append(out_index)
            vocabs[f"column_{col}"] = vocab
        else:
            numeric = np.array(
                [np.nan if v == fmt["missing_token"] else float(v) for v in raw_col]
            )
            columns.append(numeric)
    x = np.ascontiguousarray(np.column_stack(columns))
    cut = parts[0].shape[0]
    train = {"X": x[:cut], "y": y[:cut], "categorical_feature": cat_indices}
    test = {"X": x[cut:], "y": y[cut:], "categorical_feature": cat_indices}
    meta = {
        "task": "binary",
        "split": spec["split"],
        "categorical_feature": cat_indices,
        "category_vocab_sha256": _vocab_digest(vocabs),
        "missing_fraction": float(np.isnan(x).mean()),
        "positive_rate": float(y.mean()),
    }
    return train, test, meta


def _load_libsvm(dataset_id):
    spec = sources()["datasets"][dataset_id]
    fmt = spec["format"]
    x, y = _parse_libsvm(
        _member_bytes(dataset_id),
        spec["expected_shape"]["columns"],
        {k: float(v) for k, v in fmt["label_map"].items()},
        fmt["one_based_indices"],
    )
    _check_shape(dataset_id, x.shape[0], x.shape[1])
    split = spec["split"]
    train, test = _split_hash(
        x, y, split["train_fraction"], split["seed"], sparse_matrix=True,
        extra={"sparse": True},
    )
    meta = {
        "task": "binary",
        "split": split,
        "nnz": int(x.nnz),
        "density": float(x.nnz) / float(x.shape[0] * x.shape[1]),
        "positive_rate": float(y.mean()),
    }
    return train, test, meta


def load_rcv1_train_binary():
    return _load_libsvm("rcv1_train_binary")


def load_news20_binary():
    return _load_libsvm("news20_binary")


def load_mslr_web10k():
    spec = sources()["datasets"]["mslr_web10k"]
    n_features = spec["format"]["n_features"]
    out = []
    for member in (spec["split"]["train_member"], spec["split"]["test_member"]):
        x, y, group = _parse_letor(_member_bytes("mslr_web10k", member), n_features)
        out.append({"X": x, "y": y, "group": group})
    _check_shape("mslr_web10k", out[0]["X"].shape[0], out[0]["X"].shape[1])
    return out[0], out[1], {"task": "ranking", "split": spec["split"]}


def load_higgs():
    raw = _member_bytes("higgs")
    _, table = _read_table(raw, ",", header=False)
    data = table.astype(np.float64)
    _check_shape("higgs", data.shape[0], data.shape[1])
    y, x = data[:, 0].copy(), np.ascontiguousarray(data[:, 1:])
    split = sources()["datasets"]["higgs"]["split"]
    train, test = _split_prefix(x, y, split["train_rows"])
    return train, test, {"task": "binary", "split": split}


LOADERS = {
    "year_prediction_msd": load_year_prediction_msd,
    "bank_marketing": load_bank_marketing,
    "covertype": load_covertype,
    "adult": load_adult,
    "rcv1_train_binary": load_rcv1_train_binary,
    "mslr_web10k": load_mslr_web10k,
    "higgs": load_higgs,
    "news20_binary": load_news20_binary,
}


def load(dataset_id, allow_unpinned=False):
    """(train, test, meta) for a real dataset.

    Refuses to load an unpinned dataset unless told otherwise, and when
    told otherwise stamps `pinned: false` on the metadata so the result
    record and verify.py both know.
    """
    if dataset_id not in LOADERS:
        raise DataUnavailable(f"no loader for {dataset_id!r}")
    pinned, reason = check_pinned(dataset_id)
    if not pinned and not allow_unpinned:
        raise DataUnavailable(reason)
    train, test, meta = LOADERS[dataset_id]()
    meta.update(
        {
            "dataset": dataset_id,
            "data_kind": "real",
            "pinned": bool(pinned),
            "pin_reason": reason,
            "source": {
                k: sources()["datasets"][dataset_id].get(k)
                for k in ("url", "version", "license", "citation")
            },
        }
    )
    return train, test, meta
