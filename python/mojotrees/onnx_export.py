"""Turn an export plan written by `src/mojotrees/onnx_export.mojo` into an
ONNX `ModelProto`.

This module contains **no arithmetic**. Every threshold, every leaf weight,
every NaN direction, and the whole refusal list are decided in Mojo, where
they are under test with no dependencies; see `docs/design/MODEL_EXPORT.md`.
What is left here is transcription: read the plan's token stream, hand the
arrays to `onnx.helper`, and build the graph.

The split exists because the part that can be silently wrong and the part
that needs a third-party library are different parts, and only one of them
should be hard to test.

`onnx` is an optional dependency. It is not in `pixi.toml` and not in
`python/pyproject.toml`, because nothing else in mojotrees needs it. Its
absence raises with the install line rather than degrading to something that
looks like an export.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

__all__ = [
    "OnnxPlan",
    "PLAN_MAGIC",
    "PLAN_VERSION",
    "ML_OPSET",
    "read_plan",
    "plan_to_model_proto",
    "convert_plan_file",
]

PLAN_MAGIC = "mojotrees-onnx-plan"
PLAN_VERSION = 1
ML_OPSET = 3
"""Must agree with `ONNX_PLAN_MAGIC`, `ONNX_PLAN_VERSION`, and
`ONNX_ML_OPSET` in `src/mojotrees/onnx_export.mojo`. A plan that declares a
version this module does not know is refused, not guessed at."""

# `nodes_modes`, as integers in the plan and as the strings the operator
# wants. The plan carries integers so the token stream needs no escaping.
_MODES = {0: "BRANCH_LEQ", 1: "LEAF"}

# `post_transform`, plus the one value that is not an ONNX post-transform:
# `exp`, which becomes an explicit node after the ensemble.
_POST_NONE, _POST_LOGISTIC, _POST_SOFTMAX, _POST_EXP = 0, 1, 2, 3
_POST_NAMES = {
    _POST_NONE: "NONE",
    _POST_LOGISTIC: "LOGISTIC",
    _POST_SOFTMAX: "SOFTMAX",
    _POST_EXP: "NONE",  # applied by an appended Exp node instead
}

_INT_FIELDS = (
    "nodes_treeids",
    "nodes_nodeids",
    "nodes_featureids",
    "nodes_modes",
    "nodes_truenodeids",
    "nodes_falsenodeids",
    "nodes_missing_value_tracks_true",
    "target_treeids",
    "target_nodeids",
    "target_ids",
)
_FLOAT_FIELDS = ("base_values", "nodes_values", "target_weights")


def _bits_to_float(token: str) -> float:
    """A float from its raw IEEE-754 bits in decimal, the convention the
    whole repository writes floats in. Parsing a decimal literal instead
    would let a threshold move a row by a whole leaf."""
    bits = int(token)
    if not 0 <= bits <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"float bit pattern out of range: {token}")
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


class OnnxPlan:
    """The `ai.onnx.ml.TreeEnsembleRegressor` attribute arrays, as read.

    Attribute names are the operator's, so this can be checked against the
    ONNX operator reference without a translation table.
    """

    __slots__ = (
        ("n_features", "n_targets", "post_transform", "opset")
        + _FLOAT_FIELDS
        + _INT_FIELDS
    )

    def __init__(self, **fields: object) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])

    def n_nodes(self) -> int:
        return len(self.nodes_treeids)

    def n_leaves(self) -> int:
        return len(self.target_treeids)


class _Tokens:
    def __init__(self, text: str) -> None:
        self._toks = text.split()
        self._pos = 0

    def next(self) -> str:
        if self._pos >= len(self._toks):
            raise ValueError("unexpected end of export plan")
        tok = self._toks[self._pos]
        self._pos += 1
        return tok

    def expect(self, name: str) -> None:
        got = self.next()
        if got != name:
            raise ValueError(f"expected '{name}' in export plan, found {got!r}")

    def scalar_int(self, name: str) -> int:
        self.expect(name)
        return int(self.next())

    def int_list(self, name: str) -> list[int]:
        n = self.scalar_int(name)
        if n < 0:
            raise ValueError(f"negative length for {name}")
        return [int(self.next()) for _ in range(n)]

    def float_list(self, name: str) -> list[float]:
        n = self.scalar_int(name)
        if n < 0:
            raise ValueError(f"negative length for {name}")
        return [_bits_to_float(self.next()) for _ in range(n)]


def read_plan(text: str) -> OnnxPlan:
    """Parse the token stream `save_onnx_plan` writes.

    Sections are read in the fixed order the writer emits them, and every
    array carries its own length, so a truncated file fails here rather than
    producing a short attribute the ONNX checker would accept.
    """
    t = _Tokens(text)
    t.expect(PLAN_MAGIC)
    version = int(t.next())
    if version != PLAN_VERSION:
        raise ValueError(
            f"export plan version {version} is not the {PLAN_VERSION} this "
            "build reads"
        )
    fields: dict[str, object] = {}
    fields["opset"] = t.scalar_int("opset")
    fields["n_features"] = t.scalar_int("n_features")
    fields["n_targets"] = t.scalar_int("n_targets")
    fields["post_transform"] = t.scalar_int("post_transform")
    fields["base_values"] = t.float_list("base_values")
    for name in (
        "nodes_treeids",
        "nodes_nodeids",
        "nodes_featureids",
        "nodes_modes",
    ):
        fields[name] = t.int_list(name)
    fields["nodes_values"] = t.float_list("nodes_values")
    for name in (
        "nodes_truenodeids",
        "nodes_falsenodeids",
        "nodes_missing_value_tracks_true",
        "target_treeids",
        "target_nodeids",
        "target_ids",
    ):
        fields[name] = t.int_list(name)
    fields["target_weights"] = t.float_list("target_weights")

    plan = OnnxPlan(**fields)
    _check_plan(plan)
    return plan


def _check_plan(plan: OnnxPlan) -> None:
    n_nodes = plan.n_nodes()
    for name in (
        "nodes_nodeids",
        "nodes_featureids",
        "nodes_modes",
        "nodes_values",
        "nodes_truenodeids",
        "nodes_falsenodeids",
        "nodes_missing_value_tracks_true",
    ):
        if len(getattr(plan, name)) != n_nodes:
            raise ValueError(f"{name} does not have one entry per node")
    n_leaves = plan.n_leaves()
    for name in ("target_nodeids", "target_ids", "target_weights"):
        if len(getattr(plan, name)) != n_leaves:
            raise ValueError(f"{name} does not have one entry per leaf")
    if len(plan.base_values) != plan.n_targets:
        raise ValueError("base_values does not have one entry per target")
    if plan.post_transform not in _POST_NAMES:
        raise ValueError(f"unknown post_transform {plan.post_transform}")
    for mode in plan.nodes_modes:
        if mode not in _MODES:
            raise ValueError(f"unknown node mode {mode}")


def _require_onnx():
    try:
        import onnx  # noqa: PLC0415
        from onnx import helper, numpy_helper  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ImportError(
            "ONNX export needs the 'onnx' package, which is an optional "
            "dependency of mojotrees: pip install onnx"
        ) from exc
    return onnx, helper, numpy_helper


def _double_tensor(numpy_helper, name: str, values: Sequence[float]):
    import numpy as np  # onnx depends on numpy, so this import is safe here

    return numpy_helper.from_array(
        np.asarray(values, dtype=np.float64), name=name
    )


def plan_to_model_proto(plan: OnnxPlan, *, name: str = "mojotrees"):
    """Build a `ModelProto` from a plan.

    Doubles throughout: thresholds, leaf weights, and base values go into the
    `*_as_tensor` attributes that `ai.onnx.ml` opset 3 added, so nothing is
    narrowed on the way in. The operator's output is `tensor(float)` in every
    version, which is a property of the operator and is recorded in
    `docs/design/MODEL_EXPORT.md` section 4, not something this function can
    change.
    """
    onnx, helper, numpy_helper = _require_onnx()
    if plan.opset != ML_OPSET:
        raise ValueError(
            f"plan targets ai.onnx.ml opset {plan.opset}, this converter "
            f"builds opset {ML_OPSET}"
        )

    x = helper.make_tensor_value_info(
        "X", onnx.TensorProto.DOUBLE, [None, plan.n_features]
    )
    ensemble_out = "raw" if plan.post_transform == _POST_EXP else "Y"

    node = helper.make_node(
        "TreeEnsembleRegressor",
        inputs=["X"],
        outputs=[ensemble_out],
        domain="ai.onnx.ml",
        n_targets=plan.n_targets,
        aggregate_function="SUM",
        post_transform=_POST_NAMES[plan.post_transform],
        base_values_as_tensor=_double_tensor(
            numpy_helper, "base_values", plan.base_values
        ),
        nodes_treeids=plan.nodes_treeids,
        nodes_nodeids=plan.nodes_nodeids,
        nodes_featureids=plan.nodes_featureids,
        nodes_modes=[_MODES[m] for m in plan.nodes_modes],
        nodes_values_as_tensor=_double_tensor(
            numpy_helper, "nodes_values", plan.nodes_values
        ),
        nodes_truenodeids=plan.nodes_truenodeids,
        nodes_falsenodeids=plan.nodes_falsenodeids,
        nodes_missing_value_tracks_true=(
            plan.nodes_missing_value_tracks_true
        ),
        target_treeids=plan.target_treeids,
        target_nodeids=plan.target_nodeids,
        target_ids=plan.target_ids,
        target_weights_as_tensor=_double_tensor(
            numpy_helper, "target_weights", plan.target_weights
        ),
    )
    nodes = [node]
    if plan.post_transform == _POST_EXP:
        # poisson, gamma, tweedie. ONNX has no `exp` post_transform, so the
        # link is an explicit node rather than a dropped transform.
        nodes.append(helper.make_node("Exp", inputs=["raw"], outputs=["Y"]))

    y = helper.make_tensor_value_info(
        "Y", onnx.TensorProto.FLOAT, [None, plan.n_targets]
    )
    graph = helper.make_graph(nodes, name, [x], [y])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("ai.onnx.ml", ML_OPSET),
            helper.make_opsetid("", 13),
        ],
    )


def convert_plan_file(plan_path: str | Path, onnx_path: str | Path) -> None:
    """Read a plan file and write the `ModelProto` beside it, checked."""
    onnx, _, _ = _require_onnx()
    plan = read_plan(Path(plan_path).read_text())
    model = plan_to_model_proto(plan)
    onnx.checker.check_model(model)
    Path(onnx_path).write_bytes(model.SerializeToString())
