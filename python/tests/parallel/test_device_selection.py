"""Focused tests for the Python seam over Mojo's device policy.

Policy, capability detection, crossover rules, and memory estimates belong to
``src/mojoboost/device_policy.mojo``.  These tests deliberately do not rebuild
that policy in Python.  Synthetic native responses exercise parsing and error
translation; a small CPU request proves the compiled binding is connected.
"""

import json

import pytest

from mojoboost import device_selection as ds


class _FakePolicy:
    contract = ds.CONTRACT_FULL

    def __init__(self, fields):
        self.fields = fields
        self.calls = []

    def decide(self, requested, workload):
        self.calls.append((requested, workload))
        return dict(self.fields)


def _fields(**overrides):
    fields = {
        "requested": "auto",
        "selected": "cpu",
        "blocked": "false",
        "decision": "no-validated-rule",
        "message": "no validated GPU crossover rule matched",
        "validated": "false",
        "policy_version": "1",
        "evidence_id": "none",
        "memory_estimate_complete": "true",
        "gpu_available": "true",
        "api": "metal",
        "_blocks": [],
        "_warnings": [],
        "_transfers": [],
    }
    fields.update(overrides)
    return fields


def test_workload_tracks_shape_outputs_and_declared_features():
    workload = ds.Workload(
        100,
        8,
        objective="multiclass",
        objective_code=-1,
        n_classes=4,
        max_bin=63,
        sparse=True,
        categorical=True,
        has_missing=True,
        has_eval_set=True,
    )
    assert workload.cells == 800
    assert workload.n_outputs == 4
    assert workload.to_dict() == {
        "n_rows": 100,
        "n_features": 8,
        "cells": 800,
        "objective": "multiclass",
        "objective_code": -1,
        "n_classes": 4,
        "n_outputs": 4,
        "max_bin": 63,
        "sparse": True,
        "categorical": True,
        "has_missing": True,
        "has_eval_set": True,
    }


@pytest.mark.parametrize("rows, features", [(0, 2), (2, 0)])
def test_workload_rejects_empty_dimensions(rows, features):
    with pytest.raises(ValueError, match="at least one row and one feature"):
        ds.Workload(rows, features)


class _FakeFrame:
    def __init__(self, shape, sparse=False):
        self.shape = shape
        if sparse:
            self.nnz = 1
            self.format = "csr"

    def tocsr(self):
        return self


def test_workload_extraction_is_container_agnostic():
    workload = ds.Workload.from_data(
        _FakeFrame((6, 2), sparse=True), [0, 1, 2, 2, 1, 0]
    )
    assert (workload.n_rows, workload.n_features) == (6, 2)
    assert workload.sparse is True
    assert workload.n_classes == 3
    assert workload.n_outputs == 3


def test_list_input_and_keyword_overrides_are_supported(monkeypatch):
    policy = _FakePolicy(_fields())
    monkeypatch.setattr(ds, "_policy", lambda: policy)
    report = ds.explain_device_choice(
        [[1.0, 2.0], [3.0, 4.0]],
        device="AUTO",
        objective="poisson",
        objective_code=5,
        max_bin=31,
    )
    assert report.requested == "auto"
    assert report.workload.objective == "poisson"
    assert report.workload.objective_code == 5
    assert report.workload.max_bin == 31
    assert policy.calls == [("auto", report.workload)]


def test_unknown_device_is_rejected_before_native_call(monkeypatch):
    policy = _FakePolicy(_fields())
    monkeypatch.setattr(ds, "_policy", lambda: policy)
    with pytest.raises(ValueError, match="unknown device"):
        ds.select_device("tpu", ds.Workload(10, 2))
    assert policy.calls == []


def test_native_report_is_json_serializable_and_explainable(monkeypatch):
    fields = _fields(
        memory_device_bytes="1024",
        memory_upper_bound_bytes="2048",
        session_context_open="false",
        session_kernels_ready="false",
        _warnings=[ds.Reason("unvalidated-path", "GPU path is unvalidated")],
        _transfers=[
            ds.TransferRoute(
                "bins", "copy_staged", "copy_staged", "eligible", "none", "copy"
            )
        ],
    )
    monkeypatch.setattr(ds, "_policy", lambda: _FakePolicy(fields))
    report = ds.select_device("auto", ds.Workload(1_000, 10, max_bin=63))
    payload = json.loads(report.to_json())
    assert report.resolved == "cpu"
    assert report.complete is True
    assert payload["native"]["api"] == "metal"
    assert payload["warnings"][0]["code"] == "unvalidated-path"
    assert payload["transfer_routes"][0]["role"] == "bins"
    assert "resolved to CPU" in report.explanation


def test_explicit_gpu_refusal_never_silently_falls_back(monkeypatch):
    refusal = _fields(
        requested="gpu",
        selected="none",
        blocked="true",
        decision="gpu-refused",
        message="sparse input is not supported by GPU training",
        _blocks=[ds.Reason("unsupported-feature", "sparse input")],
    )
    monkeypatch.setattr(ds, "_policy", lambda: _FakePolicy(refusal))
    workload = ds.Workload(100, 8, sparse=True)

    report = ds.explain_device_choice(workload, device="gpu")
    assert report.would_raise is True
    assert report.resolved is None
    assert report.reasons[0].code == "unsupported-feature"

    with pytest.raises(ds.DeviceUnavailableError, match="sparse input") as exc:
        ds.select_device("gpu", workload)
    assert exc.value.report.resolved is None


def test_wire_parser_preserves_repeated_native_records():
    parsed = ds._parse_decision(
        "selected=cpu\n"
        "blocked=false\n"
        "block=3:unsupported-feature:sparse: detail\n"
        "warning=7:unvalidated-path:no benchmark evidence\n"
        "transfer=bins:copy_staged:copy_staged:eligible:none:copy\n"
    )
    assert parsed["selected"] == "cpu"
    assert parsed["_blocks"] == [
        ds.Reason("unsupported-feature", "sparse: detail")
    ]
    assert parsed["_warnings"][0].code == "unvalidated-path"
    assert parsed["_transfers"][0].honored is True


def test_compiled_binding_exposes_full_native_contract():
    """A CPU request is hardware-independent and must cross the full seam."""
    assert ds.native_contract() == ds.CONTRACT_FULL
    report = ds.select_device("cpu", ds.Workload(2, 2, max_bin=63))
    assert report.requested == "cpu"
    assert report.resolved == "cpu"
    assert report.contract == ds.CONTRACT_FULL
    assert report.would_raise is False
