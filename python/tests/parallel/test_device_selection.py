"""Tests for the policy in python/mojoboost/device_selection.py.

Every test injects a `Capabilities` describing a machine, so the policy is
exercised on hardware this repository does not own and the results do not
depend on whether the machine running pytest has an accelerator. The one
test that touches real detection only checks that it answers at all.

The module is loaded directly from its file when the package cannot be
imported, because `mojoboost/__init__.py` imports the compiled extension
and this layer needs nothing from it.
"""

import importlib.util
import json
import os
import sys

import pytest

_PYTHON_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

try:
    from mojoboost import device_selection as ds
except Exception:
    _path = os.path.join(_PYTHON_DIR, "mojoboost", "device_selection.py")
    _spec = importlib.util.spec_from_file_location(
        "mojoboost_device_selection", _path
    )
    ds = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = ds
    _spec.loader.exec_module(ds)


@pytest.fixture
def gpu_caps():
    """A machine with a working Metal accelerator and 16 GiB unified."""
    return ds.Capabilities(
        gpu_available=True,
        backend="metal",
        chip="Apple M4",
        device_memory_bytes=16 * 1024**3,
        unified_memory=True,
        auto_min_cells=None,
    )


@pytest.fixture
def cpu_caps():
    """A machine with no accelerator at all."""
    return ds.Capabilities(gpu_available=False, backend=None, chip=None)


@pytest.fixture
def workload():
    return ds.Workload(n_rows=1_000_000, n_features=100)


def _rule(**kwargs):
    """A crossover rule with the evidence field filled in, since a rule
    without evidence is refused."""
    fields = dict(
        name="test-rule",
        evidence="synthetic rule, this test file only",
        measured_on="nothing",
    )
    fields.update(kwargs)
    return ds.CrossoverRule(**fields)


# --------------------------------------------------------------------
# The shipped table


def test_shipped_rule_table_is_empty():
    """No measurement in this repository establishes a crossover, so the
    table ships empty. A rule appearing here without a benchmark behind
    it is the failure this test exists to catch."""
    assert ds.CROSSOVER_RULES == ()
    assert isinstance(ds.RULES_VERSION, int)


def test_rule_requires_evidence():
    with pytest.raises(ValueError):
        ds.CrossoverRule(name="unfounded", evidence="")


# --------------------------------------------------------------------
# Explicit devices


def test_cpu_is_chosen_even_with_a_gpu_present(gpu_caps, workload):
    report = ds.select_device("cpu", workload, capabilities=gpu_caps)
    assert report.resolved == "cpu"
    assert report.reasons[0].code == ds.EXPLICIT_CPU


def test_explicit_gpu_runs_when_nothing_blocks(gpu_caps, workload):
    report = ds.select_device("gpu", workload, capabilities=gpu_caps)
    assert report.resolved == "gpu"
    assert report.reasons[0].code == ds.EXPLICIT_GPU
    # An explicit request is not a validated one.
    assert report.validated is False
    assert report.warnings


def test_explicit_gpu_raises_without_an_accelerator(cpu_caps, workload):
    with pytest.raises(ds.DeviceUnavailableError) as excinfo:
        ds.select_device("gpu", workload, capabilities=cpu_caps)
    assert "no accelerator" in str(excinfo.value)
    assert excinfo.value.report.resolved is None


def test_explicit_gpu_never_falls_back_to_cpu(cpu_caps, workload):
    """The point of the whole policy: a GPU request that cannot run must
    fail loudly rather than train on the CPU under a GPU label."""
    try:
        report = ds.select_device("gpu", workload, capabilities=cpu_caps)
    except ds.DeviceUnavailableError:
        return
    pytest.fail("device='gpu' silently resolved to %r" % report.resolved)


def test_disabled_by_env_is_reported_as_its_own_reason(workload):
    caps = ds.Capabilities(gpu_available=False, disabled_by_env=True)
    with pytest.raises(ds.DeviceUnavailableError) as excinfo:
        ds.select_device("gpu", workload, capabilities=caps)
    assert "MOJOBOOST_DISABLE_GPU" in str(excinfo.value)
    report = ds.explain_device_choice(
        workload, device="auto", capabilities=caps
    )
    assert report.resolved == "cpu"
    assert report.reasons[0].code == ds.GPU_DISABLED_ENV


def test_device_names_are_case_insensitive(gpu_caps, workload):
    for name in ("GPU", "Gpu", "AUTO", "Cpu"):
        report = ds.explain_device_choice(
            workload, device=name, capabilities=gpu_caps
        )
        assert report.requested == name.lower()


def test_unknown_device_name_raises_value_error(gpu_caps, workload):
    with pytest.raises(ValueError):
        ds.select_device("tpu", workload, capabilities=gpu_caps)


# --------------------------------------------------------------------
# Hard blocks, one per rule the estimators or kernels already enforce


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"sparse": True}, "sparse"),
        ({"custom_objective": True}, "custom objective"),
        ({"has_eval_set": True}, "validation metrics"),
        ({"objective": "lambdarank"}, "lambdarank"),
    ],
)
def test_cpu_only_features_block_the_gpu(gpu_caps, kwargs, fragment):
    workload = ds.Workload(n_rows=100_000, n_features=20, **kwargs)
    with pytest.raises(ds.DeviceUnavailableError) as excinfo:
        ds.select_device("gpu", workload, capabilities=gpu_caps)
    assert fragment in str(excinfo.value)

    auto = ds.explain_device_choice(
        workload, device="auto", capabilities=gpu_caps
    )
    assert auto.resolved == "cpu"
    assert auto.reasons[0].code == ds.UNSUPPORTED_FEATURE


def test_multiclass_blocks_only_when_the_build_lacks_it(gpu_caps):
    workload = ds.Workload(
        n_rows=100_000, n_features=20, objective="multiclass", n_classes=5
    )
    covered = ds.explain_device_choice(
        workload, device="gpu", capabilities=gpu_caps
    )
    assert covered.would_raise is False

    without = gpu_caps.replace(supports_multiclass=False)
    with pytest.raises(ds.DeviceUnavailableError) as excinfo:
        ds.select_device("gpu", workload, capabilities=without)
    assert "multiclass" in str(excinfo.value)


def test_binary_classification_is_single_output():
    workload = ds.Workload(
        n_rows=10, n_features=2, objective="binary", n_classes=2
    )
    assert workload.n_outputs == 1
    assert (
        ds.Workload(
            n_rows=10, n_features=2, objective="multiclass", n_classes=4
        ).n_outputs
        == 4
    )


def test_row_ceiling_blocks_the_gpu(gpu_caps):
    caps = gpu_caps.replace(device_memory_bytes=None, max_rows=1_000)
    workload = ds.Workload(n_rows=1_001, n_features=4)
    with pytest.raises(ds.DeviceUnavailableError) as excinfo:
        ds.select_device("gpu", workload, capabilities=caps)
    assert "index" in str(excinfo.value)
    assert ds.MAX_GPU_ROWS == 2**31 - 1


@pytest.mark.parametrize("max_bin", [1, 257])
def test_bin_count_outside_the_kernel_range_blocks_the_gpu(
    gpu_caps, max_bin
):
    workload = ds.Workload(n_rows=1_000, n_features=4, max_bin=max_bin)
    with pytest.raises(ds.DeviceUnavailableError) as excinfo:
        ds.select_device("gpu", workload, capabilities=gpu_caps)
    assert "max_bin" in str(excinfo.value)


def test_memory_over_budget_blocks_the_gpu(gpu_caps):
    workload = ds.Workload(n_rows=10_000_000, n_features=200)
    tiny = gpu_caps.replace(device_memory_bytes=64 * 1024**2)
    with pytest.raises(ds.DeviceUnavailableError) as excinfo:
        ds.select_device("gpu", workload, capabilities=tiny)
    assert "does not fit" in str(excinfo.value)

    unknown = gpu_caps.replace(device_memory_bytes=None)
    report = ds.explain_device_choice(
        workload, device="gpu", capabilities=unknown
    )
    assert report.would_raise is False
    assert "not a factor" in report.explanation


# --------------------------------------------------------------------
# Soft uncertainty


def test_undocumented_objective_keeps_auto_on_cpu_without_blocking_gpu(
    gpu_caps,
):
    workload = ds.Workload(
        n_rows=5_000_000, n_features=100, objective="tweedie"
    )
    auto = ds.explain_device_choice(
        workload, device="auto", capabilities=gpu_caps
    )
    assert auto.resolved == "cpu"
    assert [r.code for r in auto.reasons] == [
        ds.UNSUPPORTED_OBJECTIVE,
        ds.NO_VALIDATED_RULE,
    ]

    explicit = ds.select_device("gpu", workload, capabilities=gpu_caps)
    assert explicit.resolved == "gpu"
    assert any(
        r.code == ds.UNSUPPORTED_OBJECTIVE for r in explicit.reasons
    )


def test_unknown_backend_is_soft_not_hard(gpu_caps, workload):
    caps = gpu_caps.replace(backend=None, chip=None)
    auto = ds.explain_device_choice(
        workload, device="auto", capabilities=caps
    )
    assert auto.resolved == "cpu"
    assert any(r.code == ds.UNVALIDATED_PATH for r in auto.reasons)
    assert ds.select_device("gpu", workload, capabilities=caps).resolved == (
        "gpu"
    )


# --------------------------------------------------------------------
# auto with no validated rule


def test_auto_without_a_rule_chooses_cpu_and_says_why(gpu_caps, workload):
    report = ds.select_device("auto", workload, capabilities=gpu_caps)
    assert report.resolved == "cpu"
    assert report.rules_considered == 0
    assert report.matched_rule is None
    assert report.validated is False
    assert report.reasons[0].code == ds.NO_VALIDATED_RULE
    assert "no benchmark" in report.reasons[0].message


def test_auto_without_an_accelerator_chooses_cpu(cpu_caps, workload):
    report = ds.select_device("auto", workload, capabilities=cpu_caps)
    assert report.resolved == "cpu"
    assert report.reasons[0].code == ds.NO_ACCELERATOR


# --------------------------------------------------------------------
# auto with an injected rule table


def test_matching_rule_selects_the_gpu(gpu_caps):
    rules = (_rule(backend="metal", min_cells=1_000_000),)
    workload = ds.Workload(n_rows=100_000, n_features=50)
    report = ds.select_device(
        "auto", workload, capabilities=gpu_caps, rules=rules, rules_version=7
    )
    assert report.resolved == "gpu"
    assert report.matched_rule is rules[0]
    assert report.validated is True
    assert report.rules_version == 7
    assert report.reasons[0].code == ds.RULE_MATCHED


def test_workload_below_the_rule_threshold_stays_on_cpu(gpu_caps):
    rules = (_rule(backend="metal", min_cells=10_000_000),)
    workload = ds.Workload(n_rows=1_000, n_features=10)
    report = ds.select_device(
        "auto", workload, capabilities=gpu_caps, rules=rules
    )
    assert report.resolved == "cpu"
    assert report.reasons[0].code == ds.BELOW_RULE_THRESHOLD


def test_rule_scoped_to_another_backend_does_not_match(gpu_caps):
    rules = (_rule(backend="cuda", min_cells=1),)
    workload = ds.Workload(n_rows=1_000_000, n_features=100)
    report = ds.select_device(
        "auto", workload, capabilities=gpu_caps, rules=rules
    )
    assert report.resolved == "cpu"
    assert report.matched_rule is None


def test_rule_scoped_to_another_chip_does_not_match(gpu_caps):
    rules = (_rule(backend="metal", chip="Apple M3 Ultra", min_cells=1),)
    workload = ds.Workload(n_rows=1_000_000, n_features=100)
    report = ds.select_device(
        "auto", workload, capabilities=gpu_caps, rules=rules
    )
    assert report.resolved == "cpu"


def test_rule_scoped_to_objectives_and_classes(gpu_caps):
    rules = (
        _rule(
            backend="metal",
            objectives=("regression",),
            max_classes=2,
            min_rows=1,
        ),
    )
    regression = ds.Workload(n_rows=10_000, n_features=10)
    poisson = ds.Workload(
        n_rows=10_000, n_features=10, objective="poisson"
    )
    many = ds.Workload(
        n_rows=10_000, n_features=10, objective="regression", n_classes=5
    )
    assert (
        ds.select_device(
            "auto", regression, capabilities=gpu_caps, rules=rules
        ).resolved
        == "gpu"
    )
    assert (
        ds.select_device(
            "auto", poisson, capabilities=gpu_caps, rules=rules
        ).resolved
        == "cpu"
    )
    assert (
        ds.select_device(
            "auto", many, capabilities=gpu_caps, rules=rules
        ).resolved
        == "cpu"
    )


def test_a_hard_block_beats_a_matching_rule(gpu_caps):
    rules = (_rule(backend="metal", min_cells=1),)
    workload = ds.Workload(n_rows=1_000_000, n_features=100, sparse=True)
    report = ds.select_device(
        "auto", workload, capabilities=gpu_caps, rules=rules
    )
    assert report.resolved == "cpu"
    assert report.reasons[0].code == ds.UNSUPPORTED_FEATURE


# --------------------------------------------------------------------
# The MOJOBOOST_AUTO_MIN_CELLS knob, mirroring src/mojoboost/device.mojo


def test_env_threshold_selects_the_gpu_above_it(gpu_caps):
    caps = gpu_caps.replace(auto_min_cells=1_000_000)
    above = ds.Workload(n_rows=100_000, n_features=10)
    report = ds.select_device("auto", above, capabilities=caps)
    assert report.resolved == "gpu"
    assert report.reasons[0].code == ds.ENV_THRESHOLD
    # Reached through the knob, so it is explicitly not validated.
    assert report.validated is False
    assert any("not a validated threshold" in w for w in report.warnings)


def test_env_threshold_keeps_the_cpu_below_it(gpu_caps):
    caps = gpu_caps.replace(auto_min_cells=1_000_000)
    below = ds.Workload(n_rows=1_000, n_features=10)
    report = ds.select_device("auto", below, capabilities=caps)
    assert report.resolved == "cpu"
    assert report.reasons[0].code == ds.BELOW_ENV_THRESHOLD


def test_env_threshold_zero_means_whenever_the_gpu_path_covers_it(gpu_caps):
    caps = gpu_caps.replace(auto_min_cells=0)
    workload = ds.Workload(n_rows=1, n_features=1)
    assert (
        ds.select_device("auto", workload, capabilities=caps).resolved
        == "gpu"
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("-1", None),
        ("not-an-int", None),
        ("0", 0),
        ("250000", 250_000),
    ],
)
def test_env_parsing_matches_the_native_rules(raw, expected):
    """device.mojo treats unset, unparsable, and negative alike: the
    heuristic is off. This layer spells that None."""
    environ = {} if raw is None else {"MOJOBOOST_AUTO_MIN_CELLS": raw}
    assert ds._env_auto_min_cells(environ) == expected


# --------------------------------------------------------------------
# Memory estimate


def test_memory_components_follow_the_gpu_buffers():
    workload = ds.Workload(n_rows=1_000, n_features=10, max_bin=255)
    estimate = ds.estimate_gpu_memory(workload)
    assert estimate.components["binned_matrix"] == 1_000 * 10
    assert estimate.components["leaf_ids"] == 1_000 * 4
    assert estimate.components["gradients"] == 1_000 * 4
    assert estimate.components["hessians"] == 1_000 * 4
    assert estimate.components["histograms"] == 10 * 255 * 12
    assert estimate.components["feature_ids"] == 10 * 4
    assert estimate.device_bytes == sum(estimate.components.values())
    assert (
        estimate.upper_bound_bytes
        == estimate.device_bytes + estimate.partial_budget_bytes
    )
    assert estimate.host_bytes == 1_000 * 4 * 2 + 10 * 255 * 12


def test_memory_scales_with_rows_and_outputs():
    small = ds.estimate_gpu_memory(ds.Workload(1_000, 10))
    big = ds.estimate_gpu_memory(ds.Workload(2_000, 10))
    assert big.device_bytes > small.device_bytes

    multi = ds.estimate_gpu_memory(
        ds.Workload(1_000, 10, objective="multiclass", n_classes=4)
    )
    assert multi.components["gradients"] == 4 * small.components["gradients"]


# --------------------------------------------------------------------
# Reports


def test_report_is_json_serializable(gpu_caps, workload):
    report = ds.select_device("auto", workload, capabilities=gpu_caps)
    payload = json.loads(report.to_json())
    assert payload["requested"] == "auto"
    assert payload["resolved"] == "cpu"
    assert payload["rules_version"] == ds.RULES_VERSION
    assert payload["reasons"][0]["code"] == ds.NO_VALIDATED_RULE
    assert payload["workload"]["n_rows"] == workload.n_rows
    assert payload["memory"]["device_bytes"] > 0
    assert payload["capabilities"]["backend"] == "metal"


def test_explanation_reads_as_prose(gpu_caps, workload):
    report = ds.select_device("auto", workload, capabilities=gpu_caps)
    text = report.explanation
    assert str(report) == text
    assert "device='auto' resolved to CPU." in text
    assert "Apple M4" in text
    assert "1,000,000 rows x 100 features" in text
    assert "Why" in text
    assert ds.NO_VALIDATED_RULE in text


def test_explanation_of_a_refusal_carries_the_error(cpu_caps, workload):
    report = ds.explain_device_choice(
        workload, device="gpu", capabilities=cpu_caps
    )
    assert report.would_raise is True
    assert report.resolved is None
    assert "cannot run this workload" in report.explanation
    assert "Error" in report.explanation
    with pytest.raises(ds.DeviceUnavailableError):
        report.raise_if_unsupported()


def test_explain_never_raises_where_select_does(cpu_caps, workload):
    report = ds.explain_device_choice(
        workload, device="gpu", capabilities=cpu_caps
    )
    assert report.error
    with pytest.raises(ds.DeviceUnavailableError):
        ds.select_device("gpu", workload, capabilities=cpu_caps)


# --------------------------------------------------------------------
# Reading a workload off data


class _FakeFrame:
    """The smallest thing with a shape, standing in for numpy or pandas
    so this file needs neither."""

    def __init__(self, shape, sparse=False):
        self.shape = shape
        if sparse:
            self.nnz = 0
            self.format = "csr"

    def tocsr(self):
        return self


def test_explain_reads_shape_from_data(gpu_caps):
    X = _FakeFrame((5_000, 12))
    report = ds.explain_device_choice(X, capabilities=gpu_caps)
    assert report.workload.n_rows == 5_000
    assert report.workload.n_features == 12
    assert report.workload.sparse is False


def test_explain_detects_sparse_input(gpu_caps):
    X = _FakeFrame((5_000, 12), sparse=True)
    report = ds.explain_device_choice(
        X, device="gpu", capabilities=gpu_caps
    )
    assert report.workload.sparse is True
    assert report.would_raise is True
    assert "sparse" in report.error


def test_explain_counts_classes_from_labels(gpu_caps):
    X = _FakeFrame((6, 2))
    report = ds.explain_device_choice(
        X, y=[0, 1, 2, 2, 1, 0], capabilities=gpu_caps
    )
    assert report.workload.n_classes == 3
    assert report.workload.n_outputs == 3


def test_explain_accepts_a_list_of_rows(gpu_caps):
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    report = ds.explain_device_choice(X, capabilities=gpu_caps)
    assert (report.workload.n_rows, report.workload.n_features) == (3, 2)


def test_keyword_overrides_beat_inference(gpu_caps):
    X = _FakeFrame((5_000, 12))
    report = ds.explain_device_choice(
        X, capabilities=gpu_caps, objective="poisson", max_bin=63
    )
    assert report.workload.objective == "poisson"
    assert report.workload.max_bin == 63


def test_shapeless_input_is_a_clear_error(gpu_caps):
    with pytest.raises(ValueError):
        ds.explain_device_choice(object(), capabilities=gpu_caps)


def test_empty_workload_is_rejected():
    with pytest.raises(ValueError):
        ds.Workload(n_rows=0, n_features=4)


# --------------------------------------------------------------------
# Detection, the one test that looks at the real machine


def test_detect_capabilities_answers_without_raising():
    caps = ds.detect_capabilities(environ={})
    assert isinstance(caps.gpu_available, bool)
    assert caps.source == "detected"
    assert caps.backend is None or isinstance(caps.backend, str)


def test_detect_capabilities_honors_the_environment():
    caps = ds.detect_capabilities(
        environ={
            "MOJOBOOST_DISABLE_GPU": "1",
            "MOJOBOOST_AUTO_MIN_CELLS": "500000",
            "MOJOBOOST_GPU_BACKEND": "cuda",
        },
        gpu_available=True,
    )
    assert caps.gpu_available is False
    assert caps.disabled_by_env is True
    assert caps.build_has_accelerator is True
    assert caps.auto_min_cells == 500_000
    assert caps.backend == "cuda"


def test_capabilities_replace_rejects_unknown_fields(gpu_caps):
    assert gpu_caps.replace(chip="Apple M5").chip == "Apple M5"
    with pytest.raises(TypeError):
        gpu_caps.replace(nonsense=True)
