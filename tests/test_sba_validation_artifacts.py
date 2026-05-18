import math

from experiments.sba_validation.artifact_utils import (
    collect_nonfinite_paths,
    gate_payload,
    validate_required_fields,
)


def test_collect_nonfinite_paths_finds_nested_nan_and_inf():
    payload = {"outer": {"nan": float("nan"), "values": [1.0, float("inf")]}}
    failures = collect_nonfinite_paths(payload, "root")
    assert "root.outer.nan" in failures
    assert "root.outer.values[1]" in failures


def test_validate_required_fields_reports_missing_field():
    failures = validate_required_fields({"valid": True}, ["valid", "seed"], "artifact")
    assert failures == ["artifact missing required field: seed"]


def test_gate_payload_rejects_nonfinite_numbers():
    payload = gate_payload(
        gate="unit",
        command="python -m pytest",
        seed=123,
        formula_variant="sba-v1-valid-position-sinh-cosh-shared-denominator",
        mask_policy="valid_positions_only",
        dtype_policy="float64_cpu",
        checks={"finite": True, "bad_value": math.inf},
        failures=[],
        log_path="experiments/sba_validation/unit.log",
    )
    assert payload["valid"] is False
    assert "checks.bad_value" in payload["failure_reasons"][0]
