import math

from experiments.sba_validation.artifact_utils import (
    collect_nonfinite_paths,
    gate_payload,
    validate_required_fields,
)
from experiments.sba_validation.run_cuda_synthetic_gate import CUDA_SYNTHETIC_CASES, build_case_label
from experiments.sba_validation.run_cpu_math_gate import run_cpu_math_gate


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


def test_cpu_math_gate_payload_is_valid_and_records_jacobian_status(tmp_path):
    artifact_path = tmp_path / "cpu_math_gate.json"
    log_path = tmp_path / "cpu_math_gate.log"
    payload = run_cpu_math_gate(artifact_path=artifact_path, log_path=log_path)
    assert payload["valid"] is True
    assert payload["checks"]["formula_equivalence_max_diff"] <= 1e-8
    assert payload["checks"]["max_masked_leakage"] == 0.0
    assert payload["checks"]["max_signed_l1"] <= 1.0 + 1e-8
    assert payload["checks"]["finite_outputs"] is True
    assert payload["checks"]["finite_gradients"] is True
    assert payload["checks"]["jacobian_claim_status"] in {"supported", "falsified"}
    assert artifact_path.exists()
    assert log_path.exists()


def test_cuda_synthetic_cases_are_bounded_and_labeled():
    labels = [build_case_label(case) for case in CUDA_SYNTHETIC_CASES]
    assert labels == [
        "b1_h2_q16_k16_d32_float32_mask0.00",
        "b1_h2_q32_k32_d64_float32_mask0.25",
        "b1_h4_q64_k64_d64_float16_mask0.50",
    ]
