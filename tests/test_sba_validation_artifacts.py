import math

from experiments.sba_validation.artifact_utils import (
    collect_nonfinite_paths,
    gate_payload,
    validate_required_fields,
)
from experiments.sba_validation.compare_paired_reduced_sba import compare_paired_artifacts
from experiments.sba_validation.run_cuda_synthetic_gate import CUDA_SYNTHETIC_CASES, build_case_label
from experiments.sba_validation.run_cpu_math_gate import run_cpu_math_gate
from experiments.sba_validation.run_paired_reduced_sba import PAIRED_REDUCED_CONFIG, REASONING3_TASKS
from experiments.sba_validation.run_tiny_model_gate import TINY_MODEL_GATE_COMMAND

# NOTE: Tests of the standalone SBA modules (benchmark_sba_implementations,
# run_scratch_compare) moved to the symmetric-bipolar-attention repo when SBA
# was split out. See test_sba_standalone_units.py there.


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


def test_tiny_model_gate_command_uses_sba_patch_and_threshold():
    assert "--enable-sba-patch" in TINY_MODEL_GATE_COMMAND
    assert "--vram-threshold-mb 7500" in TINY_MODEL_GATE_COMMAND


def test_paired_reduced_config_is_bounded():
    assert REASONING3_TASKS == [
        "data/task0_gsm8k_cot.jsonl",
        "data/task1_arc_cot.jsonl",
        "data/task2_csqa_cot.jsonl",
    ]
    assert PAIRED_REDUCED_CONFIG["seed"] == 43
    assert PAIRED_REDUCED_CONFIG["rank"] == 4
    assert PAIRED_REDUCED_CONFIG["max_length"] == 256
    assert PAIRED_REDUCED_CONFIG["epochs"] == 2
    assert PAIRED_REDUCED_CONFIG["vram_threshold_mb"] == 7500.0


def test_compare_rejects_mismatched_manifest(tmp_path):
    root = tmp_path / "paired_reduced"
    for arm in ["control", "treatment"]:
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        (arm_dir / "run_status.json").write_text(
            '{"status":"completed","seed":43,"git_revision":"abc"}', encoding="utf-8"
        )
        (arm_dir / "metrics.json").write_text(
            '{"avg_acc":0.5,"bwt":0.0,"peak_vram_mb":1000.0,"old_task_deltas":[0.0,0.0]}',
            encoding="utf-8",
        )
        manifest = [{"task_index": 0, "path": "data/task0_gsm8k_cot.jsonl"}]
        if arm == "treatment":
            manifest = [{"task_index": 0, "path": "data/task1_arc_cot.jsonl"}]
        (arm_dir / "task_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
        (arm_dir / "instrumentation.json").write_text(
            '{"sba_patch":{"patched_module_count":1,"patched_module_names":["model.layers.0.self_attn"]}}',
            encoding="utf-8",
        )
        (arm_dir / "config.json").write_text(
            '{"seed":43,"rank":4,"max_length":256,"epochs":2,"model_id":"model","git_revision":"abc"}',
            encoding="utf-8",
        )
    result = compare_paired_artifacts(root, fail_invalid=False)
    assert result["valid"] is False
    assert "task manifests differ across arms" in result["failures"]
