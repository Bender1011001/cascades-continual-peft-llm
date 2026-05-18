from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_ARM_FILES = ["run_status.json", "metrics.json", "task_manifest.json", "instrumentation.json", "config.json"]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _nonfinite(value: Any, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(_nonfinite(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_nonfinite(item, f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        failures.append(f"non-finite numeric value at {path}")
    return failures


def compare_paired_artifacts(root: Path, vram_threshold_mb: float = 7500.0, fail_invalid: bool = True) -> dict[str, Any]:
    failures: list[str] = []
    arms: dict[str, dict[str, Any]] = {}
    for arm in ["control", "treatment"]:
        arm_dir = root / arm
        payloads: dict[str, Any] = {}
        for name in REQUIRED_ARM_FILES:
            path = arm_dir / name
            if not path.exists():
                failures.append(f"{arm} missing required artifact: {name}")
            else:
                payloads[name] = _load(path)
        if len(payloads) == len(REQUIRED_ARM_FILES):
            arms[arm] = payloads
    if set(arms) == {"control", "treatment"}:
        for arm, payloads in arms.items():
            if payloads["run_status.json"].get("status") != "completed":
                failures.append(f"{arm} status is not completed")
            failures.extend(_nonfinite(payloads["metrics.json"], f"{arm}.metrics"))
            if float(payloads["metrics.json"].get("peak_vram_mb", math.inf)) > vram_threshold_mb:
                failures.append(f"{arm} peak_vram_mb exceeds {vram_threshold_mb:.0f}")
        if arms["control"]["task_manifest.json"] != arms["treatment"]["task_manifest.json"]:
            failures.append("task manifests differ across arms")
        for field in ["seed", "rank", "max_length", "epochs", "model_id"]:
            if arms["control"]["config.json"].get(field) != arms["treatment"]["config.json"].get(field):
                failures.append(f"paired config field differs: {field}")
        if arms["control"]["config.json"].get("git_revision") != arms["treatment"]["config.json"].get("git_revision"):
            failures.append("git revisions differ across arms")
        sba_patch = arms["treatment"]["instrumentation.json"].get("sba_patch", {})
        if int(sba_patch.get("patched_module_count", 0)) <= 0:
            failures.append("treatment SBA patch inactive")
    if failures:
        result = {"valid": False, "failures": failures}
        if fail_invalid:
            print(json.dumps(result, indent=2, allow_nan=False))
            raise SystemExit(2)
        return result
    control_metrics = arms["control"]["metrics.json"]
    treatment_metrics = arms["treatment"]["metrics.json"]
    delta_bwt_points = (treatment_metrics["bwt"] - control_metrics["bwt"]) * 100.0
    delta_avg_acc_points = (treatment_metrics["avg_acc"] - control_metrics["avg_acc"]) * 100.0
    old_task_delta_gaps_points = [
        (treatment - control) * 100.0
        for treatment, control in zip(treatment_metrics["old_task_deltas"], control_metrics["old_task_deltas"])
    ]
    return {
        "valid": True,
        "failures": [],
        "delta_bwt_points": delta_bwt_points,
        "delta_avg_acc_points": delta_avg_acc_points,
        "old_task_delta_gaps_points": old_task_delta_gaps_points,
        "control_peak_vram_mb": control_metrics["peak_vram_mb"],
        "treatment_peak_vram_mb": treatment_metrics["peak_vram_mb"],
        "vram_threshold_mb": vram_threshold_mb,
        "sba_patch_active": True,
        "continue": delta_bwt_points >= 1.5
        and delta_avg_acc_points >= -2.0
        and min(old_task_delta_gaps_points, default=0.0) >= -3.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired reduced SBA artifacts")
    parser.add_argument("--root", default="experiments/sba_validation/paired_reduced")
    parser.add_argument("--out", default="experiments/sba_validation/paired_reduced/comparison.json")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    result = compare_paired_artifacts(Path(args.root), args.vram_threshold_mb, fail_invalid=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
