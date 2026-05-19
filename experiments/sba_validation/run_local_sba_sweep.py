from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


SWEEP_CASES = [
    {"name": "short128_seed44", "seed": 44, "max_length": 128, "epochs": 2, "vram_threshold_mb": 7500.0},
    {"name": "short192_seed43", "seed": 43, "max_length": 192, "epochs": 2, "vram_threshold_mb": 7500.0},
    {"name": "short128_seed45", "seed": 45, "max_length": 128, "epochs": 2, "vram_threshold_mb": 7500.0},
]


def run_command(command: list[str], timeout_s: float | None = None) -> dict[str, object]:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "elapsed_s": time.time() - started,
            "output_tail": completed.stdout[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "elapsed_s": time.time() - started,
            "output_tail": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "timeout",
        }


def artifact_status(root: Path) -> dict[str, object]:
    status: dict[str, object] = {"root": str(root)}
    for arm in ["treatment", "control"]:
        arm_dir = root / arm
        arm_status = arm_dir / "run_status.json"
        arm_metrics = arm_dir / "metrics.json"
        status[arm] = {
            "status": json.loads(arm_status.read_text(encoding="utf-8")) if arm_status.exists() else None,
            "metrics": json.loads(arm_metrics.read_text(encoding="utf-8")) if arm_metrics.exists() else None,
        }
    comparison = root / "comparison.json"
    status["comparison"] = json.loads(comparison.read_text(encoding="utf-8")) if comparison.exists() else None
    return status


def run_case(case: dict[str, object], output_root: Path, deadline: float) -> dict[str, object]:
    case_root = output_root / str(case["name"])
    case_root.mkdir(parents=True, exist_ok=True)
    timeout_s = max(1.0, deadline - time.time())
    base = [
        sys.executable,
        "experiments/sba_validation/run_paired_reduced_sba.py",
        "--output-root",
        str(case_root.relative_to(ROOT)).replace("\\", "/"),
        "--seed",
        str(case["seed"]),
        "--max-length",
        str(case["max_length"]),
        "--epochs",
        str(case["epochs"]),
        "--vram-threshold-mb",
        str(case["vram_threshold_mb"]),
    ]
    results: dict[str, object] = {"case": case, "root": str(case_root), "commands": []}
    for arm in ["treatment", "control"]:
        if time.time() >= deadline:
            results["deadline_reached_before"] = arm
            break
        arm_cmd = base + ["--arm", arm]
        arm_result = run_command(arm_cmd, timeout_s=max(1.0, deadline - time.time()))
        results["commands"].append({"arm": arm, **arm_result})
        if arm_result["returncode"] != 0:
            break
    treatment_status = case_root / "treatment" / "run_status.json"
    control_status = case_root / "control" / "run_status.json"
    if treatment_status.exists() and control_status.exists():
        treatment = json.loads(treatment_status.read_text(encoding="utf-8"))
        control = json.loads(control_status.read_text(encoding="utf-8"))
        if treatment.get("status") == "completed" and control.get("status") == "completed":
            compare_cmd = [
                sys.executable,
                "experiments/sba_validation/compare_paired_reduced_sba.py",
                "--root",
                str(case_root.relative_to(ROOT)).replace("\\", "/"),
                "--out",
                str((case_root / "comparison.json").relative_to(ROOT)).replace("\\", "/"),
                "--vram-threshold-mb",
                str(case["vram_threshold_mb"]),
            ]
            results["commands"].append({"arm": "compare", **run_command(compare_cmd, timeout_s=max(1.0, deadline - time.time()))})
    results["artifact_status"] = artifact_status(case_root)
    return results


def summarize(output_root: Path) -> dict[str, object]:
    cases = []
    for child in sorted(output_root.iterdir()):
        if child.is_dir():
            cases.append(artifact_status(child))
    valid_comparisons = [case["comparison"] for case in cases if case.get("comparison", {}).get("valid")]
    return {
        "cases": cases,
        "valid_comparison_count": len(valid_comparisons),
        "delta_avg_acc_points": [item["delta_avg_acc_points"] for item in valid_comparisons],
        "delta_bwt_points": [item["delta_bwt_points"] for item in valid_comparisons],
        "continue_count": sum(1 for item in valid_comparisons if item.get("continue")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded local GPU SBA-vs-control evidence sweep")
    parser.add_argument("--output-root", default="experiments/sba_validation/local_gpu_sweep")
    parser.add_argument("--hours", type=float, default=4.5)
    args = parser.parse_args()
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.hours * 3600.0
    sweep_log: dict[str, object] = {"started_at": time.time(), "deadline": deadline, "cases": []}
    for case in SWEEP_CASES:
        if time.time() >= deadline:
            break
        result = run_case(case, output_root, deadline)
        sweep_log["cases"].append(result)
        (output_root / "sweep_log.json").write_text(json.dumps(sweep_log, indent=2, allow_nan=False), encoding="utf-8")
    summary = summarize(output_root)
    summary["finished_at"] = time.time()
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
