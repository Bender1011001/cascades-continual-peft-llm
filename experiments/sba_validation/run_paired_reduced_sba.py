from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cascades.data as data
from experiments.sba_validation.artifact_utils import git_revision, write_json


REASONING3_TASKS = [
    "data/task0_gsm8k_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task2_csqa_cot.jsonl",
]
PAIRED_REDUCED_CONFIG = {
    "model_id": "p-e-w/Qwen3-4B-Instruct-2507-heretic",
    "seed": 43,
    "rank": 4,
    "max_length": 256,
    "epochs": 2,
    "vram_threshold_mb": 7500.0,
    "task_suite": "reasoning3",
}


def paired_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    config = dict(PAIRED_REDUCED_CONFIG)
    for key in ("seed", "rank", "max_length", "epochs", "vram_threshold_mb"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def apply_reasoning3_manifest() -> list[str]:
    data.TASK_FILES = list(REASONING3_TASKS)
    data.TASK_NAMES = {i: f"Task {i} ({Path(path).stem})" for i, path in enumerate(REASONING3_TASKS)}
    data.NUM_TASKS = len(REASONING3_TASKS)
    return list(REASONING3_TASKS)


def task_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, file_name in enumerate(REASONING3_TASKS):
        path = ROOT / file_name
        rows.append(
            {
                "task_index": index,
                "path": file_name,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return rows


def run_arm(arm: str, output_root: Path, enable_sba_patch: bool, config: dict[str, object] | None = None) -> None:
    import train

    run_config = dict(PAIRED_REDUCED_CONFIG if config is None else config)
    apply_reasoning3_manifest()
    train.NUM_TASKS = len(REASONING3_TASKS)
    arm_dir = output_root / arm
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True)
    write_json(arm_dir / "task_manifest.json", task_manifest())
    artifact_config = dict(run_config)
    artifact_config.update({"arm": arm, "enable_sba_patch": enable_sba_patch, "git_revision": git_revision()})
    write_json(arm_dir / "config.json", artifact_config)
    patch_report_path = arm_dir / "sba_patch_report.json"
    status = "completed"
    reason = None
    try:
        matrix = train.train_cascades(
            seed=int(run_config["seed"]),
            model_id=str(run_config["model_id"]),
            output_prefix=str(arm_dir / "cascades"),
            epochs=int(run_config["epochs"]),
            rank=int(run_config["rank"]),
            max_length=int(run_config["max_length"]),
            enable_sleep=True,
            abort_on_nonfinite=True,
            vram_threshold_mb=float(run_config["vram_threshold_mb"]),
            enable_sba_patch=enable_sba_patch,
            sba_patch_report_path=str(patch_report_path) if enable_sba_patch else None,
        )
    except Exception as exc:
        status = "failed"
        reason = str(exc)
        matrix = None
    if matrix is not None:
        import numpy as np
        from cascades.metrics import average_accuracy, backward_transfer

        np.save(arm_dir / "accuracy_matrix.npy", matrix)
        metrics = {
            "arm": arm,
            "avg_acc": float(average_accuracy(matrix)),
            "bwt": float(backward_transfer(matrix)),
            "old_task_deltas": [float(matrix[-1, i] - matrix[i, i]) for i in range(matrix.shape[0] - 1)],
            "peak_vram_mb": (
                float(__import__("torch").cuda.max_memory_allocated() / (1024**2))
                if __import__("torch").cuda.is_available()
                else 0.0
            ),
            "seed": int(run_config["seed"]),
            "rank": int(run_config["rank"]),
            "max_length": int(run_config["max_length"]),
            "epochs": int(run_config["epochs"]),
            "git_revision": git_revision(),
        }
        write_json(arm_dir / "metrics.json", metrics)
    sba_patch = (
        json.loads(patch_report_path.read_text(encoding="utf-8"))
        if patch_report_path.exists()
        else {"patched_module_count": 0, "patched_module_names": []}
    )
    write_json(arm_dir / "instrumentation.json", {"sba_patch": sba_patch})
    write_json(
        arm_dir / "run_status.json",
        {"status": status, "reason": reason, "seed": int(run_config["seed"]), "git_revision": git_revision()},
    )
    if status != "completed":
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired reduced SBA validation")
    parser.add_argument("--arm", choices=["control", "treatment"], required=True)
    parser.add_argument("--output-root", default="experiments/sba_validation/paired_reduced")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--vram-threshold-mb", type=float, default=None)
    args = parser.parse_args()
    run_arm(
        args.arm,
        ROOT / args.output_root,
        enable_sba_patch=args.arm == "treatment",
        config=paired_config_from_args(args),
    )
    print(f"SBA paired reduced {args.arm} completed")


if __name__ == "__main__":
    main()
