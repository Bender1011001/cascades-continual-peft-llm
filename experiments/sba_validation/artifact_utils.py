from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def collect_nonfinite_paths(value: Any, path: str) -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            failures.extend(collect_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(collect_nonfinite_paths(item, f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        failures.append(path)
    return failures


def validate_required_fields(
    payload: dict[str, Any],
    required: Iterable[str],
    label: str,
) -> list[str]:
    failures: list[str] = []
    for field in required:
        if field not in payload:
            failures.append(f"{label} missing required field: {field}")
    return failures


def gate_payload(
    gate: str,
    command: str,
    seed: int,
    formula_variant: str,
    mask_policy: str,
    dtype_policy: str,
    checks: dict[str, Any],
    failures: list[str],
    log_path: str,
    device: dict[str, Any] | None = None,
    peak_vram_mb: float | None = None,
) -> dict[str, Any]:
    nonfinite_paths = collect_nonfinite_paths(checks, "checks")
    all_failures = list(failures) + [f"non-finite numeric value at {p}" for p in nonfinite_paths]
    payload: dict[str, Any] = {
        "gate": gate,
        "valid": len(all_failures) == 0,
        "failure_reasons": all_failures,
        "git_revision": git_revision(),
        "command": command,
        "seed": seed,
        "device": device or {"type": "cpu"},
        "formula_variant": formula_variant,
        "mask_policy": mask_policy,
        "dtype_policy": dtype_policy,
        "checks": checks,
        "peak_vram_mb": peak_vram_mb,
        "log_path": log_path,
    }
    return payload
