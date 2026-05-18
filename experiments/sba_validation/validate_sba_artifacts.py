from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sba_validation.artifact_utils import collect_nonfinite_paths, validate_required_fields


REQUIRED_GATE_FIELDS = [
    "gate",
    "valid",
    "failure_reasons",
    "git_revision",
    "command",
    "seed",
    "formula_variant",
    "mask_policy",
    "dtype_policy",
    "checks",
    "log_path",
]


def validate_gate_artifact(path: Path) -> dict[str, object]:
    failures: list[str] = []
    if not path.exists():
        return {"valid": False, "failures": [f"missing artifact: {path}"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures.extend(validate_required_fields(payload, REQUIRED_GATE_FIELDS, str(path)))
    failures.extend([f"non-finite numeric value at {p}" for p in collect_nonfinite_paths(payload, str(path))])
    if payload.get("valid") is not True:
        failures.append(f"artifact valid field is not true: {path}")
    return {"valid": len(failures) == 0, "failures": failures, "artifact": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SBA gate artifact JSON")
    parser.add_argument("artifact")
    args = parser.parse_args()
    result = validate_gate_artifact(Path(args.artifact))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
