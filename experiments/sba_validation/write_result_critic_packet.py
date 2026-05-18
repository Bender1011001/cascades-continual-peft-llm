from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ESCALATION_GUARD = (
    "Larger-suite escalation remains blocked unless every prior gate passes, "
    "the result critic accepts the evidence quality, and the user explicitly "
    "approves escalation in a later decision."
)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "valid": False,
            "missing": True,
            "failure_reasons": [f"missing artifact: {path}"],
            "artifact_path": str(path),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "missing": False,
            "failure_reasons": [f"invalid JSON in {path}: {exc}"],
            "artifact_path": str(path),
        }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "missing": False,
            "failure_reasons": [f"artifact is not a JSON object: {path}"],
            "artifact_path": str(path),
        }
    payload.setdefault("artifact_path", str(path))
    return payload


def _failure_reasons(payload: dict[str, Any]) -> list[Any]:
    reasons = payload.get("failure_reasons", payload.get("failures", []))
    if reasons is None:
        return []
    if isinstance(reasons, list):
        return reasons
    return [reasons]


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _fmt_reasons(payload: dict[str, Any]) -> str:
    reasons = _failure_reasons(payload)
    if not reasons:
        return "none"
    return "; ".join(str(reason) for reason in reasons)


def _checks(payload: dict[str, Any]) -> dict[str, Any]:
    checks = payload.get("checks", {})
    return checks if isinstance(checks, dict) else {}


def _gate_line(label: str, payload: dict[str, Any]) -> str:
    return (
        f"- {label}: valid={_fmt(payload.get('valid'))}; "
        f"artifact={payload.get('artifact_path', 'missing')}; "
        f"reasons={_fmt_reasons(payload)}"
    )


def write_packet(root: Path, out: Path) -> str:
    cpu = _load(root / "cpu_math_gate.json")
    cuda = _load(root / "cuda_synthetic_gate.json")
    tiny = _load(root / "tiny_model_gate.json")
    comparison = _load(root / "paired_reduced" / "comparison.json")

    cpu_checks = _checks(cpu)
    all_prior_gates_passed = all(payload.get("valid") is True for payload in (cpu, cuda, tiny, comparison))
    escalation_allowed = False

    lines = [
        "# SBA Validation Result Critic Packet",
        "",
        "## Gate verdicts",
        _gate_line("CPU math gate", cpu),
        _gate_line("Synthetic CUDA gate", cuda),
        _gate_line("Tiny patched-model gate", tiny),
        _gate_line("Paired reduced comparison", comparison),
        "",
        "## CPU math criteria",
        f"- Formula variant: {_fmt(cpu.get('formula_variant'))}",
        f"- Formula equivalence max diff: {_fmt(cpu_checks.get('formula_equivalence_max_diff'))}",
        f"- Mask policy: {_fmt(cpu.get('mask_policy'))}",
        f"- Masked leakage max: {_fmt(cpu_checks.get('max_masked_leakage'))}",
        f"- Signed L1 max: {_fmt(cpu_checks.get('max_signed_l1'))}",
        f"- Finite outputs: {_fmt(cpu_checks.get('finite_outputs'))}",
        f"- Finite gradients: {_fmt(cpu_checks.get('finite_gradients'))}",
        f"- Jacobian claim: {_fmt(cpu_checks.get('jacobian_claim'))}",
        f"- Jacobian claim status: {_fmt(cpu_checks.get('jacobian_claim_status'))}",
        f"- SBA Jacobian Frobenius norm: {_fmt(cpu_checks.get('sba_jacobian_frobenius_norm'))}",
        f"- Softmax Jacobian Frobenius norm: {_fmt(cpu_checks.get('softmax_jacobian_frobenius_norm'))}",
        "",
        "## VRAM and paired deltas",
        f"- CPU peak VRAM MB: {_fmt(cpu.get('peak_vram_mb'))}",
        f"- Synthetic CUDA peak VRAM MB: {_fmt(cuda.get('peak_vram_mb'))}",
        f"- Tiny-model peak VRAM MB: {_fmt(tiny.get('peak_vram_mb'))}",
        f"- Paired control peak VRAM MB: {_fmt(comparison.get('control_peak_vram_mb'))}",
        f"- Paired treatment peak VRAM MB: {_fmt(comparison.get('treatment_peak_vram_mb'))}",
        f"- Paired delta BWT points: {_fmt(comparison.get('delta_bwt_points'))}",
        f"- Paired delta average accuracy points: {_fmt(comparison.get('delta_avg_acc_points'))}",
        f"- Paired old-task delta gap points: {_fmt(comparison.get('old_task_delta_gaps_points'))}",
        f"- Paired continue recommendation: {_fmt(comparison.get('continue'))}",
        "",
        "## Escalation decision guard",
        f"- All prior gates passed: {_fmt(all_prior_gates_passed)}",
        f"- Larger-suite escalation allowed now: {_fmt(escalation_allowed)}",
        f"- Guard: {ESCALATION_GUARD}",
        "- Required next decision: route this packet to result criticism; do not claim catastrophic forgetting is solved.",
        "- Current default: SBA remains default-off.",
    ]
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write SBA validation result critic packet")
    parser.add_argument("--root", default="experiments/sba_validation")
    parser.add_argument("--out", default="experiments/sba_validation/RESULT_CRITIC_PACKET.md")
    args = parser.parse_args()
    write_packet(Path(args.root), Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
