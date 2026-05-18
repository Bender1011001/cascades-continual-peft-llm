from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascades.math_ops import (
    SBA_FORMULA_VARIANT,
    sba_attention_diagnostics,
    sba_jacobian_frobenius_norm,
    sba_optimized_attention,
    sba_reference_attention,
    sba_value_attention,
    softmax_jacobian_frobenius_norm,
)
from experiments.sba_validation.artifact_utils import gate_payload, write_json


SEED = 20260518
MASK_POLICY = "valid_positions_only_zero_masked_weights"
DTYPE_POLICY = "float64_cpu_primary_float32_smoke"


def _finite_grad_check() -> bool:
    logits = torch.tensor(
        [[[15.0, -12.0, 0.5, -0.25], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid_mask = torch.tensor([[[True, True, True, False], [True, True, True, True]]])
    weights = sba_optimized_attention(logits, valid_mask)
    loss = weights.square().sum() + weights.sum()
    loss.backward()
    return logits.grad is not None and bool(torch.isfinite(logits.grad).all().item())


def run_cpu_math_gate(
    artifact_path: Path = Path("experiments/sba_validation/cpu_math_gate.json"),
    log_path: Path = Path("experiments/sba_validation/cpu_math_gate.log"),
) -> dict[str, object]:
    torch.manual_seed(SEED)
    failures: list[str] = []
    generator = torch.Generator().manual_seed(SEED)
    equivalence_diffs: list[float] = []
    signed_l1_values: list[float] = []
    masked_leakage_values: list[float] = []
    finite_outputs = True
    for shape in [(2, 3), (2, 4, 5), (2, 2, 3, 7)]:
        logits = torch.randn(*shape, generator=generator, dtype=torch.float64) * 3.0
        valid_mask = torch.rand(*shape, generator=generator) > 0.25
        valid_mask[..., 0] = True
        reference = sba_reference_attention(logits, valid_mask)
        optimized = sba_optimized_attention(logits, valid_mask)
        equivalence_diffs.append(float((reference - optimized).abs().max().item()))
        finite_outputs = finite_outputs and bool(torch.isfinite(optimized).all().item())
        diagnostics = sba_attention_diagnostics(optimized, valid_mask)
        signed_l1_values.append(float(diagnostics["max_signed_l1"]))
        masked_leakage_values.append(float(diagnostics["max_masked_leakage"]))
    all_masked_logits = torch.tensor([[[1.0, -2.0, 3.0]]], dtype=torch.float64)
    all_masked = torch.zeros_like(all_masked_logits, dtype=torch.bool)
    all_masked_weights = sba_optimized_attention(all_masked_logits, all_masked)
    values = torch.randn(1, 3, 2, generator=generator, dtype=torch.float64)
    weights, attended = sba_value_attention(all_masked_logits, values, all_masked)
    finite_outputs = finite_outputs and bool(torch.isfinite(weights).all().item())
    finite_outputs = finite_outputs and bool(torch.isfinite(attended).all().item())
    if not torch.all(all_masked_weights == 0.0).item():
        failures.append("all-masked row produced nonzero SBA weights")
    max_equivalence = max(equivalence_diffs, default=0.0)
    max_signed_l1 = max(signed_l1_values, default=0.0)
    max_masked_leakage = max(masked_leakage_values, default=0.0)
    if max_equivalence > 1e-8:
        failures.append(f"formula equivalence max diff {max_equivalence} exceeds 1e-8")
    if max_masked_leakage > 0.0:
        failures.append(f"masked leakage {max_masked_leakage} exceeds 0.0")
    if max_signed_l1 > 1.0 + 1e-8:
        failures.append(f"signed L1 {max_signed_l1} exceeds 1 + 1e-8")
    finite_gradients = _finite_grad_check()
    if not finite_outputs:
        failures.append("SBA outputs or value aggregation are non-finite")
    if not finite_gradients:
        failures.append("SBA gradients are non-finite")
    jacobian_logits = torch.tensor([[0.25, -0.75, 1.25]], dtype=torch.float64)
    jacobian_mask = torch.tensor([[True, True, True]])
    sba_jacobian = sba_jacobian_frobenius_norm(jacobian_logits, jacobian_mask)
    softmax_jacobian = softmax_jacobian_frobenius_norm(jacobian_logits, jacobian_mask)
    jacobian_claim_status = "supported" if sba_jacobian <= softmax_jacobian else "falsified"
    checks = {
        "formula_equivalence_max_diff": max_equivalence,
        "max_masked_leakage": max_masked_leakage,
        "max_signed_l1": max_signed_l1,
        "finite_outputs": finite_outputs,
        "finite_gradients": finite_gradients,
        "jacobian_claim": "sba_jacobian_frobenius_norm <= softmax_jacobian_frobenius_norm",
        "jacobian_claim_status": jacobian_claim_status,
        "sba_jacobian_frobenius_norm": sba_jacobian,
        "softmax_jacobian_frobenius_norm": softmax_jacobian,
    }
    payload = gate_payload(
        gate="cpu_math",
        command="python experiments\\sba_validation\\run_cpu_math_gate.py",
        seed=SEED,
        formula_variant=SBA_FORMULA_VARIANT,
        mask_policy=MASK_POLICY,
        dtype_policy=DTYPE_POLICY,
        checks=checks,
        failures=failures,
        log_path=str(log_path),
    )
    write_json(artifact_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "CPU math gate completed\n"
        f"valid={payload['valid']}\n"
        f"failure_reasons={payload['failure_reasons']}\n"
        f"checks={checks}\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBA CPU math validation gate")
    parser.add_argument("--out", default="experiments/sba_validation/cpu_math_gate.json")
    parser.add_argument("--log", default="experiments/sba_validation/cpu_math_gate.log")
    args = parser.parse_args()
    payload = run_cpu_math_gate(Path(args.out), Path(args.log))
    if not payload["valid"]:
        print(payload["failure_reasons"])
        raise SystemExit(2)
    print(f"CPU math gate valid: {args.out}")


if __name__ == "__main__":
    main()
