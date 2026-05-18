from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascades.math_ops import SBA_FORMULA_VARIANT, sba_attention_diagnostics, sba_optimized_attention
from experiments.sba_validation.artifact_utils import gate_payload, write_json


CUDA_SYNTHETIC_CASES = [
    {"batch": 1, "heads": 2, "queries": 16, "keys": 16, "dim": 32, "dtype": "float32", "mask_density": 0.0},
    {"batch": 1, "heads": 2, "queries": 32, "keys": 32, "dim": 64, "dtype": "float32", "mask_density": 0.25},
    {"batch": 1, "heads": 4, "queries": 64, "keys": 64, "dim": 64, "dtype": "float16", "mask_density": 0.50},
]


def build_case_label(case: dict[str, object]) -> str:
    return (
        f"b{case['batch']}_h{case['heads']}_q{case['queries']}_k{case['keys']}"
        f"_d{case['dim']}_{case['dtype']}_mask{float(case['mask_density']):.2f}"
    )


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def run_cuda_synthetic_gate(
    artifact_path: Path = Path("experiments/sba_validation/cuda_synthetic_gate.json"),
    log_path: Path = Path("experiments/sba_validation/cuda_synthetic_gate.log"),
    vram_threshold_mb: float = 7500.0,
) -> dict[str, object]:
    seed = 20260518
    torch.manual_seed(seed)
    failures: list[str] = []
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
        payload = gate_payload(
            gate="cuda_synthetic",
            command="python experiments\\sba_validation\\run_cuda_synthetic_gate.py --vram-threshold-mb 7500",
            seed=seed,
            formula_variant=SBA_FORMULA_VARIANT,
            mask_policy="valid_positions_only_zero_masked_weights",
            dtype_policy="float32_and_float16_cuda",
            checks={"cases": []},
            failures=failures,
            log_path=str(log_path),
            device={"type": "cuda", "available": False},
        )
        write_json(artifact_path, payload)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("CUDA unavailable\n", encoding="utf-8")
        return payload
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    case_results: list[dict[str, object]] = []
    for case in CUDA_SYNTHETIC_CASES:
        dtype = _dtype(str(case["dtype"]))
        label = build_case_label(case)
        shape = (int(case["batch"]), int(case["heads"]), int(case["queries"]), int(case["keys"]))
        logits = torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)
        mask = torch.rand(*shape, device=device) >= float(case["mask_density"])
        mask[..., 0] = True
        start = time.perf_counter()
        weights = sba_optimized_attention(logits, mask)
        loss = weights.float().square().mean()
        loss.backward()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        diagnostics = sba_attention_diagnostics(weights.detach().float(), mask)
        grad_finite = logits.grad is not None and bool(torch.isfinite(logits.grad).all().item())
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        case_result = {
            "label": label,
            "shape": list(shape),
            "dtype": str(case["dtype"]),
            "mask_density": float(case["mask_density"]),
            "forward_finite": bool(torch.isfinite(weights).all().item()),
            "backward_finite": grad_finite,
            "max_signed_l1": diagnostics["max_signed_l1"],
            "max_masked_leakage": diagnostics["max_masked_leakage"],
            "elapsed_ms": elapsed_ms,
            "peak_vram_mb": peak_mb,
        }
        case_results.append(case_result)
        if not case_result["forward_finite"]:
            failures.append(f"{label} forward output is non-finite")
        if not case_result["backward_finite"]:
            failures.append(f"{label} backward gradient is non-finite")
        if float(case_result["max_signed_l1"]) > 1.0 + 2e-3:
            failures.append(f"{label} signed L1 bound exceeded CUDA tolerance")
        if float(case_result["max_masked_leakage"]) > 1e-6:
            failures.append(f"{label} masked leakage exceeded 1e-6")
        if peak_mb > vram_threshold_mb:
            failures.append(f"{label} peak VRAM {peak_mb:.0f} MB exceeded {vram_threshold_mb:.0f} MB")
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    checks = {"cases": case_results, "vram_threshold_mb": vram_threshold_mb}
    payload = gate_payload(
        gate="cuda_synthetic",
        command="python experiments\\sba_validation\\run_cuda_synthetic_gate.py --vram-threshold-mb 7500",
        seed=seed,
        formula_variant=SBA_FORMULA_VARIANT,
        mask_policy="valid_positions_only_zero_masked_weights",
        dtype_policy="float32_and_float16_cuda",
        checks=checks,
        failures=failures,
        log_path=str(log_path),
        device={"type": "cuda", "available": True, "name": torch.cuda.get_device_name(device)},
        peak_vram_mb=peak_vram_mb,
    )
    write_json(artifact_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"CUDA synthetic gate valid={payload['valid']}\nchecks={checks}\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBA synthetic CUDA stress gate")
    parser.add_argument("--out", default="experiments/sba_validation/cuda_synthetic_gate.json")
    parser.add_argument("--log", default="experiments/sba_validation/cuda_synthetic_gate.log")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    payload = run_cuda_synthetic_gate(Path(args.out), Path(args.log), args.vram_threshold_mb)
    if not payload["valid"]:
        print(payload["failure_reasons"])
        raise SystemExit(2)
    print(f"CUDA synthetic gate valid: {args.out}")


if __name__ == "__main__":
    main()
