from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascades.math_ops import SBA_FORMULA_VARIANT
from cascades.sba_patch import apply_sba_patch
from experiments.sba_validation.artifact_utils import gate_payload, write_json


DEFAULT_MODEL_ID = "p-e-w/Qwen3-4B-Instruct-2507-heretic"
TINY_MODEL_GATE_COMMAND = (
    "python experiments\\sba_validation\\run_tiny_model_gate.py "
    "--model-id p-e-w/Qwen3-4B-Instruct-2507-heretic "
    "--enable-sba-patch --vram-threshold-mb 7500"
)


def _exception_failure(stage: str, exc: Exception) -> str:
    return f"{stage} failed: {exc.__class__.__name__}: {exc}"


def _cuda_device(device: torch.device) -> dict[str, object]:
    try:
        return {"type": "cuda", "available": True, "name": torch.cuda.get_device_name(device)}
    except Exception as exc:
        return {"type": "cuda", "available": True, "name": f"unavailable: {exc.__class__.__name__}: {exc}"}


def _write_invalid_gate_artifact(
    artifact_path: Path,
    log_path: Path,
    seed: int,
    model_id: str,
    failures: list[str],
    checks: dict[str, Any] | None = None,
    device: dict[str, object] | None = None,
    peak_vram_mb: float | None = None,
) -> dict[str, object]:
    checks = checks or {"model_id": model_id}
    payload = gate_payload(
        gate="tiny_model",
        command=TINY_MODEL_GATE_COMMAND,
        seed=seed,
        formula_variant=SBA_FORMULA_VARIANT,
        mask_policy="model_attention_mask_valid_tokens",
        dtype_policy="nf4_bfloat16_compute",
        checks=checks,
        failures=failures,
        log_path=str(log_path),
        device=device,
        peak_vram_mb=peak_vram_mb,
    )
    write_json(artifact_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"Tiny model gate valid={payload['valid']}\nfailures={payload['failure_reasons']}\nchecks={checks}\n",
        encoding="utf-8",
    )
    return payload


def run_tiny_model_gate(
    model_id: str,
    artifact_path: Path,
    log_path: Path,
    vram_threshold_mb: float,
) -> dict[str, object]:
    seed = 20260518
    torch.manual_seed(seed)
    failures: list[str] = []
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable for tiny patched-model gate")
        payload = gate_payload(
            gate="tiny_model",
            command=TINY_MODEL_GATE_COMMAND,
            seed=seed,
            formula_variant=SBA_FORMULA_VARIANT,
            mask_policy="model_attention_mask_valid_tokens",
            dtype_policy="nf4_bfloat16_compute",
            checks={"model_id": model_id},
            failures=failures,
            log_path=str(log_path),
            device={"type": "cuda", "available": False},
        )
        write_json(artifact_path, payload)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("CUDA unavailable\n", encoding="utf-8")
        return payload

    device = torch.device("cuda")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as exc:
        return _write_invalid_gate_artifact(
            artifact_path=artifact_path,
            log_path=log_path,
            seed=seed,
            model_id=model_id,
            failures=[_exception_failure("cuda setup", exc)],
            device=_cuda_device(device),
        )
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    except Exception as exc:
        return _write_invalid_gate_artifact(
            artifact_path=artifact_path,
            log_path=log_path,
            seed=seed,
            model_id=model_id,
            failures=[_exception_failure("model-load", exc)],
            device=_cuda_device(device),
        )
    try:
        patch_report = apply_sba_patch(model)
    except Exception as exc:
        return _write_invalid_gate_artifact(
            artifact_path=artifact_path,
            log_path=log_path,
            seed=seed,
            model_id=model_id,
            failures=[_exception_failure("patch", exc)],
            checks={"model_id": model_id, "patch_attempted": True},
            device=_cuda_device(device),
        )
    try:
        encoded = tokenizer("SBA validation tiny forward backward check", return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        model.train()
        outputs = model(**encoded, labels=labels)
        loss = outputs.loss
        forward_finite = bool(torch.isfinite(loss).all().item())
    except Exception as exc:
        return _write_invalid_gate_artifact(
            artifact_path=artifact_path,
            log_path=log_path,
            seed=seed,
            model_id=model_id,
            failures=[_exception_failure("forward", exc)],
            checks={
                "model_id": model_id,
                "patched_module_count": patch_report["patched_module_count"],
                "patched_module_names": patch_report["patched_module_names"],
            },
            device=_cuda_device(device),
        )
    try:
        loss.backward()
        grad_finite_values: list[bool] = []
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_finite_values.append(bool(torch.isfinite(parameter.grad).all().item()))
        backward_finite = bool(grad_finite_values) and all(grad_finite_values)
    except Exception as exc:
        return _write_invalid_gate_artifact(
            artifact_path=artifact_path,
            log_path=log_path,
            seed=seed,
            model_id=model_id,
            failures=[_exception_failure("backward", exc)],
            checks={
                "model_id": model_id,
                "patched_module_count": patch_report["patched_module_count"],
                "patched_module_names": patch_report["patched_module_names"],
                "input_shape": list(encoded["input_ids"].shape),
                "finite_loss": forward_finite,
            },
            device=_cuda_device(device),
        )
    try:
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    except Exception as exc:
        return _write_invalid_gate_artifact(
            artifact_path=artifact_path,
            log_path=log_path,
            seed=seed,
            model_id=model_id,
            failures=[_exception_failure("vram-check", exc)],
            checks={
                "model_id": model_id,
                "patched_module_count": patch_report["patched_module_count"],
                "patched_module_names": patch_report["patched_module_names"],
                "input_shape": list(encoded["input_ids"].shape),
                "finite_loss": forward_finite,
                "finite_gradients": backward_finite,
                "no_silent_fallback": patch_report["patched_module_count"] > 0,
                "vram_threshold_mb": vram_threshold_mb,
            },
            device=_cuda_device(device),
        )
    if patch_report["patched_module_count"] <= 0:
        failures.append("no SBA patched modules were installed")
    if not forward_finite:
        failures.append("tiny patched-model loss is non-finite")
    if not backward_finite:
        failures.append("tiny patched-model gradients are non-finite")
    if peak_vram_mb > vram_threshold_mb:
        failures.append(f"tiny patched-model peak VRAM {peak_vram_mb:.0f} MB exceeded {vram_threshold_mb:.0f} MB")
    checks = {
        "model_id": model_id,
        "patched_module_count": patch_report["patched_module_count"],
        "patched_module_names": patch_report["patched_module_names"],
        "input_shape": list(encoded["input_ids"].shape),
        "finite_loss": forward_finite,
        "finite_gradients": backward_finite,
        "no_silent_fallback": patch_report["patched_module_count"] > 0,
        "vram_threshold_mb": vram_threshold_mb,
    }
    payload = gate_payload(
        gate="tiny_model",
        command=TINY_MODEL_GATE_COMMAND,
        seed=seed,
        formula_variant=SBA_FORMULA_VARIANT,
        mask_policy="model_attention_mask_valid_tokens",
        dtype_policy="nf4_bfloat16_compute",
        checks=checks,
        failures=failures,
        log_path=str(log_path),
        device={"type": "cuda", "available": True, "name": torch.cuda.get_device_name(device)},
        peak_vram_mb=peak_vram_mb,
    )
    write_json(artifact_path, payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"Tiny model gate valid={payload['valid']}\nchecks={checks}\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SBA tiny patched-model gate")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--enable-sba-patch", action="store_true", required=True)
    parser.add_argument("--out", default="experiments/sba_validation/tiny_model_gate.json")
    parser.add_argument("--log", default="experiments/sba_validation/tiny_model_gate.log")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    payload = run_tiny_model_gate(args.model_id, Path(args.out), Path(args.log), args.vram_threshold_mb)
    if not payload["valid"]:
        print(payload["failure_reasons"])
        raise SystemExit(2)
    print(f"Tiny model gate valid: {args.out}")


if __name__ == "__main__":
    main()
