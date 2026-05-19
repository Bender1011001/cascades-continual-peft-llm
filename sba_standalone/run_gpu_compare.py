from __future__ import annotations

import argparse
import gc
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sba_standalone.artifacts import git_revision, read_json, utc_now, write_json
from sba_standalone.data import DEFAULT_DATA_FILES, build_dataloaders
from sba_standalone.patch import apply_sba_patch, collect_patch_report


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DENIED_MODEL_MARKERS = ("cascades", "heretic", "abliterated", "instruct", "extreme", "digital-twin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SBA vs softmax GPU comparison on a base LM")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-files", nargs="+", default=DEFAULT_DATA_FILES)
    parser.add_argument("--output-dir", default="sba_standalone/results/qwen25_1p5b_len256_rank16_seed43")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-limit-per-file", type=int, default=96)
    parser.add_argument("--eval-limit-per-file", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--vram-threshold-mb", type=float, default=7900.0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def validate_base_model_id(model_id: str) -> None:
    lowered = model_id.lower()
    blocked = [marker for marker in DENIED_MODEL_MARKERS if marker in lowered]
    if blocked:
        raise ValueError(
            f"model_id={model_id!r} looks fine-tuned or project-specific; blocked markers={blocked}. "
            "Use a regular base language model."
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def current_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated() / (1024 * 1024))


def enforce_vram(threshold_mb: float) -> None:
    peak = peak_vram_mb()
    if peak > threshold_mb:
        raise RuntimeError(f"peak VRAM {peak:.1f} MB exceeded threshold {threshold_mb:.1f} MB")


@torch.no_grad()
def evaluate_proxy_accuracy(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(**batch)
        total_loss += float(out.loss.item())
        batches += 1
    avg_loss = total_loss / max(batches, 1)
    return {"loss": avg_loss, "proxy_accuracy": math.exp(-avg_loss), "batches": float(batches)}


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model, tokenizer


def add_lora(model, args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
        )
    elif not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[target.strip() for target in args.lora_targets.split(",") if target.strip()],
    )
    return get_peft_model(model, config)


def train_one_arm(args: argparse.Namespace, arm_name: str, enable_sba: bool, output_dir: Path) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cuda")
    arm_dir = output_dir / arm_name
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    status: dict[str, Any] = {
        "arm": arm_name,
        "enable_sba": enable_sba,
        "started_at": utc_now(),
        "status": "running",
    }
    write_json(arm_dir / "status.json", status)
    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(args)
        model = add_lora(model, args)
        patch_report = None
        if enable_sba:
            patch_report = apply_sba_patch(model)
            write_json(arm_dir / "sba_patch_report.json", patch_report)
        else:
            patch_report = collect_patch_report(model)

        train_loader, eval_loader, data_manifest = build_dataloaders(
            tokenizer=tokenizer,
            data_files=args.data_files,
            max_length=args.max_length,
            train_limit_per_file=args.train_limit_per_file,
            eval_limit_per_file=args.eval_limit_per_file,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        write_json(arm_dir / "data_manifest.json", data_manifest)
        enforce_vram(args.vram_threshold_mb)

        model.train()
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps = 0
        micro_steps = 0
        for epoch in range(args.epochs):
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss / args.grad_accum_steps
                loss.backward()
                losses.append(float(out.loss.item()))
                micro_steps += 1
                if micro_steps % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                if micro_steps % 4 == 0:
                    enforce_vram(args.vram_threshold_mb)
            write_json(
                arm_dir / f"epoch_{epoch + 1}_progress.json",
                {
                    "epoch": epoch + 1,
                    "optimizer_steps": optimizer_steps,
                    "micro_steps": micro_steps,
                    "last_loss": losses[-1] if losses else None,
                    "current_vram_mb": current_vram_mb(),
                    "peak_vram_mb": peak_vram_mb(),
                },
            )

        if micro_steps % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        eval_metrics = evaluate_proxy_accuracy(model, eval_loader, device)
        elapsed = time.perf_counter() - started
        result = {
            "arm": arm_name,
            "valid": True,
            "status": "completed",
            "model_id": args.model_id,
            "base_model_only_input": True,
            "enable_sba": enable_sba,
            "patch_report": patch_report,
            "train_loss_first": losses[0] if losses else None,
            "train_loss_last": losses[-1] if losses else None,
            "train_loss_mean": float(sum(losses) / len(losses)) if losses else None,
            "optimizer_steps": optimizer_steps,
            "micro_steps": micro_steps,
            "eval": eval_metrics,
            "peak_vram_mb": peak_vram_mb(),
            "elapsed_seconds": elapsed,
            "finished_at": utc_now(),
        }
        write_json(arm_dir / "result.json", result)
        return result
    except Exception as exc:
        result = {
            "arm": arm_name,
            "valid": False,
            "status": "failed",
            "error": repr(exc),
            "model_id": args.model_id,
            "enable_sba": enable_sba,
            "peak_vram_mb": peak_vram_mb(),
            "finished_at": utc_now(),
        }
        write_json(arm_dir / "result.json", result)
        raise
    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def write_comparison(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    baseline = read_json(output_dir / "baseline" / "result.json")
    sba = read_json(output_dir / "sba" / "result.json")
    baseline_acc = float(baseline.get("eval", {}).get("proxy_accuracy", 0.0))
    sba_acc = float(sba.get("eval", {}).get("proxy_accuracy", 0.0))
    comparison = {
        "valid": bool(baseline.get("valid") and sba.get("valid")),
        "model_id": args.model_id,
        "base_model_only_input": True,
        "baseline_proxy_accuracy": baseline_acc,
        "sba_proxy_accuracy": sba_acc,
        "delta_proxy_accuracy": sba_acc - baseline_acc,
        "delta_proxy_accuracy_points": (sba_acc - baseline_acc) * 100.0,
        "baseline_peak_vram_mb": baseline.get("peak_vram_mb"),
        "sba_peak_vram_mb": sba.get("peak_vram_mb"),
        "sba_patched_module_count": sba.get("patch_report", {}).get("patched_module_count", 0),
        "git_revision": git_revision(),
        "finished_at": utc_now(),
    }
    write_json(output_dir / "comparison.json", comparison)
    return comparison


def main() -> None:
    args = parse_args()
    validate_base_model_id(args.model_id)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this standalone full-GPU comparison")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "config.json",
        {
            "model_id": args.model_id,
            "data_files": args.data_files,
            "seed": args.seed,
            "max_length": args.max_length,
            "train_limit_per_file": args.train_limit_per_file,
            "eval_limit_per_file": args.eval_limit_per_file,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": args.lora_targets,
            "vram_threshold_mb": args.vram_threshold_mb,
            "load_in_4bit": not args.no_4bit,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "git_revision": git_revision(),
            "started_at": utc_now(),
        },
    )
    train_one_arm(args, "baseline", enable_sba=False, output_dir=output_dir)
    train_one_arm(args, "sba", enable_sba=True, output_dir=output_dir)
    comparison = write_comparison(output_dir, args)
    print(f"Wrote {output_dir / 'comparison.json'}")
    print(comparison)


if __name__ == "__main__":
    main()
