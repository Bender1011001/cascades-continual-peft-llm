from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import cascades.data as data
from cascades.eval import evaluate_accuracy
from experiments.sba_validation.artifact_utils import git_revision, write_json


REASONING3_TASKS = [
    "data/task0_gsm8k_cot.jsonl",
    "data/task1_arc_cot.jsonl",
    "data/task2_csqa_cot.jsonl",
]


def apply_reasoning3_manifest() -> None:
    data.TASK_FILES = list(REASONING3_TASKS)
    data.TASK_NAMES = {i: f"Task {i} ({Path(path).stem})" for i, path in enumerate(REASONING3_TASKS)}
    data.NUM_TASKS = len(REASONING3_TASKS)


def run_raw_qwen_baseline(
    model_id: str,
    max_length: int,
    seed: int,
    out: Path,
    vram_threshold_mb: float,
) -> dict[str, object]:
    apply_reasoning3_manifest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        payload = {
            "valid": False,
            "failure_reasons": ["CUDA unavailable"],
            "model_id": model_id,
            "max_length": max_length,
            "seed": seed,
            "git_revision": git_revision(),
        }
        write_json(out, payload)
        return payload

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")

    task_accs: list[float] = []
    for task_id in range(len(REASONING3_TASKS)):
        loader = data.prepare_data(tokenizer, task_id, base_seed=seed, max_length=max_length)
        acc = float(evaluate_accuracy(model, loader, device))
        task_accs.append(acc)
        print(f"Task {task_id}: {acc * 100:.2f}%")

    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
    payload = {
        "valid": peak_vram_mb <= vram_threshold_mb,
        "failure_reasons": [] if peak_vram_mb <= vram_threshold_mb else [f"peak VRAM {peak_vram_mb:.0f} MB exceeded threshold {vram_threshold_mb:.0f} MB"],
        "model_id": model_id,
        "task_suite": "reasoning3",
        "seed": seed,
        "max_length": max_length,
        "task_accs": task_accs,
        "avg_acc": float(sum(task_accs) / max(len(task_accs), 1)),
        "peak_vram_mb": peak_vram_mb,
        "git_revision": git_revision(),
    }
    write_json(out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw Qwen proxy accuracy on reasoning3 without CASCADES training")
    parser.add_argument("--model-id", default="p-e-w/Qwen3-4B-Instruct-2507-heretic")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--out", default="experiments/sba_validation/raw_qwen_short128.json")
    parser.add_argument("--vram-threshold-mb", type=float, default=7500.0)
    args = parser.parse_args()
    payload = run_raw_qwen_baseline(
        model_id=args.model_id,
        max_length=args.max_length,
        seed=args.seed,
        out=ROOT / args.out,
        vram_threshold_mb=args.vram_threshold_mb,
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    if not payload["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
