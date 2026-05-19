from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sba_standalone.artifacts import git_revision, utc_now, write_json


BYTE_VOCAB_SIZE = 257
EOS_TOKEN = 256


@dataclass(frozen=True)
class ScratchConfig:
    vocab_size: int = BYTE_VOCAB_SIZE
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    mlp_ratio: int = 4


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x.float()
        normed = work * torch.rsqrt(work.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(dtype=x.dtype)


def sba_weights(logits: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    work = logits.float()
    valid = causal_mask.to(device=logits.device)
    floor = torch.full_like(work, torch.finfo(work.dtype).min)
    zeros = torch.zeros_like(work)
    pos_logits = torch.where(valid, work, floor)
    neg_logits = torch.where(valid, -work, floor)
    row_max = torch.maximum(
        pos_logits.max(dim=-1, keepdim=True).values,
        neg_logits.max(dim=-1, keepdim=True).values,
    )
    e_pos = torch.where(valid, torch.exp(pos_logits - row_max), zeros)
    e_neg = torch.where(valid, torch.exp(neg_logits - row_max), zeros)
    denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work.dtype).tiny)
    weights = torch.where(valid, (e_pos - e_neg) / denom, zeros)
    return weights.to(dtype=logits.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ScratchConfig, attention: str) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if attention not in {"softmax", "sba"}:
            raise ValueError("attention must be softmax or sba")
        self.attention = attention
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attention == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            weights = sba_weights(logits, mask)
            y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, config: ScratchConfig, attention: str) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, attention)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class ScratchLM(nn.Module):
    def __init__(self, config: ScratchConfig, attention: str) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, attention) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss


def load_texts(data_files: list[str]) -> list[str]:
    texts: list[str] = []
    for raw_path in data_files:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = str(item.get("prompt", "")).strip()
                response = str(item.get("response", "")).strip()
                texts.append(f"Question:\n{prompt}\n\nAnswer:\n{response}\n")
    if not texts:
        raise ValueError("no training text loaded")
    return texts


def encode_texts(texts: list[str], repeat: int) -> torch.Tensor:
    tokens: list[int] = []
    for _ in range(repeat):
        for text in texts:
            tokens.extend(text.encode("utf-8", errors="ignore"))
            tokens.append(EOS_TOKEN)
    return torch.tensor(tokens, dtype=torch.long)


def make_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[start : start + block_size] for start in starts]).to(device, non_blocking=True)
    y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts]).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(
    model: ScratchLM,
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        x, y = make_batch(tokens, batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    loss_value = sum(losses) / len(losses)
    model.train()
    return {"loss": loss_value, "perplexity": math.exp(min(loss_value, 20.0))}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def should_save_checkpoint(step: int, final_step: int, args: argparse.Namespace) -> bool:
    if not getattr(args, "save_checkpoints", False):
        return False
    if step == final_step:
        return True
    interval = int(getattr(args, "checkpoint_every", 0) or 0)
    return interval > 0 and step % interval == 0


def checkpoint_path_for_step(output_dir: Path, attention: str, step: int, config: ScratchConfig) -> Path:
    filename = f"{attention}_step{step:06d}_{config.n_layer}l_{config.n_head}h_{config.n_embd}d.pt"
    return output_dir / attention / "checkpoints" / filename


def save_checkpoint(
    model: ScratchLM,
    output_dir: Path,
    attention: str,
    step: int,
    config: ScratchConfig,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> str:
    path = checkpoint_path_for_step(output_dir, attention, step, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "sba-standalone-scratch-lm-v1",
        "attention": attention,
        "step": step,
        "scratch_config": asdict(config),
        "args": vars(args),
        "metrics": metrics,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "saved_at": utc_now(),
    }
    torch.save(checkpoint, path)
    return str(path)


def train_arm(
    attention: str,
    config: ScratchConfig,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    arm_dir = output_dir / attention
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = ScratchLM(config, attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    parameter_count = count_parameters(model)
    started = time.perf_counter()
    metrics_path = arm_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_val = float("inf")
    last_train_loss = None
    saved_checkpoints: list[str] = []

    for step in range(1, args.steps + 1):
        x, y = make_batch(train_tokens, args.batch_size, config.block_size, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(x, y)
            scaled_loss = loss / args.grad_accum_steps
        scaled_loss.backward()
        if step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last_train_loss = float(loss.item())

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val = evaluate(model, val_tokens, args.eval_batch_size, config.block_size, args.eval_iters, device, amp_dtype)
            best_val = min(best_val, val["loss"])
            event = {
                "step": step,
                "train_loss": last_train_loss,
                "val_loss": val["loss"],
                "val_perplexity": val["perplexity"],
                "best_val_loss": best_val,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            write_json(arm_dir / "latest.json", event)
        if step < args.steps and should_save_checkpoint(step, args.steps, args):
            saved_checkpoints.append(
                save_checkpoint(
                    model=model,
                    output_dir=output_dir,
                    attention=attention,
                    step=step,
                    config=config,
                    args=args,
                    metrics={"train_loss": last_train_loss, "best_val_loss_seen": best_val},
                )
            )

    final_val = evaluate(model, val_tokens, args.eval_batch_size, config.block_size, args.eval_iters * 2, device, amp_dtype)
    if should_save_checkpoint(args.steps, args.steps, args):
        saved_checkpoints.append(
            save_checkpoint(
                model=model,
                output_dir=output_dir,
                attention=attention,
                step=args.steps,
                config=config,
                args=args,
                metrics={
                    "train_loss": last_train_loss,
                    "best_val_loss_seen": best_val,
                    "final_val_loss": final_val["loss"],
                    "final_val_perplexity": final_val["perplexity"],
                },
            )
        )
    result = {
        "attention": attention,
        "valid": bool(math.isfinite(final_val["loss"])),
        "parameter_count": parameter_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "block_size": config.block_size,
        "train_tokens": int(train_tokens.numel()),
        "val_tokens": int(val_tokens.numel()),
        "last_train_loss": last_train_loss,
        "final_val_loss": final_val["loss"],
        "final_val_perplexity": final_val["perplexity"],
        "best_val_loss_seen": best_val,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at": utc_now(),
        "saved_checkpoints": saved_checkpoints,
    }
    write_json(arm_dir / "result.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train matched from-scratch softmax and SBA byte-level LMs")
    parser.add_argument("--data-files", nargs="+", default=[str(path) for path in Path("data").glob("*.jsonl")])
    parser.add_argument("--output-dir", default="sba_standalone/results/h200_scratch_quick")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-iters", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--data-repeat", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--arms", choices=["both", "softmax", "sba"], default="both")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if args.dtype in {"auto", "bf16"} and torch.cuda.is_bf16_supported() else torch.float16
    config = ScratchConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = load_texts(args.data_files)
    tokens = encode_texts(texts, args.data_repeat)
    split = int(tokens.numel() * (1.0 - args.val_fraction))
    train_tokens = tokens[:split].pin_memory()
    val_tokens = tokens[split:].pin_memory()
    write_json(
        output_dir / "config.json",
        {
            "script": "run_scratch_compare.py",
            "scratch_config": asdict(config),
            "args": vars(args),
            "amp_dtype": str(amp_dtype),
            "cuda_device": torch.cuda.get_device_name(0),
            "git_revision": git_revision(),
            "text_count": len(texts),
            "total_tokens": int(tokens.numel()),
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "started_at": utc_now(),
        },
    )

    results: dict[str, Any] = {}
    if args.arms in {"both", "softmax"}:
        results["softmax"] = train_arm("softmax", config, train_tokens, val_tokens, args, output_dir, amp_dtype)
    if args.arms in {"both", "sba"}:
        results["sba"] = train_arm("sba", config, train_tokens, val_tokens, args, output_dir, amp_dtype)

    if "softmax" in results and "sba" in results:
        comparison = {
            "valid": bool(results["softmax"]["valid"] and results["sba"]["valid"]),
            "softmax_final_val_loss": results["softmax"]["final_val_loss"],
            "sba_final_val_loss": results["sba"]["final_val_loss"],
            "delta_val_loss_sba_minus_softmax": results["sba"]["final_val_loss"] - results["softmax"]["final_val_loss"],
            "softmax_best_val_loss": results["softmax"]["best_val_loss_seen"],
            "sba_best_val_loss": results["sba"]["best_val_loss_seen"],
            "delta_best_val_loss_sba_minus_softmax": results["sba"]["best_val_loss_seen"] - results["softmax"]["best_val_loss_seen"],
            "softmax_peak_vram_mb": results["softmax"]["peak_vram_mb"],
            "sba_peak_vram_mb": results["sba"]["peak_vram_mb"],
            "parameter_count": results["softmax"]["parameter_count"],
            "finished_at": utc_now(),
        }
        write_json(output_dir / "comparison.json", comparison)
        print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
