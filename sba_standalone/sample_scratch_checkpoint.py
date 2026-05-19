from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sba_standalone.run_scratch_compare import EOS_TOKEN, ScratchConfig, ScratchLM


DEFAULT_PROMPTS = [
    "Question:\nSolve for x: 2x + 5 = 17.\n\nAnswer:\n",
    "Question:\nWrite a Python function that returns the factorial of n.\n\nAnswer:\n",
    "Question:\nIf a list is sorted, how can binary search find a target value?\n\nAnswer:\n",
]


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    prompt_path = Path(path)
    if prompt_path.suffix.lower() == ".jsonl":
        prompts: list[str] = []
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompts.append(str(item.get("prompt", "")))
        return prompts
    return [chunk.strip() for chunk in prompt_path.read_text(encoding="utf-8").split("\n---\n") if chunk.strip()]


def encode(text: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([list(text.encode("utf-8", errors="ignore"))], dtype=torch.long, device=device)


def decode(tokens: list[int]) -> str:
    raw = bytes([token for token in tokens if 0 <= token < 256])
    return raw.decode("utf-8", errors="replace")


@torch.no_grad()
def sample_one(
    model: ScratchLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    model.eval()
    ids = encode(prompt, device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        window = ids[:, -model.config.block_size :]
        logits, _ = model(window)
        next_logits = logits[:, -1, :].float()
        next_logits[:, EOS_TOKEN] = -float("inf")
        if temperature <= 0:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                values, indices = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]), dim=-1)
                probs = F.softmax(values, dim=-1)
                chosen = torch.multinomial(probs, num_samples=1)
                next_id = int(indices.gather(-1, chosen).item())
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return decode(generated)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ScratchLM, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ScratchConfig(**checkpoint["scratch_config"])
    model = ScratchLM(config, checkpoint["attention"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a standalone SBA/softmax scratch checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    prompts = load_prompts(args.prompts)
    outputs = []
    for prompt in prompts:
        outputs.append(
            {
                "prompt": prompt,
                "completion": sample_one(
                    model=model,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                ),
            }
        )
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "attention": checkpoint.get("attention"),
        "scratch_config": checkpoint.get("scratch_config"),
        "samples": outputs,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
