from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_OUTPUT = "data/open_reason_code_mix.jsonl"
DEFAULT_MANIFEST = "data/open_reason_code_mix_manifest.json"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    split: str
    limit: int
    kind: str
    license_hint: str
    synthetic_prompt_prefix: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def squeeze_blank_lines(text: str, max_consecutive: int = 2) -> str:
    lines = text.splitlines()
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            out.append(line.rstrip())
        else:
            blanks += 1
            if blanks <= max_consecutive:
                out.append("")
    return "\n".join(out).strip()


def clamp_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "\n\n[truncated for length]", True


def first_message(conversations: Any, speaker: str) -> str:
    if not isinstance(conversations, list):
        return ""
    for item in conversations:
        if not isinstance(item, dict):
            continue
        if clean_text(item.get("from") or item.get("role")).lower() == speaker:
            return clean_text(item.get("value") or item.get("content"))
    return ""


def extract_numina(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    prompt = clean_text(row.get("problem"))
    response = clean_text(row.get("solution"))
    if not prompt or not response:
        messages = row.get("messages")
        prompt = prompt or first_message(messages, "user")
        response = response or first_message(messages, "assistant")
    return prompt, response, {"row_source": clean_text(row.get("source"))}


def extract_open_code_reasoning(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    prompt = clean_text(row.get("input"))
    response = clean_text(row.get("output") or row.get("solution"))
    metadata = {
        "id": clean_text(row.get("id")),
        "row_source": clean_text(row.get("source")),
        "row_license": clean_text(row.get("license")),
        "row_dataset": clean_text(row.get("dataset")),
        "row_difficulty": clean_text(row.get("difficulty")),
    }
    return prompt, response, {key: value for key, value in metadata.items() if value}


def extract_open_code_solution(row: dict[str, Any], synthetic_prompt_prefix: str) -> tuple[str, str, dict[str, Any]]:
    prompt = clean_text(row.get("input"))
    response = clean_text(row.get("output") or row.get("solution"))
    if prompt in {"", "-"}:
        prompt = f"{synthetic_prompt_prefix}\n\nSolve the programming problem and provide the final code solution."
    metadata = {
        "id": clean_text(row.get("id")),
        "row_source": clean_text(row.get("source")),
        "row_license": clean_text(row.get("license")),
        "row_dataset": clean_text(row.get("dataset")),
        "row_difficulty": clean_text(row.get("difficulty")),
        "row_index": clean_text(row.get("index")),
        "synthetic_prompt": "true" if synthetic_prompt_prefix else "false",
    }
    return prompt, response, {key: value for key, value in metadata.items() if value}


def extract_open_thoughts(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    conversations = row.get("conversations")
    prompt = first_message(conversations, "user")
    response = first_message(conversations, "assistant")
    return prompt, response, {"system_excerpt": clean_text(row.get("system"))[:240]}


EXTRACTORS: dict[str, Callable[[dict[str, Any]], tuple[str, str, dict[str, Any]]]] = {
    "numina": extract_numina,
    "open_code_reasoning": extract_open_code_reasoning,
    "open_thoughts": extract_open_thoughts,
}


def source_specs(args: argparse.Namespace) -> list[SourceSpec]:
    return [
        SourceSpec(
            name="numina_math_cot",
            dataset="AI-MO/NuminaMath-CoT",
            config="default",
            split="train",
            limit=args.numina_limit,
            kind="numina",
            license_hint="open Hugging Face dataset; verify current dataset card before redistribution",
        ),
        SourceSpec(
            name="nvidia_open_code_reasoning_split0",
            dataset="nvidia/OpenCodeReasoning",
            config="split_0",
            split="split_0",
            limit=args.open_code_split0_limit,
            kind="open_code_reasoning",
            license_hint="row-level license field preserved when present",
        ),
        SourceSpec(
            name="nvidia_open_code_reasoning_split1",
            dataset="nvidia/OpenCodeReasoning",
            config="split_1",
            split="split_1",
            limit=args.open_code_split1_limit,
            kind="open_code_solution",
            license_hint="row-level license field preserved when present",
            synthetic_prompt_prefix="This example is an open-source code-reasoning completion from nvidia/OpenCodeReasoning split_1 whose original prompt column is redacted as '-'.",
        ),
        SourceSpec(
            name="open_thoughts_114k",
            dataset="open-thoughts/OpenThoughts-114k",
            config="default",
            split="train",
            limit=args.open_thoughts_limit,
            kind="open_thoughts",
            license_hint="open Hugging Face dataset; verify current dataset card before redistribution",
        ),
    ]


def should_keep(prompt: str, response: str, min_prompt_chars: int, min_response_chars: int) -> tuple[bool, str]:
    if len(prompt) < min_prompt_chars:
        return False, "prompt_too_short"
    if len(response) < min_response_chars:
        return False, "response_too_short"
    lowered = f"{prompt}\n{response}".lower()
    if "as an ai language model" in lowered:
        return False, "generic_ai_disclaimer"
    return True, "ok"


def row_to_training_item(
    spec: SourceSpec,
    row: dict[str, Any],
    min_prompt_chars: int,
    min_response_chars: int,
    max_prompt_chars: int,
    max_response_chars: int,
) -> tuple[dict[str, Any] | None, str]:
    if spec.kind == "open_code_solution":
        prompt, response, metadata = extract_open_code_solution(row, spec.synthetic_prompt_prefix)
    else:
        prompt, response, metadata = EXTRACTORS[spec.kind](row)
    prompt = squeeze_blank_lines(clean_text(prompt))
    response = squeeze_blank_lines(clean_text(response))
    keep, reason = should_keep(prompt, response, min_prompt_chars, min_response_chars)
    if not keep:
        return None, reason
    prompt, prompt_truncated = clamp_text(prompt, max_prompt_chars)
    response, response_truncated = clamp_text(response, max_response_chars)
    item = {
        "source": spec.name,
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "prompt": prompt,
        "response": response,
        "metadata": {
            **metadata,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "prompt_truncated": prompt_truncated,
            "response_truncated": response_truncated,
        },
    }
    return item, "ok"


def iter_dataset(spec: SourceSpec, seed: int, shuffle_buffer: int):
    from datasets import load_dataset

    dataset = load_dataset(spec.dataset, spec.config, split=spec.split, streaming=True)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


def excerpt(text: str, limit: int = 500) -> str:
    text = clean_text(text).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def build_mix(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    specs = [spec for spec in source_specs(args) if spec.limit > 0]
    counts: Counter[str] = Counter()
    skipped: dict[str, Counter[str]] = {spec.name: Counter() for spec in specs}
    samples: dict[str, list[dict[str, Any]]] = {spec.name: [] for spec in specs}
    attempted: Counter[str] = Counter()
    truncated: Counter[str] = Counter()

    with output.open("w", encoding="utf-8") as handle:
        for index, spec in enumerate(specs):
            for row in iter_dataset(spec, seed=args.seed + index, shuffle_buffer=args.shuffle_buffer):
                if counts[spec.name] >= spec.limit:
                    break
                attempted[spec.name] += 1
                item, reason = row_to_training_item(
                    spec=spec,
                    row=dict(row),
                    min_prompt_chars=args.min_prompt_chars,
                    min_response_chars=args.min_response_chars,
                    max_prompt_chars=args.max_prompt_chars,
                    max_response_chars=args.max_response_chars,
                )
                if item is None:
                    skipped[spec.name][reason] += 1
                    if attempted[spec.name] >= spec.limit * args.max_attempt_multiplier:
                        break
                    continue
                if item["metadata"].get("prompt_truncated") or item["metadata"].get("response_truncated"):
                    truncated[spec.name] += 1
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                counts[spec.name] += 1
                if len(samples[spec.name]) < args.samples_per_source:
                    samples[spec.name].append(
                        {
                            "prompt_excerpt": excerpt(item["prompt"]),
                            "response_excerpt": excerpt(item["response"]),
                            "metadata": item["metadata"],
                        }
                    )

    file_bytes = output.stat().st_size if output.exists() else 0
    manifest = {
        "created_at": utc_now(),
        "output": str(output),
        "output_bytes": file_bytes,
        "total_examples": sum(counts.values()),
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "filters": {
            "min_prompt_chars": args.min_prompt_chars,
            "min_response_chars": args.min_response_chars,
            "max_prompt_chars": args.max_prompt_chars,
            "max_response_chars": args.max_response_chars,
            "max_attempt_multiplier": args.max_attempt_multiplier,
        },
        "sources": [
            {
                **asdict(spec),
                "written": counts[spec.name],
                "attempted": attempted[spec.name],
                "skipped": dict(skipped[spec.name]),
                "truncated": truncated[spec.name],
                "samples": samples[spec.name],
            }
            for spec in specs
        ],
        "notes": [
            "This script intentionally uses open Hugging Face datasets instead of repo-local synthetic task files.",
            "The output schema remains prompt/response JSONL so run_scratch_compare.py can consume it directly.",
            "Row-level license metadata is preserved for nvidia/OpenCodeReasoning when provided by the source row.",
            "nvidia/OpenCodeReasoning split_1 has '-' in the prompt column, so the builder uses an explicit synthetic prompt and preserves the row id/index metadata.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a verified open-source reasoning/code JSONL mix for SBA training")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--shuffle-buffer", type=int, default=20_000)
    parser.add_argument("--numina-limit", type=int, default=60_000)
    parser.add_argument("--open-code-split0-limit", type=int, default=45_000)
    parser.add_argument("--open-code-split1-limit", type=int, default=0)
    parser.add_argument("--open-thoughts-limit", type=int, default=35_000)
    parser.add_argument("--min-prompt-chars", type=int, default=40)
    parser.add_argument("--min-response-chars", type=int, default=120)
    parser.add_argument("--max-prompt-chars", type=int, default=12_000)
    parser.add_argument("--max-response-chars", type=int, default=16_000)
    parser.add_argument("--max-attempt-multiplier", type=int, default=4)
    parser.add_argument("--samples-per-source", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_mix(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("SBA_DATASET_BUILDER_NORMAL_EXIT") != "1":
        os._exit(0)


if __name__ == "__main__":
    main()
