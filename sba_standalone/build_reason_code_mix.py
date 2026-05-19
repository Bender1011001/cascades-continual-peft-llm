from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "data/hf_reason_code_mix.jsonl"


def clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def write_example(handle, source: str, prompt: Any, response: Any) -> bool:
    prompt_text = clean_text(prompt)
    response_text = clean_text(response)
    if len(prompt_text) < 20 or len(response_text) < 20:
        return False
    handle.write(
        json.dumps({"source": source, "prompt": prompt_text, "response": response_text}, ensure_ascii=False) + "\n"
    )
    return True


def build_mix(output: Path, openr1_limit: int, magicoder_limit: int, codealpaca_limit: int) -> dict[str, Any]:
    from datasets import load_dataset

    counts = {"openr1_math": 0, "magicoder": 0, "codealpaca": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True):
            if counts["openr1_math"] >= openr1_limit:
                break
            response = clean_text(row.get("solution"))
            answer = clean_text(row.get("answer"))
            if answer:
                response = f"{response}\n\nFinal answer: {answer}"
            if write_example(handle, "openr1_math", row.get("problem"), response):
                counts["openr1_math"] += 1

        for row in load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True):
            if counts["magicoder"] >= magicoder_limit:
                break
            if write_example(handle, "magicoder", row.get("problem"), row.get("solution")):
                counts["magicoder"] += 1

        for row in load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train", streaming=True):
            if counts["codealpaca"] >= codealpaca_limit:
                break
            if write_example(handle, "codealpaca", row.get("prompt"), row.get("completion")):
                counts["codealpaca"] += 1

    return {"output": str(output), "counts": counts, "total": sum(counts.values())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reasoning/code JSONL mix for standalone SBA training")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--openr1-limit", type=int, default=40_000)
    parser.add_argument("--magicoder-limit", type=int, default=30_000)
    parser.add_argument("--codealpaca-limit", type=int, default=15_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_mix(Path(args.output), args.openr1_limit, args.magicoder_limit, args.codealpaca_limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
