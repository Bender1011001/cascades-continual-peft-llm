from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_FILES = [
    "data/task0_logic_cot.jsonl",
    "data/task1_decomp_cot.jsonl",
    "data/task2_action_cot.jsonl",
]


@dataclass(frozen=True)
class Example:
    prompt: str
    response: str
    source: str


class SupervisedDataset(Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def load_examples(paths: list[str]) -> list[Example]:
    examples: list[Example] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                examples.append(Example(prompt=item["prompt"], response=item["response"], source=raw_path))
    return examples


def format_prompt(prompt: str) -> str:
    return f"Question:\n{prompt.strip()}\n\nAnswer:\n"


def encode_example(tokenizer, example: Example, max_length: int) -> dict[str, list[int]] | None:
    prompt_ids = tokenizer(format_prompt(example.prompt), add_special_tokens=False).input_ids
    response_text = example.response.strip()
    if tokenizer.eos_token:
        response_text = response_text + tokenizer.eos_token
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    if len(prompt_ids) >= max_length - 1:
        prompt_ids = prompt_ids[: max_length - 1]
    room = max_length - len(prompt_ids)
    if room <= 0:
        return None
    response_ids = response_ids[:room]
    if not response_ids:
        return None
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def split_and_encode(
    tokenizer,
    data_files: list[str],
    max_length: int,
    train_limit_per_file: int,
    eval_limit_per_file: int,
    seed: int,
) -> tuple[list[dict[str, list[int]]], list[dict[str, list[int]]]]:
    train_rows: list[dict[str, list[int]]] = []
    eval_rows: list[dict[str, list[int]]] = []
    rng = random.Random(seed)
    for data_file in data_files:
        examples = load_examples([data_file])
        rng.shuffle(examples)
        train_examples = examples[:train_limit_per_file]
        eval_examples = examples[train_limit_per_file : train_limit_per_file + eval_limit_per_file]
        for example in train_examples:
            row = encode_example(tokenizer, example, max_length)
            if row is not None:
                train_rows.append(row)
        for example in eval_examples:
            row = encode_example(tokenizer, example, max_length)
            if row is not None:
                eval_rows.append(row)
    if not train_rows or not eval_rows:
        raise ValueError("encoded dataset is empty; reduce limits or max_length constraints")
    return train_rows, eval_rows


def make_collate_fn(pad_token_id: int):
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for row in batch:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(row["attention_mask"] + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_dataloaders(
    tokenizer,
    data_files: list[str],
    max_length: int,
    train_limit_per_file: int,
    eval_limit_per_file: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    train_rows, eval_rows = split_and_encode(
        tokenizer=tokenizer,
        data_files=data_files,
        max_length=max_length,
        train_limit_per_file=train_limit_per_file,
        eval_limit_per_file=eval_limit_per_file,
        seed=seed,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collate = make_collate_fn(int(pad_token_id))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        SupervisedDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    eval_loader = DataLoader(SupervisedDataset(eval_rows), batch_size=batch_size, shuffle=False, collate_fn=collate)
    manifest = {
        "data_files": data_files,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "max_length": max_length,
        "train_limit_per_file": train_limit_per_file,
        "eval_limit_per_file": eval_limit_per_file,
    }
    return train_loader, eval_loader, manifest
