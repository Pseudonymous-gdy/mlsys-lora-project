#!/usr/bin/env python3
"""Run a real GSM8K + tokenizer smoke test without downloading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from transformers import AutoTokenizer

    from data.gsm8k import GSM8KDataConfig, load_gsm8k_splits, tokenize_training_example
    from evaluation.exact_match import extract_reference_answer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    config = GSM8KDataConfig(seed=args.seed, max_length=args.max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    splits = load_gsm8k_splits(config)
    sample_count = min(args.samples, len(splits["train"]))
    tokenized = [
        tokenize_training_example(
            splits["train"][index], tokenizer, config, example_id=f"train-{index}"
        )
        for index in range(sample_count)
    ]
    kept = [example for example in tokenized if example["keep"]]
    if not kept:
        raise RuntimeError("smoke test filtered every sampled example")
    if not all(any(label != -100 for label in example["labels"]) for example in kept):
        raise RuntimeError("at least one kept example has no supervised answer tokens")
    if not all(
        extract_reference_answer(example["reference_answer"]) for example in kept
    ):
        raise RuntimeError("at least one reference answer cannot be parsed")

    train_questions = set(splits["train"]["question"])
    validation_questions = set(splits["validation"]["question"])
    test_questions = set(splits["test"]["question"])
    overlap = {
        "train_validation": len(train_questions & validation_questions),
        "train_test": len(train_questions & test_questions),
        "validation_test": len(validation_questions & test_questions),
    }
    summary = {
        "model_tokenizer": args.model,
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "sampled": sample_count,
        "kept": len(kept),
        "truncated": sum(example["was_truncated"] for example in kept),
        "min_tokens": min(example["num_non_padding_tokens"] for example in kept),
        "max_tokens": max(example["num_non_padding_tokens"] for example in kept),
        "question_overlap": overlap,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
