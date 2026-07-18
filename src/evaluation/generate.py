"""Model-agnostic batched generation that produces auditable JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .exact_match import score_prediction


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def generate_predictions(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate continuations and immediately attach exact-match decisions.

    The function receives an already loaded model so the training owner can use
    the correct class for either Qwen checkpoint without duplicating evaluation
    logic.  Greedy decoding is the default to keep repeated runs deterministic.
    """

    import torch

    examples = list(examples)
    if not examples:
        raise ValueError("examples must not be empty")
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}
    if generation_kwargs:
        kwargs.update(dict(generation_kwargs))

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    device = _model_device(model)
    records: list[dict[str, Any]] = []
    model.eval()
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [str(example["prompt"]) for example in batch]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt")
            if device is not None:
                encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(**encoded, **kwargs)
            continuation_ids = generated[:, prompt_width:]
            completions = tokenizer.batch_decode(
                continuation_ids, skip_special_tokens=True
            )

            for example, completion in zip(batch, completions):
                reference = str(
                    example.get("reference_answer", example.get("gold_answer", ""))
                )
                score = score_prediction(completion, reference)
                records.append(
                    {
                        "example_id": str(example.get("example_id", len(records))),
                        "question": str(example.get("question", "")),
                        "prompt": str(example["prompt"]),
                        "prediction": completion,
                        "reference_answer": reference,
                        **score,
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
    return records


def save_predictions_jsonl(
    records: Iterable[Mapping[str, Any]], output_path: str | Path
) -> None:
    """Write one completion per line so interrupted evaluation remains inspectable."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
