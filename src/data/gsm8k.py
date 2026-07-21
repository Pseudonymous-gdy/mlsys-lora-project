"""Deterministic GSM8K loading and completion-only tokenization.

The same functions must be used by full fine-tuning and LoRA.  This keeps the
examples, token order, padding policy, and supervised labels identical so that
method comparisons are not confounded by the input pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math solver. Show the reasoning clearly and finish with "
    'a separate line in the exact form "#### <answer>".'
)
DEFAULT_GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


@dataclass(frozen=True)
class GSM8KDataConfig:
    """Configuration for a reproducible GSM8K input pipeline."""

    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_revision: str = DEFAULT_GSM8K_REVISION
    validation_size: int = 500
    seed: int = 42
    max_length: int = 512
    prompt_format: str = "chat"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.validation_size <= 0:
            raise ValueError("validation_size must be positive")
        if not self.dataset_revision.strip():
            raise ValueError("dataset_revision must not be empty")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.prompt_format not in {"chat", "plain"}:
            raise ValueError("prompt_format must be 'chat' or 'plain'")


def build_plain_prompt(
    question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """Build the model-independent form used when no chat template is desired."""

    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    return f"{system_prompt}\n\nQuestion: {question}\nAnswer:\n"


def render_prompt(
    question: str,
    tokenizer: Any | None,
    prompt_format: str = "chat",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render one generation prompt without placing the reference answer in it."""

    if prompt_format == "plain":
        return build_plain_prompt(question, system_prompt)
    if prompt_format != "chat":
        raise ValueError("prompt_format must be 'chat' or 'plain'")
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("chat prompt_format requires a tokenizer with a chat template")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]
    # Rendering first is deliberate. Transformers 4 and 5 return different
    # objects for apply_chat_template(tokenize=True), whereas text is stable.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def split_reference_answer(answer: str) -> tuple[str, str]:
    """Split GSM8K reasoning from the canonical ``####`` answer suffix."""

    reasoning, marker, final_answer = answer.rpartition("####")
    if not marker or not final_answer.strip():
        raise ValueError("GSM8K answer is missing a non-empty '####' final answer")
    return reasoning.rstrip(), final_answer.strip()


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)
    input_ids = (
        encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    )
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("expected a single tokenized string")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def tokenize_training_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    config: GSM8KDataConfig,
    *,
    example_id: str | int | None = None,
) -> dict[str, Any]:
    """Tokenize one example while preserving its question and final answer.

    If the full chain of thought is too long, only the end of the reasoning is
    truncated.  The question, ``####`` final answer, and EOS token are never
    silently removed.  An example that cannot fit those mandatory parts is
    marked with ``keep=False`` for deterministic filtering by the caller.
    """

    question = str(example["question"]).strip()
    answer = str(example["answer"]).strip()
    prompt = render_prompt(
        question,
        tokenizer,
        prompt_format=config.prompt_format,
        system_prompt=config.system_prompt,
    )
    reasoning, final_answer = split_reference_answer(answer)

    prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=False)
    reasoning_text = f"{reasoning}\n" if reasoning else ""
    reasoning_ids = _token_ids(tokenizer, reasoning_text, add_special_tokens=False)
    final_text = f"#### {final_answer}"
    final_ids = _token_ids(tokenizer, final_text, add_special_tokens=False)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    mandatory_length = len(prompt_ids) + len(final_ids) + 1
    base = {
        "example_id": str(example_id) if example_id is not None else "",
        "question": question,
        "prompt": prompt,
        "reference_answer": answer,
        "gold_answer": final_answer,
    }
    if mandatory_length > config.max_length:
        return {
            **base,
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "num_non_padding_tokens": 0,
            "was_truncated": True,
            "keep": False,
        }

    reasoning_budget = config.max_length - mandatory_length
    kept_reasoning_ids = reasoning_ids[:reasoning_budget]
    was_truncated = len(kept_reasoning_ids) != len(reasoning_ids)
    target_ids = kept_reasoning_ids + final_ids + [int(eos_token_id)]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids.copy()

    return {
        **base,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "num_non_padding_tokens": len(input_ids),
        "was_truncated": was_truncated,
        "keep": True,
    }


def load_gsm8k_splits(config: GSM8KDataConfig) -> Any:
    """Load official splits and deterministically carve validation from train."""

    from datasets import DatasetDict, load_dataset

    raw = load_dataset(
        config.dataset_name,
        config.dataset_config,
        revision=config.dataset_revision,
    )
    if "train" not in raw or "test" not in raw:
        raise ValueError("GSM8K dataset must contain train and test splits")
    required_columns = {"question", "answer"}
    for split_name in ("train", "test"):
        missing = required_columns.difference(raw[split_name].column_names)
        if missing:
            raise ValueError(
                f"{split_name} split is missing columns: {sorted(missing)}"
            )
    if config.validation_size >= len(raw["train"]):
        raise ValueError("validation_size must be smaller than the train split")

    divided = raw["train"].train_test_split(
        test_size=config.validation_size,
        seed=config.seed,
        shuffle=True,
    )
    return DatasetDict(
        {
            "train": divided["train"],
            "validation": divided["test"],
            "test": raw["test"],
        }
    )


def _tokenize_split(
    dataset: Any, tokenizer: Any, config: GSM8KDataConfig, split: str
) -> Any:
    def transform(example: Mapping[str, Any], index: int) -> dict[str, Any]:
        return tokenize_training_example(
            example,
            tokenizer,
            config,
            example_id=f"{split}-{index}",
        )

    mapped = dataset.map(transform, with_indices=True, desc=f"Tokenizing GSM8K {split}")
    mapped = mapped.filter(
        lambda example: bool(example["keep"]), desc=f"Filtering {split}"
    )
    return mapped.remove_columns(["keep"])


def _format_evaluation_split(
    dataset: Any, tokenizer: Any, config: GSM8KDataConfig
) -> Any:
    def transform(example: Mapping[str, Any], index: int) -> dict[str, Any]:
        _, final_answer = split_reference_answer(str(example["answer"]))
        return {
            "example_id": f"test-{index}",
            "prompt": render_prompt(
                str(example["question"]),
                tokenizer,
                prompt_format=config.prompt_format,
                system_prompt=config.system_prompt,
            ),
            "reference_answer": str(example["answer"]),
            "gold_answer": final_answer,
        }

    return dataset.map(transform, with_indices=True, desc="Formatting GSM8K test")


def prepare_gsm8k_datasets(config: GSM8KDataConfig, tokenizer: Any) -> Any:
    """Return training/validation tensors plus leakage-free test prompts."""

    from datasets import DatasetDict

    raw = load_gsm8k_splits(config)
    return DatasetDict(
        {
            "train": _tokenize_split(raw["train"], tokenizer, config, "train"),
            "validation": _tokenize_split(
                raw["validation"], tokenizer, config, "validation"
            ),
            "test": _format_evaluation_split(raw["test"], tokenizer, config),
        }
    )


class CausalLMCollator:
    """Right-pad completion-only examples and expose real token counts."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8) -> None:
        if pad_token_id is None:
            raise ValueError("pad_token_id must be defined")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive or None")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty batch")
        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature, length in zip(features, lengths):
            padding = max_length - length
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * padding)
            attention_masks.append(list(feature["attention_mask"]) + [0] * padding)
            labels.append(list(feature["labels"]) + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "num_non_padding_tokens": torch.tensor(sum(lengths), dtype=torch.long),
        }
