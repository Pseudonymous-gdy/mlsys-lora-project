from __future__ import annotations

import pytest

from data.gsm8k import (
    CausalLMCollator,
    GSM8KDataConfig,
    build_plain_prompt,
    render_prompt,
    split_reference_answer,
    tokenize_training_example,
)


class CharacterTokenizer:
    eos_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False):
        del truncation
        ids = [10 + ord(character) for character in text]
        return {"input_ids": ([2] + ids) if add_special_tokens else ids}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        return (
            "<system>"
            + messages[0]["content"]
            + "<user>"
            + messages[1]["content"]
            + "<assistant>"
        )


def test_plain_prompt_has_question_but_not_reference_answer():
    prompt = build_plain_prompt("What is 20 + 22?")
    assert "What is 20 + 22?" in prompt
    assert "#### 42" not in prompt


def test_data_revision_is_pinned_and_required():
    assert len(GSM8KDataConfig().dataset_revision) == 40
    with pytest.raises(ValueError, match="dataset_revision"):
        GSM8KDataConfig(dataset_revision="")


def test_chat_prompt_uses_generation_boundary():
    prompt = render_prompt("What is 20 + 22?", CharacterTokenizer(), "chat")
    assert prompt.endswith("<assistant>")
    assert "What is 20 + 22?" in prompt


def test_reference_answer_requires_marker():
    assert split_reference_answer("reasoning\n#### 42") == ("reasoning", "42")
    with pytest.raises(ValueError, match="missing"):
        split_reference_answer("answer without marker")


def test_tokenization_masks_prompt_and_preserves_final_answer_when_truncated():
    tokenizer = CharacterTokenizer()
    example = {
        "question": "What is 20 + 22?",
        "answer": ("long reasoning " * 30) + "\n#### 42",
    }
    prompt = build_plain_prompt(example["question"])
    mandatory = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    mandatory += len(tokenizer("#### 42", add_special_tokens=False)["input_ids"]) + 1
    config = GSM8KDataConfig(max_length=mandatory + 10, prompt_format="plain")
    tokenized = tokenize_training_example(example, tokenizer, config, example_id=7)

    assert tokenized["keep"] is True
    assert tokenized["was_truncated"] is True
    assert len(tokenized["input_ids"]) == config.max_length
    prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    assert tokenized["labels"][:prompt_length] == [-100] * prompt_length
    expected_final = tokenizer("#### 42", add_special_tokens=False)["input_ids"]
    assert tokenized["input_ids"][-(len(expected_final) + 1) : -1] == expected_final
    assert tokenized["input_ids"][-1] == tokenizer.eos_token_id


def test_example_is_filtered_if_question_and_final_answer_cannot_fit():
    tokenized = tokenize_training_example(
        {"question": "A very long question", "answer": "reason\n#### 42"},
        CharacterTokenizer(),
        GSM8KDataConfig(max_length=5, prompt_format="plain"),
    )
    assert tokenized["keep"] is False
    assert tokenized["input_ids"] == []


def test_collator_right_pads_and_reports_real_tokens():
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    batch = CausalLMCollator(pad_token_id=0, pad_to_multiple_of=4)(features)
    assert tuple(batch["input_ids"].shape) == (2, 4)
    assert batch["input_ids"][1].tolist() == [4, 5, 0, 0]
    assert batch["labels"][1].tolist() == [-100, 5, -100, -100]
    assert batch["num_non_padding_tokens"].item() == 5
