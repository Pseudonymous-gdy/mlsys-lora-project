import torch

from evaluation.generate import generate_predictions, save_predictions_jsonl


class FakeGenerationTokenizer:
    padding_side = "right"

    def __call__(self, prompts, padding=True, return_tensors="pt"):
        assert padding is True and return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2] for _ in prompts], dtype=torch.long),
            "attention_mask": torch.ones((len(prompts), 2), dtype=torch.long),
        }

    def batch_decode(self, continuation_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [
            "#### 42" if row[0].item() == 42 else "#### 41" for row in continuation_ids
        ]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["do_sample"] is False
        assert attention_mask.shape == input_ids.shape
        continuation = torch.tensor(
            [[42], [41]], dtype=torch.long, device=input_ids.device
        )
        return torch.cat([input_ids, continuation[: len(input_ids)]], dim=1)


def test_generation_scores_and_serializes_records(tmp_path):
    tokenizer = FakeGenerationTokenizer()
    examples = [
        {"example_id": "a", "prompt": "q1", "reference_answer": "#### 42"},
        {"example_id": "b", "prompt": "q2", "reference_answer": "#### 42"},
    ]
    records = generate_predictions(FakeModel(), tokenizer, examples, batch_size=2)
    assert [record["correct"] for record in records] == [True, False]
    assert tokenizer.padding_side == "right"
    output = tmp_path / "predictions.jsonl"
    save_predictions_jsonl(records, output)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
