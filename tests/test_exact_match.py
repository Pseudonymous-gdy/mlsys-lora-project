import json

import pytest

from evaluation.exact_match import (
    compute_exact_match,
    evaluate_jsonl,
    extract_prediction_answer,
    extract_reference_answer,
    normalize_numeric_answer,
    score_prediction,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,200", "1200"),
        ("42.0", "42"),
        ("-0.50", "-0.5"),
        ("6/8", "3/4"),
        ("50%", "50%"),
        (None, None),
    ],
)
def test_numeric_normalization(raw, expected):
    assert normalize_numeric_answer(raw) == expected


def test_extraction_prefers_explicit_final_answer_markers():
    assert extract_reference_answer("work 10 + 32 = 42\n#### 42") == "42"
    assert extract_prediction_answer("I considered 41. Final answer: 42") == "42"
    assert extract_prediction_answer(r"Therefore \boxed{1,200}.") == "1200"
    assert extract_prediction_answer("work says 41\n#### 42") == "42"
    assert extract_prediction_answer("work\n#### 6/8") == "3/4"
    assert extract_prediction_answer("no numeric answer") is None


def test_score_and_exact_match_are_deterministic():
    assert score_prediction("work\n#### 42", "reason\n#### 42")["correct"] is True
    assert score_prediction("work\n#### 41", "reason\n#### 42")["correct"] is False
    result = compute_exact_match(
        ["#### 42", "#### 41", "I cannot answer"],
        ["reason\n#### 42", "reason\n#### 42", "reason\n#### 7"],
    )
    assert result["exact_match"] == pytest.approx(1 / 3)
    assert result["correct"] == 1
    assert result["unparseable"] == 1


def test_exact_match_rejects_mismatched_or_empty_inputs():
    with pytest.raises(ValueError, match="same length"):
        compute_exact_match(["1"], [])
    with pytest.raises(ValueError, match="empty"):
        compute_exact_match([], [])


def test_jsonl_evaluation(tmp_path):
    input_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "summary.json"
    rows = [
        {"prediction": "#### 42", "reference_answer": "work\n#### 42"},
        {"prediction": "#### 9", "gold_answer": "10"},
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = evaluate_jsonl(input_path, output_path)
    assert summary["exact_match"] == 0.5
    assert json.loads(output_path.read_text(encoding="utf-8"))["correct"] == 1
