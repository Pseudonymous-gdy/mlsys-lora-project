"""Robust, deterministic GSM8K answer extraction and exact match."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_NUMBER_TEXT = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
_FRACTION_TEXT = r"[-+]?\d+\s*/\s*\d+"
_NUMERIC_CANDIDATE_PATTERN = re.compile(f"(?:{_FRACTION_TEXT})|(?:{_NUMBER_TEXT})")
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_PHRASE_PATTERN = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:)?\s*([^\n]+)", re.IGNORECASE
)


def _last_numeric_candidate(text: str) -> str | None:
    # The fraction alternative comes first so ``6/8`` is one candidate rather
    # than two integers where the denominator would incorrectly win.
    candidates = list(_NUMERIC_CANDIDATE_PATTERN.finditer(text))
    if not candidates:
        return None
    return candidates[-1].group(0)


def normalize_numeric_answer(answer: Any) -> str | None:
    """Canonicalize a scalar number without floating-point rounding."""

    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    text = text.strip("`*_ ").rstrip(".!?;:").strip()
    text = text.replace("$", "").replace("£", "").replace("€", "")
    text = text.replace(",", "").replace(" ", "")

    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1]
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            value = Fraction(int(numerator), int(denominator))
            canonical = (
                str(value.numerator)
                if value.denominator == 1
                else f"{value.numerator}/{value.denominator}"
            )
        else:
            value = Decimal(text)
            if not value.is_finite():
                return None
            if value == 0:
                canonical = "0"
            else:
                canonical = format(value.normalize(), "f")
                if "." in canonical:
                    canonical = canonical.rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return f"{canonical}%" if is_percent else canonical


def extract_reference_answer(reference: str) -> str | None:
    """Extract the official answer after GSM8K's final ``####`` marker."""

    _, marker, suffix = str(reference).rpartition("####")
    if not marker:
        return normalize_numeric_answer(reference)
    candidate = _last_numeric_candidate(suffix)
    return normalize_numeric_answer(candidate)


def extract_prediction_answer(prediction: str) -> str | None:
    """Extract a final number using explicit signals before last-number fallback."""

    text = str(prediction)
    if "####" in text:
        candidate = _last_numeric_candidate(text.rsplit("####", 1)[1])
        if candidate is not None:
            return normalize_numeric_answer(candidate)

    boxed = list(_BOXED_PATTERN.finditer(text))
    if boxed:
        candidate = _last_numeric_candidate(boxed[-1].group(1))
        if candidate is not None:
            return normalize_numeric_answer(candidate)

    phrases = list(_FINAL_PHRASE_PATTERN.finditer(text))
    if phrases:
        candidate = _last_numeric_candidate(phrases[-1].group(1))
        if candidate is not None:
            return normalize_numeric_answer(candidate)

    return normalize_numeric_answer(_last_numeric_candidate(text))


def score_prediction(prediction: str, reference: str) -> dict[str, Any]:
    """Score one model completion against either a raw or canonical reference."""

    predicted_answer = extract_prediction_answer(prediction)
    gold_answer = extract_reference_answer(reference)
    correct = (
        predicted_answer is not None
        and gold_answer is not None
        and predicted_answer == gold_answer
    )
    return {
        "predicted_answer": predicted_answer,
        "gold_answer": gold_answer,
        "correct": bool(correct),
    }


def compute_exact_match(
    predictions: Sequence[str] | Iterable[str],
    references: Sequence[str] | Iterable[str],
) -> dict[str, Any]:
    """Return exact match in the closed interval [0, 1] plus per-item details."""

    predictions = list(predictions)
    references = list(references)
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("cannot evaluate an empty prediction set")

    details = [
        {
            "prediction": prediction,
            "reference": reference,
            **score_prediction(prediction, reference),
        }
        for prediction, reference in zip(predictions, references)
    ]
    correct = sum(int(item["correct"]) for item in details)
    return {
        "exact_match": correct / len(details),
        "correct": correct,
        "total": len(details),
        "unparseable": sum(item["predicted_answer"] is None for item in details),
        "details": details,
    }


def evaluate_jsonl(input_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    records: list[Mapping[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(value)
    predictions = [str(record["prediction"]) for record in records]
    references = [
        str(record.get("reference_answer", record.get("gold_answer", "")))
        for record in records
    ]
    summary = compute_exact_match(predictions, references)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions", type=Path, help="JSONL with prediction and reference_answer"
    )
    parser.add_argument("--output", type=Path, help="optional detailed JSON output")
    args = parser.parse_args()
    summary = evaluate_jsonl(args.predictions, args.output)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("exact_match", "correct", "total", "unparseable")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
