"""Select a learning rate per method from the common-grid validation sweep.

Selection follows the reported protocol: mean validation first-turn exact match
over the selection seeds, with mean validation loss as the tie-breaker. Runs
scored on any split other than ``validation`` are rejected so a tuning table can
never be built from test-set numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TUNING_SWEEP = "hyperparameter_tuning"
# Full FT legitimately reports rank=None, so presence and value are checked
# separately.
REQUIRED_KEYS = ("rank",)
REQUIRED_VALUES = (
    "method",
    "learning_rate",
    "seed",
    "exact_match_first_turn",
    "evaluation_split",
)


@dataclass(frozen=True)
class Cell:
    """One aggregated (method, rank, learning rate) cell of the sweep."""

    method: str
    rank: int | None
    learning_rate: float
    seeds: tuple[int, ...]
    exact_match_percent: float
    exact_match_std: float | None
    validation_loss: float | None

    @property
    def label(self) -> str:
        if self.method == "full_ft":
            return "Full FT"
        return f"LoRA-r{self.rank}"


def load_tuning_records(results_dir: Path) -> list[dict[str, Any]]:
    """Collect completed validation-split records of the tuning sweep."""

    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record.get("sweep")) != TUNING_SWEEP:
            continue
        if str(record.get("status")) != "completed":
            continue

        missing = [field for field in REQUIRED_KEYS if field not in record]
        missing += [
            field for field in REQUIRED_VALUES if record.get(field) is None
        ]
        if missing:
            raise ValueError(f"{path} is missing tuning fields: {missing}")

        split = str(record["evaluation_split"])
        if split != "validation":
            raise ValueError(
                f"{path} was scored on the '{split}' split; the tuning table "
                "must only use validation-split runs"
            )

        record["source_file"] = str(path)
        records.append(record)
    return records


def aggregate_cells(records: Sequence[Mapping[str, Any]]) -> list[Cell]:
    """Average each (method, rank, learning rate) cell over its seeds."""

    if not records:
        raise ValueError("no completed tuning records found")

    grouped: dict[tuple[str, int | None, float], list[Mapping[str, Any]]] = {}
    for record in records:
        rank = record["rank"]
        key = (
            str(record["method"]),
            None if rank is None else int(rank),
            float(record["learning_rate"]),
        )
        grouped.setdefault(key, []).append(record)

    cells: list[Cell] = []
    for (method, rank, learning_rate), group in grouped.items():
        seeds = sorted(int(item["seed"]) for item in group)
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"{method} rank={rank} lr={learning_rate} has duplicate seeds: {seeds}"
            )

        scores = [100.0 * float(item["exact_match_first_turn"]) for item in group]
        losses = [
            float(item["validation_loss"])
            for item in group
            if item.get("validation_loss") is not None
        ]

        cells.append(
            Cell(
                method=method,
                rank=rank,
                learning_rate=learning_rate,
                seeds=tuple(seeds),
                exact_match_percent=statistics.mean(scores),
                exact_match_std=(
                    statistics.stdev(scores) if len(scores) > 1 else None
                ),
                validation_loss=(
                    statistics.mean(losses) if len(losses) == len(group) else None
                ),
            )
        )

    return sorted(
        cells,
        key=lambda cell: (cell.method, cell.rank or 0, cell.learning_rate),
    )


def select_learning_rate(cells: Iterable[Cell]) -> Cell:
    """Highest mean exact match; lower mean validation loss breaks a tie."""

    candidates = list(cells)
    if not candidates:
        raise ValueError("cannot select from an empty cell list")

    best = max(candidates, key=lambda cell: cell.exact_match_percent)
    tied = [
        cell
        for cell in candidates
        if cell.exact_match_percent == best.exact_match_percent
    ]
    if len(tied) == 1:
        return best

    resolvable = [cell for cell in tied if cell.validation_loss is not None]
    if len(resolvable) != len(tied):
        # Without a loss for every tied cell the tie cannot be broken as
        # documented, so fall back to the smallest rate for determinism.
        return min(tied, key=lambda cell: cell.learning_rate)
    return min(resolvable, key=lambda cell: cell.validation_loss)


def selections_by_method(cells: Sequence[Cell]) -> dict[str, Cell]:
    """Select one rate per method label present in the sweep."""

    by_label: dict[str, list[Cell]] = {}
    for cell in cells:
        by_label.setdefault(cell.label, []).append(cell)
    return {
        label: select_learning_rate(group) for label, group in sorted(by_label.items())
    }


def _format_score(cell: Cell, *, bold: bool, latex: bool) -> str:
    if cell.exact_match_std is None:
        body = f"{cell.exact_match_percent:.1f}"
    else:
        separator = " \\pm " if latex else " ± "
        body = f"{cell.exact_match_percent:.1f}{separator}{cell.exact_match_std:.1f}"
    if latex:
        return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"
    return f"**{body}**" if bold else body


def _rate_label(learning_rate: float, *, latex: bool) -> str:
    mantissa, exponent = f"{learning_rate:.0e}".split("e")
    exponent = str(int(exponent))
    if not latex:
        return f"{mantissa}e{exponent}"
    if mantissa == "1":
        return f"$10^{{{exponent}}}$"
    return f"${mantissa}\\times10^{{{exponent}}}$"


def render_grid(cells: Sequence[Cell], *, latex: bool = False) -> str:
    """Render the Table 1 grid of methods against learning rates."""

    rates = sorted({cell.learning_rate for cell in cells})
    selected = selections_by_method(cells)
    labels = sorted({cell.label for cell in cells})
    lookup = {(cell.label, cell.learning_rate): cell for cell in cells}

    header = [_rate_label(rate, latex=latex) for rate in rates]
    lines: list[str] = []

    if latex:
        lines.append("\\begin{tabular}{l" + "c" * len(rates) + "}")
        lines.append("\\toprule")
        lines.append("& " + " & ".join(header) + " \\\\")
        lines.append("\\midrule")
    else:
        lines.append("| Method | " + " | ".join(header) + " |")
        lines.append("|---|" + "---|" * len(rates))

    for label in labels:
        row: list[str] = []
        for rate in rates:
            cell = lookup.get((label, rate))
            if cell is None:
                row.append("--" if latex else "—")
                continue
            row.append(
                _format_score(
                    cell,
                    bold=selected[label].learning_rate == rate,
                    latex=latex,
                )
            )
        if latex:
            lines.append(f"{label} & " + " & ".join(row) + " \\\\")
        else:
            lines.append(f"| {label} | " + " | ".join(row) + " |")

    if latex:
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    return "\n".join(lines)


def render_long_table(cells: Sequence[Cell]) -> str:
    """Render the per-rate appendix table including the tie-breaker column."""

    selected = selections_by_method(cells)
    lines = [
        "| Method | AdamW learning rate | Validation first-turn EM (%) "
        "| Validation loss | Seeds |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        loss = (
            "—" if cell.validation_loss is None else f"{cell.validation_loss:.4f}"
        )
        seeds = ", ".join(str(seed) for seed in cell.seeds)
        lines.append(
            f"| {cell.label} | {_rate_label(cell.learning_rate, latex=False)} "
            f"| {_format_score(cell, bold=selected[cell.label].learning_rate == cell.learning_rate, latex=False)} "
            f"| {loss} | {seeds} |"
        )
    return "\n".join(lines)


def build_report(cells: Sequence[Cell]) -> str:
    selected = selections_by_method(cells)
    seed_counts = sorted({len(cell.seeds) for cell in cells})

    sections = [
        "Common-grid validation sweep "
        f"({len(cells)} cells, {sum(len(cell.seeds) for cell in cells)} runs, "
        f"seeds per cell: {seed_counts})",
        "",
        render_grid(cells),
        "",
        "Per-rate detail",
        "",
        render_long_table(cells),
        "",
        "Selected learning rates",
        "",
    ]
    for label, cell in selected.items():
        std = (
            ""
            if cell.exact_match_std is None
            else f" ± {cell.exact_match_std:.1f}"
        )
        sections.append(
            f"  {label}: {_rate_label(cell.learning_rate, latex=False)} "
            f"({cell.exact_match_percent:.1f}{std} %)"
        )
    sections.extend(["", "LaTeX", "", render_grid(cells, latex=True)])
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=Path("results"))
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional machine-readable summary output",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="optional report output",
    )
    args = parser.parse_args()

    records = load_tuning_records(args.results)
    cells = aggregate_cells(records)
    report = build_report(cells)
    print(report)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report + "\n", encoding="utf-8")

    if args.json is not None:
        selected = selections_by_method(cells)
        payload = {
            "cells": [
                {
                    "method": cell.method,
                    "rank": cell.rank,
                    "learning_rate": cell.learning_rate,
                    "seeds": list(cell.seeds),
                    "exact_match_first_turn_percent_mean": cell.exact_match_percent,
                    "exact_match_first_turn_percent_std": cell.exact_match_std,
                    "validation_loss_mean": cell.validation_loss,
                }
                for cell in cells
            ],
            "selected": {
                label: {
                    "learning_rate": cell.learning_rate,
                    "exact_match_first_turn_percent_mean": cell.exact_match_percent,
                }
                for label, cell in selected.items()
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
