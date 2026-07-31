import json

import pytest

from analysis.tuning_table import (
    aggregate_cells,
    load_tuning_records,
    render_grid,
    select_learning_rate,
    selections_by_method,
)


def make_tuning_record(
    method,
    learning_rate,
    seed,
    first_turn,
    *,
    rank=None,
    split="validation",
    validation_loss=0.5,
    sweep="hyperparameter_tuning",
):
    return {
        "run_id": f"{method}_{learning_rate}_{seed}",
        "method": method,
        "rank": rank,
        "learning_rate": learning_rate,
        "seed": seed,
        "exact_match": first_turn - 0.05,
        "exact_match_first_turn": first_turn,
        "validation_loss": validation_loss,
        "evaluation_split": split,
        "sweep": sweep,
        "status": "completed",
    }


def write_results(tmp_path, records):
    for index, record in enumerate(records):
        directory = tmp_path / f"run_{index:03d}"
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_cells_average_over_seeds():
    records = [
        make_tuning_record("full_ft", 2e-5, seed, score)
        for seed, score in zip((11, 22, 33), (0.36, 0.38, 0.34))
    ]
    cells = aggregate_cells(records)

    assert len(cells) == 1
    cell = cells[0]
    assert cell.label == "Full FT"
    assert cell.seeds == (11, 22, 33)
    assert cell.exact_match_percent == pytest.approx(36.0)
    assert cell.exact_match_std == pytest.approx(2.0)


def test_single_seed_cell_has_no_standard_deviation():
    cells = aggregate_cells([make_tuning_record("lora", 2e-4, 11, 0.4, rank=16)])
    assert cells[0].exact_match_std is None
    assert "±" not in render_grid(cells)


def test_duplicate_seed_is_rejected():
    records = [
        make_tuning_record("full_ft", 2e-5, 11, 0.36),
        make_tuning_record("full_ft", 2e-5, 11, 0.38),
    ]
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate_cells(records)


def test_selection_prefers_highest_exact_match_per_method():
    records = [
        make_tuning_record("full_ft", 2e-5, 11, 0.36),
        make_tuning_record("full_ft", 2e-4, 11, 0.28),
        make_tuning_record("lora", 2e-5, 11, 0.36, rank=16),
        make_tuning_record("lora", 2e-4, 11, 0.40, rank=16),
    ]
    selected = selections_by_method(aggregate_cells(records))

    assert selected["Full FT"].learning_rate == pytest.approx(2e-5)
    assert selected["LoRA-r16"].learning_rate == pytest.approx(2e-4)


def test_validation_loss_breaks_an_exact_match_tie():
    records = [
        make_tuning_record("full_ft", 2e-5, 11, 0.36, validation_loss=0.9),
        make_tuning_record("full_ft", 5e-5, 11, 0.36, validation_loss=0.7),
    ]
    chosen = select_learning_rate(aggregate_cells(records))
    assert chosen.learning_rate == pytest.approx(5e-5)


def test_grid_marks_the_selected_rate():
    records = [
        make_tuning_record("full_ft", 2e-5, 11, 0.36),
        make_tuning_record("full_ft", 2e-4, 11, 0.28),
    ]
    cells = aggregate_cells(records)

    assert "**36.0**" in render_grid(cells)
    assert "\\mathbf{36.0}" in render_grid(cells, latex=True)
    assert "10^{-4}" in render_grid(cells, latex=True)


def test_test_split_records_are_rejected(tmp_path):
    write_results(
        tmp_path,
        [make_tuning_record("full_ft", 2e-5, 11, 0.36, split="test")],
    )
    with pytest.raises(ValueError, match="must only use validation-split runs"):
        load_tuning_records(tmp_path)


def test_other_sweeps_and_unfinished_runs_are_ignored(tmp_path):
    records = [
        make_tuning_record("full_ft", 2e-5, 11, 0.36),
        make_tuning_record("full_ft", 2e-5, 22, 0.34, sweep="main", split="test"),
    ]
    records.append(
        {
            **make_tuning_record("full_ft", 5e-5, 11, 0.30),
            "status": "oom",
            "exact_match_first_turn": None,
        }
    )
    write_results(tmp_path, records)

    loaded = load_tuning_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["learning_rate"] == pytest.approx(2e-5)


def test_missing_tuning_field_is_reported(tmp_path):
    record = make_tuning_record("full_ft", 2e-5, 11, 0.36)
    record["exact_match_first_turn"] = None
    write_results(tmp_path, [record])

    with pytest.raises(ValueError, match="missing tuning fields"):
        load_tuning_records(tmp_path)
