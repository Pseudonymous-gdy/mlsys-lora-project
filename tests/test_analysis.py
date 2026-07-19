import json

import pandas as pd
import pytest

from analysis.aggregate import (
    aggregate_results,
    load_result_attempts,
    load_result_records,
    summarize_batch_feasibility,
    validate_result_record,
    write_aggregates,
)
from analysis.plot_results import (
    plot_rank_sweep,
    plot_sequence_length_sweep,
    plot_tradeoff,
)


def make_record(method, rank, seed, memory, throughput, quality):
    return {
        "method": method,
        "rank": rank,
        "max_length": 512,
        "micro_batch_size": 1,
        "peak_memory_gb": memory,
        "tokens_per_second": throughput,
        "training_time_seconds": 100.0,
        "exact_match": quality,
        "trainable_parameters": 1_000_000 if method == "lora" else 800_000_000,
        "checkpoint_size_mb": 10.0 if method == "lora" else 1600.0,
        "seed": seed,
        "sweep": "final_seeds",
    }


def test_validation_rejects_bad_schema():
    record = make_record("lora", 16, 1, 5, 100, 0.5)
    del record["exact_match"]
    with pytest.raises(ValueError, match="missing"):
        validate_result_record(record)


def test_aggregation_statistics_pareto_and_plot(tmp_path):
    records = [
        make_record("full_ft", None, 1, 10, 100, 0.80),
        make_record("full_ft", None, 2, 10, 102, 0.82),
        make_record("lora", 16, 1, 5, 120, 0.80),
        make_record("lora", 16, 2, 5, 122, 0.82),
        make_record("lora", 8, 1, 4, 110, 0.79),
        make_record("lora", 8, 2, 4, 112, 0.79),
    ]
    aggregate = aggregate_results(records)
    assert len(aggregate) == 3
    full = aggregate[aggregate["method"] == "full_ft"].iloc[0]
    assert full["tokens_per_second_mean"] == pytest.approx(101)
    assert full["n_runs"] == 2
    assert not bool(full["pareto_efficient"])
    assert aggregate[aggregate["method"] == "lora"]["pareto_efficient"].all()

    csv_path = tmp_path / "aggregate.csv"
    json_path = tmp_path / "aggregate.json"
    write_aggregates(aggregate, csv_path, json_path)
    assert len(pd.read_csv(csv_path)) == 3
    assert len(json.loads(json_path.read_text(encoding="utf-8"))) == 3
    image_path = plot_tradeoff(aggregate, tmp_path / "tradeoff.png")
    assert image_path.stat().st_size > 1000


def test_rank_and_sequence_plots(tmp_path):
    records = []
    for rank in (4, 8, 16, 32):
        record = make_record(
            "lora", rank, 42, 4 + rank / 100, 120 - rank, 0.7 + rank / 1000
        )
        record["sweep"] = "rank"
        records.append(record)
    rank_frame = aggregate_results(records)
    assert plot_rank_sweep(rank_frame, tmp_path / "rank.png").stat().st_size > 1000

    records = []
    for method, rank in (("full_ft", None), ("lora", 16)):
        for length in (256, 512, 1024):
            record = make_record(method, rank, 42, length / 100, 1000 / length, 0.8)
            record["sweep"] = "sequence_length"
            record["max_length"] = length
            records.append(record)
    sequence_frame = aggregate_results(records)
    assert (
        plot_sequence_length_sweep(sequence_frame, tmp_path / "sequence.png")
        .stat()
        .st_size
        > 1000
    )


def test_result_loader_skips_oom_records(tmp_path):
    completed = make_record("lora", 16, 1, 5, 100, 0.5)
    completed["sweep"] = "max_batch"
    completed["micro_batch_size"] = 8
    oom = {
        "status": "oom",
        "sweep": "max_batch",
        "method": "lora",
        "rank": 16,
        "max_length": 512,
        "micro_batch_size": 16,
    }
    path = tmp_path / "runs.json"
    path.write_text(json.dumps([completed, oom]), encoding="utf-8")
    loaded = load_result_records([tmp_path])
    assert len(loaded) == 1
    assert loaded[0]["method"] == "lora"
    attempts = load_result_attempts([tmp_path])
    feasibility = summarize_batch_feasibility(attempts)
    assert set(feasibility["sweep"]) == {"max_batch"}
    max_batch = feasibility[feasibility["sweep"] == "max_batch"].iloc[0]
    assert max_batch["maximum_feasible_micro_batch"] == 8
    assert max_batch["first_oom_micro_batch"] == 16
