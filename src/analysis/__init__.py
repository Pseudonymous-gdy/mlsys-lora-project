"""Experiment configuration, aggregation, Pareto analysis, and plotting."""

from .aggregate import (
    RESULT_METRICS,
    aggregate_results,
    load_result_attempts,
    load_result_records,
    mark_pareto_efficient,
    summarize_batch_feasibility,
    validate_result_record,
)

__all__ = [
    "RESULT_METRICS",
    "aggregate_results",
    "load_result_attempts",
    "load_result_records",
    "mark_pareto_efficient",
    "summarize_batch_feasibility",
    "validate_result_record",
]
