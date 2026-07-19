"""Plot the memory-throughput-quality trade-off from aggregated results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _point_label(row: pd.Series) -> str:
    method = "Full FT" if row["method"] == "full_ft" else f"LoRA-r{int(row['rank'])}"
    quality = float(row["exact_match_mean"])
    return f"{method}\nEM={quality:.3f}"


def plot_tradeoff(
    frame: pd.DataFrame, output_path: str | Path, title: str | None = None
) -> Path:
    required = {
        "method",
        "rank",
        "peak_memory_gb_mean",
        "tokens_per_second_mean",
        "exact_match_mean",
        "pareto_efficient",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"aggregate frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("cannot plot an empty aggregate frame")

    figure, axis = plt.subplots(figsize=(9, 6))
    colors = frame["method"].map({"full_ft": "#d95f02", "lora": "#1b9e77"})
    markers = frame["pareto_efficient"].map({True: "*", False: "o"})
    for index, row in frame.iterrows():
        marker = markers.loc[index]
        axis.scatter(
            row["peak_memory_gb_mean"],
            row["tokens_per_second_mean"],
            c=colors.loc[index],
            marker=marker,
            s=180 if marker == "*" else 75,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )
        axis.annotate(
            _point_label(row),
            (row["peak_memory_gb_mean"], row["tokens_per_second_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Peak GPU memory (GB) — lower is better")
    axis.set_ylabel("Non-padding tokens/s — higher is better")
    axis.set_title(title or "LoRA vs Full Fine-tuning Trade-off")
    axis.grid(alpha=0.25)
    axis.text(
        0.99,
        0.01,
        "★ Pareto-efficient",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_rank_sweep(frame: pd.DataFrame, output_path: str | Path) -> Path:
    rank_frame = frame[(frame["method"] == "lora") & frame["rank"].notna()].sort_values(
        "rank"
    )
    if rank_frame.empty:
        raise ValueError("no LoRA rank rows available to plot")
    metrics = [
        ("peak_memory_gb_mean", "Peak memory (GB)"),
        ("tokens_per_second_mean", "Non-padding tokens/s"),
        ("exact_match_mean", "Exact match"),
        ("checkpoint_size_mb_mean", "Checkpoint size (MB)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (column, label) in zip(axes.flat, metrics):
        axis.plot(rank_frame["rank"], rank_frame[column], marker="o", color="#1b9e77")
        std_column = column.replace("_mean", "_std")
        if std_column in rank_frame and rank_frame[std_column].notna().any():
            axis.fill_between(
                rank_frame["rank"],
                rank_frame[column] - rank_frame[std_column].fillna(0),
                rank_frame[column] + rank_frame[std_column].fillna(0),
                alpha=0.15,
                color="#1b9e77",
            )
        axis.set_xlabel("LoRA rank")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.suptitle("LoRA Rank Sweep")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_sequence_length_sweep(frame: pd.DataFrame, output_path: str | Path) -> Path:
    if frame.empty:
        raise ValueError("no sequence-length rows available to plot")
    metrics = [
        ("peak_memory_gb_mean", "Peak memory (GB)"),
        ("tokens_per_second_mean", "Non-padding tokens/s"),
        ("exact_match_mean", "Exact match"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for method, group in frame.groupby("method"):
        group = group.sort_values("max_length")
        label = "Full FT" if method == "full_ft" else "LoRA-16"
        for axis, (column, ylabel) in zip(axes, metrics):
            axis.plot(group["max_length"], group[column], marker="o", label=label)
            axis.set_xlabel("Maximum sequence length")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Sequence-Length Sweep")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/tradeoff.png"))
    parser.add_argument("--title")
    parser.add_argument("--sweep", help="filter the aggregate to one sweep")
    parser.add_argument(
        "--kind",
        choices=("tradeoff", "rank", "sequence"),
        default="tradeoff",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.aggregate_csv)
    if args.sweep:
        frame = frame[frame["sweep"] == args.sweep]
    if args.kind == "rank":
        output = plot_rank_sweep(frame, args.output)
    elif args.kind == "sequence":
        output = plot_sequence_length_sweep(frame, args.output)
    else:
        output = plot_tradeoff(frame, args.output, args.title)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
