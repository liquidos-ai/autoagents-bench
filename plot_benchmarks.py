#!/usr/bin/env python3
"""Create Seaborn visualisations from benchmark_results.json."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

BENCH_RESULTS_PATH = Path(os.getenv("BENCH_RESULTS_PATH", "benchmark_results.json"))


@dataclass
class BenchmarkConfig:
    total_requests: int
    concurrency: int
    model: str
    prompt_template: str


def load_config() -> BenchmarkConfig:
    config_path = Path(os.getenv("BENCH_CONFIG", "benchmark.yaml"))
    try:
        raw = config_path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark config not found at {config_path}") from exc

    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a YAML mapping")

    try:
        total_requests = int(data["total_requests"])
        concurrency = int(data["concurrency"])
        model = str(data.get("model", "gpt-4o-mini"))
        prompt_template = str(
            data.get(
                "prompt_template",
                "Calculate the average trip duration in minutes using the available tool.",
            )
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc

    return BenchmarkConfig(
        total_requests=total_requests,
        concurrency=concurrency,
        model=model,
        prompt_template=prompt_template,
    )


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark results file not found: {path}")

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            "Benchmark results JSON must be an object mapping benchmarks to metrics"
        )

    frame = pd.DataFrame.from_dict(raw, orient="index").reset_index()
    frame = frame.rename(columns={"index": "benchmark"})

    numeric_columns = [col for col in frame.columns if col != "benchmark"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def plot_metrics_grid(frame: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        {
            "column": "throughput_rps",
            "ylabel": "Requests per Second",
            "title": "Throughput (Higher is Better)",
            "palette": "Blues_d",
        },
        {
            "column": "average_latency_ms",
            "ylabel": "Latency (ms)",
            "title": "Average Latency (Lower is Better)",
            "palette": "Oranges_d",
        },
        {
            "column": "total_duration",
            "ylabel": "Duration (s)",
            "title": "Total Duration (Lower is Better)",
            "palette": "Greens_d",
        },
        {
            "column": "p95_latency_ms",
            "ylabel": "Latency (ms)",
            "title": "P95 Latency (Lower is Better)",
            "palette": "Purples_d",
        },
    ]

    cols = 2
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=frame,
            x="benchmark",
            y=metric["column"],
            hue="benchmark",
            palette=metric["palette"],
            legend=False,
            ax=ax,
        )
        ax.set_title(metric["title"], fontsize=12)
        ax.set_ylabel(metric["ylabel"])
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15, labelsize=10)

    for ax in axes[len(metrics) :]:
        ax.remove()

    total_requests = frame["total_requests"].iloc[0]
    config = load_config()
    model_name = config.model
    fig.suptitle(
        f"Benchmark Summary Model - {model_name} (Total Requests: {total_requests})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / f"benchmark_grid_{total_requests}.png", dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot benchmark metrics using Seaborn")
    parser.add_argument(
        "--input",
        type=Path,
        default=BENCH_RESULTS_PATH,
        help="Path to benchmark_results.json (defaults to BENCH_RESULTS_PATH or benchmark_results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where PNG files will be written",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk", palette="deep")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_results(args.input)
    frame = frame.sort_values("throughput_rps", ascending=False)

    plot_metrics_grid(frame, output_dir)


if __name__ == "__main__":
    main()
