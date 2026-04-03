import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .types import BenchmarkRun, DynamicResponse, OptimizationResult


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def ensure_result_dirs(root: Path) -> dict[str, Path]:
    directories = {
        "tables": root / "results/tables",
        "figures": root / "results/figures",
        "summary": root / "results/summary",
        "metadata": root / "results/metadata",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def write_csv(table: list[dict[str, Any]], destination: Path) -> Path:
    pd.DataFrame(table).to_csv(destination, index=False)
    return destination


def _plot_convergence(result: OptimizationResult, destination: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.history, linewidth=2)
    ax.set_title(f"{result.algorithm.upper()} convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def _plot_time_history(response: DynamicResponse, destination: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx in range(response.relative_displacements_m.shape[1]):
        ax.plot(
            response.time,
            response.relative_displacements_m[:, idx],
            label=f"Story {idx + 1}",
        )
    ax.set_title("Story displacement time histories")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative displacement (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def _plot_reduction_bars(
    table: list[dict[str, Any]], destination: Path, title: str
) -> Path:
    frame = pd.DataFrame(table)
    frame = frame[frame["story"] != "mean"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    stories = frame["story"].astype(int)
    for algorithm in ("pso", "woa", "hpw"):
        ax.plot(stories, frame[algorithm], marker="o", label=algorithm.upper())
    ax.set_title(title)
    ax.set_xlabel("Story")
    ax.set_ylabel("Reduction (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def write_manifest(root: Path, payload: dict[str, Any]) -> Path:
    destination = root / "results/metadata/run_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return destination


def write_report(root: Path, run: BenchmarkRun) -> Path:
    lines = [
        f"# Run Summary: {run.benchmark.name}",
        "",
        f"- backend: `{run.backend}`",
        f"- mode: `{run.mode}`",
    ]
    for note in run.notes:
        lines.append(f"- note: {note}")
    if run.optimizations:
        lines.append("")
        lines.append("## Optimizer results")
        for algorithm, result in run.optimizations.items():
            lines.append(
                f"- {algorithm.upper()}: best={result.best_value:.6f}, kd={result.best_position[0]:.4f}, cd={result.best_position[1]:.4f}, iterations={result.iterations}"
            )
    destination = root / "results/summary/report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def publish_run(root: Path, run: BenchmarkRun) -> dict[str, Path]:
    paths = ensure_result_dirs(root)
    generated: dict[str, Path] = {}
    for table_name, table in run.tables.items():
        generated[f"table:{table_name}"] = write_csv(
            table, paths["tables"] / f"{run.benchmark.name}_{table_name}.csv"
        )
    for algorithm, result in run.optimizations.items():
        generated[f"figure:{algorithm}_convergence"] = _plot_convergence(
            result,
            paths["figures"] / f"{run.benchmark.name}_{algorithm}_convergence.png",
        )
    if "table4" in run.tables:
        generated["figure:reduction"] = _plot_reduction_bars(
            run.tables["table4"],
            paths["figures"] / f"{run.benchmark.name}_reduction.png",
            "Percentage of displacement reduction",
        )
    if "table12" in run.tables:
        generated["figure:reduction"] = _plot_reduction_bars(
            run.tables["table12"],
            paths["figures"] / f"{run.benchmark.name}_reduction.png",
            "Percentage of displacement reduction",
        )
    if run.uncontrolled is not None:
        generated["figure:time_history_uncontrolled"] = _plot_time_history(
            run.uncontrolled,
            paths["figures"] / f"{run.benchmark.name}_uncontrolled_time_history.png",
        )
    for algorithm, response in run.controlled.items():
        generated[f"figure:{algorithm}_time_history"] = _plot_time_history(
            response,
            paths["figures"] / f"{run.benchmark.name}_{algorithm}_time_history.png",
        )
    generated["report"] = write_report(root, run)
    generated["manifest"] = write_manifest(
        root,
        {
            "benchmark": run.benchmark.name,
            "backend": run.backend,
            "mode": run.mode,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "notes": run.notes,
            "optimizations": {
                name: asdict(result) for name, result in run.optimizations.items()
            },
        },
    )
    return generated
