from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

import numpy as np
from tqdm.auto import tqdm

from .analysis import top_floor_displacement_ratio
from .benchmarks import get_benchmark, with_tmd_mass
from .io import load_record
from .models import build_controlled_mck, build_uncontrolled_mck
from .opensees_model import analyze_with_backend
from .optimizers import OptimizerConfig, run_optimizer
from .reference import get_reference_params
from .reporting import publish_run
from .state_space import state_space_objective
from .types import (
    BenchmarkRun,
    BuildingConfig,
    DynamicResponse,
    OptimizationResult,
    TMDParameters,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_algorithm_config() -> dict[str, dict[str, float | int]]:
    with (ROOT / "configs/algorithms.toml").open("rb") as handle:
        return tomllib.load(handle)


def _optimizer_config(
    algorithm: str, profile: str, show_progress: bool = False, progress_label: str = ""
) -> OptimizerConfig:
    payload = _load_algorithm_config()
    profiles = payload["profiles"]
    try:
        profile_payload = profiles[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown optimization profile: {profile}") from exc
    global_cfg = profile_payload["global"]
    local = profile_payload[algorithm]
    merged = {
        **global_cfg,
        **local,
        "show_progress": show_progress,
        "progress_label": progress_label,
    }
    return OptimizerConfig(**merged)


def _bounds(config: BuildingConfig) -> np.ndarray:
    return np.array(
        [
            [config.stiffness_bounds_kn_per_m[0], config.stiffness_bounds_kn_per_m[1]],
            [config.damping_bounds_kns_per_m[0], config.damping_bounds_kns_per_m[1]],
        ],
        dtype=float,
    )


def _objective_factory(config: BuildingConfig, record, backend: str):
    uncontrolled_cache = analyze_with_backend(
        config, record, params=None, backend=backend
    )

    def objective(position: np.ndarray) -> float:
        params = TMDParameters(
            mass_ton=config.tmd_mass_ton,
            stiffness_kn_per_m=float(position[0]),
            damping_kns_per_m=float(position[1]),
        )
        controlled = analyze_with_backend(
            config, record, params=params, backend=backend
        )
        transient_ratio = top_floor_displacement_ratio(controlled, uncontrolled_cache)
        omega = np.linspace(0.1, 40.0, 512)
        state_ratio = state_space_objective(
            build_controlled_mck(config, params),
            build_uncontrolled_mck(config),
            omega,
        )
        return float(0.75 * transient_ratio + 0.25 * state_ratio)

    return objective, uncontrolled_cache


def _optimize_algorithms_for_record(
    config: BuildingConfig, record, backend: str, profile: str, progress: bool = False
) -> tuple[DynamicResponse, dict[str, OptimizationResult], dict[str, DynamicResponse]]:
    objective, uncontrolled = _objective_factory(config, record, backend)
    optimizations: dict[str, OptimizationResult] = {}
    controlled: dict[str, DynamicResponse] = {}
    for algorithm in ("pso", "woa", "hpw"):
        label = f"{record.name}:{algorithm.upper()}"
        result = run_optimizer(
            algorithm,
            objective,
            _bounds(config),
            _optimizer_config(
                algorithm, profile, show_progress=progress, progress_label=label
            ),
        )
        params = TMDParameters(
            mass_ton=config.tmd_mass_ton,
            stiffness_kn_per_m=float(result.best_position[0]),
            damping_kns_per_m=float(result.best_position[1]),
        )
        optimizations[algorithm] = result
        controlled[algorithm] = analyze_with_backend(
            config, record, params=params, backend=backend
        )
    return uncontrolled, optimizations, controlled


def _displacement_table(
    uncontrolled: DynamicResponse, controlled: dict[str, DynamicResponse]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for story_idx, baseline in enumerate(
        uncontrolled.peak_story_displacements_m, start=1
    ):
        row: dict[str, object] = {"story": story_idx, "without_tmd": float(baseline)}
        for algorithm, response in controlled.items():
            row[algorithm] = float(response.peak_story_displacements_m[story_idx - 1])
        rows.append(row)
    return rows


def _reduction_table(
    uncontrolled: DynamicResponse, controlled: dict[str, DynamicResponse]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    algorithm_means: dict[str, float] = {}
    for algorithm, response in controlled.items():
        reductions = 100.0 * (
            1.0
            - response.peak_story_displacements_m
            / uncontrolled.peak_story_displacements_m
        )
        algorithm_means[algorithm] = float(np.mean(reductions))
    for story_idx in range(1, len(uncontrolled.peak_story_displacements_m) + 1):
        row: dict[str, object] = {"story": story_idx}
        for algorithm, response in controlled.items():
            reductions = 100.0 * (
                1.0
                - response.peak_story_displacements_m
                / uncontrolled.peak_story_displacements_m
            )
            row[algorithm] = float(reductions[story_idx - 1])
        rows.append(row)
    rows.append({"story": "mean", **algorithm_means})
    return rows


def _example_table_payload(
    config: BuildingConfig,
    uncontrolled: DynamicResponse,
    controlled: dict[str, DynamicResponse],
) -> dict[str, list[dict[str, object]]]:
    displacement_key = "table3" if config.name == "example1" else "table11"
    reduction_key = "table4" if config.name == "example1" else "table12"
    return {
        displacement_key: _displacement_table(uncontrolled, controlled),
        reduction_key: _reduction_table(uncontrolled, controlled),
    }


def run_example(
    name: str, backend: str = "auto", profile: str = "full", progress: bool = False
) -> BenchmarkRun:
    config = get_benchmark(name)
    notes: list[str] = []
    record = load_record("el_centro")
    uncontrolled, optimizations, controlled = _optimize_algorithms_for_record(
        config, record, backend, profile, progress=progress
    )
    tables = _example_table_payload(config, uncontrolled, controlled)

    run = BenchmarkRun(
        benchmark=config,
        backend=backend,
        mode="simulate",
        uncontrolled=uncontrolled,
        controlled=controlled,
        optimizations=optimizations,
        tables=tables,
        figures={},
        notes=notes,
    )
    run.figures = publish_run(ROOT, run)
    return run


def run_mass_sweep(backend: str = "auto") -> dict[str, object]:
    config = get_benchmark("example1")
    record = load_record("el_centro")
    rows = []
    for mass in (90.0, 96.0, 100.0, 104.0, 108.0, 112.0, 116.0):
        tuned_config = with_tmd_mass(config, mass)
        params = get_reference_params("example1", "pso", mass_ton=mass)
        response = analyze_with_backend(
            tuned_config,
            record,
            params=params,
            backend=backend,
        )
        row = {
            "story": 10,
            "mass": mass,
            "top_floor": float(response.peak_story_displacements_m[-1]),
        }
        rows.append(row)
    publish_simple_table("mass_sweep", rows)
    return {"mode": "simulate", "rows": rows}


def _run_far_field_simulate(
    backend: str = "auto", profile: str = "full", progress: bool = False
) -> dict[str, object]:
    config = get_benchmark("example1")
    record_names = [
        ("Northridge", "northridge"),
        ("Duzce, Turkey", "duzce_turkey"),
        ("Hector Mine", "hector_mine"),
        ("Kobe, Japan", "kobe_japan"),
        ("Landers", "landers"),
        ("Manjil, Iran", "manjil_iran"),
    ]
    rows: list[dict[str, object]] = []
    record_iterable = record_names
    if progress:
        record_iterable = tqdm(
            record_names, desc="Far-field records", dynamic_ncols=True
        )
    for label, record_name in record_iterable:
        record = load_record(record_name)
        uncontrolled, optimizations, controlled_map = _optimize_algorithms_for_record(
            config, record, backend, profile, progress=progress
        )
        for algorithm, result in optimizations.items():
            controlled = controlled_map[algorithm]
            story_reduction = 100.0 * (
                1.0
                - controlled.peak_story_displacements_m
                / uncontrolled.peak_story_displacements_m
            )
            row = {
                "gm": label,
                "case": algorithm.upper(),
                "mean": float(np.mean(story_reduction)),
                "kd": float(result.best_position[0]),
                "cd": float(result.best_position[1]),
                "objective": float(result.best_value),
                "iterations": int(result.iterations),
                "runtime_s": float(result.runtime_s),
                "profile": profile,
            }
            for index, value in enumerate(story_reduction, start=1):
                row[f"story_{index}"] = float(value)
            rows.append(row)
    publish_simple_table("far_field_simulated", rows)
    return {"mode": "simulate", "rows": rows}


def run_far_field(
    backend: str = "auto", profile: str = "full", progress: bool = False
) -> dict[str, object]:
    return _run_far_field_simulate(backend=backend, profile=profile, progress=progress)


def publish_simple_table(stem: str, rows: list[dict[str, object]]) -> None:
    from .reporting import ensure_result_dirs, write_csv

    paths = ensure_result_dirs(ROOT)
    write_csv(rows, paths["tables"] / f"{stem}.csv")


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()
