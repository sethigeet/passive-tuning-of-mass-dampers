from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .analysis import floor_displacement_ratio
from .benchmarks import get_benchmark, with_tmd_mass
from .io import load_record
from .opensees_model import analyze_with_backend
from .optimizers import OptimizerConfig, run_optimizer
from .reference import get_reference_params
from .reporting import publish_run
from .spectra import (
    fundamental_period,
    pseudo_spectral_acceleration,
    scale_record_to_target_spectral_acceleration,
)
from .types import (
    ALGORITHMS,
    AlgorithmConfig,
    AlgorithmName,
    BenchmarkRun,
    BuildingConfig,
    DynamicResponse,
    GlobalOptimizerSettings,
    HPWOptimizerSettings,
    OptimizationProfileSettings,
    OptimizationResult,
    PSOOptimizerSettings,
    TMDParameters,
    WOAOptimizerSettings,
)

ROOT = Path(__file__).resolve().parents[2]


def _require_table(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table.")
    table: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings.")
        table[key] = item
    return table


def _require_int(value: object, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer.")
    return value


def _require_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a float.")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{context} must be a float.")


def _load_global_optimizer_settings(
    payload: dict[str, object], *, context: str
) -> GlobalOptimizerSettings:
    return GlobalOptimizerSettings(
        seed=_require_int(payload["seed"], context=f"{context}.seed"),
        convergence_window=_require_int(
            payload["convergence_window"],
            context=f"{context}.convergence_window",
        ),
        convergence_tolerance=_require_float(
            payload["convergence_tolerance"],
            context=f"{context}.convergence_tolerance",
        ),
    )


def _load_pso_optimizer_settings(
    payload: dict[str, object], *, context: str
) -> PSOOptimizerSettings:
    return PSOOptimizerSettings(
        population=_require_int(payload["population"], context=f"{context}.population"),
        iterations=_require_int(payload["iterations"], context=f"{context}.iterations"),
        c1=_require_float(payload["c1"], context=f"{context}.c1"),
        c2=_require_float(payload["c2"], context=f"{context}.c2"),
        inertia_start=_require_float(
            payload["inertia_start"],
            context=f"{context}.inertia_start",
        ),
        inertia_end=_require_float(
            payload["inertia_end"], context=f"{context}.inertia_end"
        ),
    )


def _load_woa_optimizer_settings(
    payload: dict[str, object], *, context: str
) -> WOAOptimizerSettings:
    return WOAOptimizerSettings(
        population=_require_int(payload["population"], context=f"{context}.population"),
        iterations=_require_int(payload["iterations"], context=f"{context}.iterations"),
        b=_require_float(payload["b"], context=f"{context}.b"),
    )


def _load_hpw_optimizer_settings(
    payload: dict[str, object], *, context: str
) -> HPWOptimizerSettings:
    return HPWOptimizerSettings(
        population=_require_int(payload["population"], context=f"{context}.population"),
        iterations=_require_int(payload["iterations"], context=f"{context}.iterations"),
        c1=_require_float(payload["c1"], context=f"{context}.c1"),
        c2=_require_float(payload["c2"], context=f"{context}.c2"),
        inertia_start=_require_float(
            payload["inertia_start"],
            context=f"{context}.inertia_start",
        ),
        inertia_end=_require_float(
            payload["inertia_end"], context=f"{context}.inertia_end"
        ),
        b=_require_float(payload["b"], context=f"{context}.b"),
    )


def _load_profile_settings(
    payload: dict[str, object], *, context: str
) -> OptimizationProfileSettings:
    return OptimizationProfileSettings(
        global_settings=_load_global_optimizer_settings(
            _require_table(payload["global"], context=f"{context}.global"),
            context=f"{context}.global",
        ),
        pso=_load_pso_optimizer_settings(
            _require_table(payload["pso"], context=f"{context}.pso"),
            context=f"{context}.pso",
        ),
        woa=_load_woa_optimizer_settings(
            _require_table(payload["woa"], context=f"{context}.woa"),
            context=f"{context}.woa",
        ),
        hpw=_load_hpw_optimizer_settings(
            _require_table(payload["hpw"], context=f"{context}.hpw"),
            context=f"{context}.hpw",
        ),
    )


@lru_cache(maxsize=1)
def _load_algorithm_config() -> AlgorithmConfig:
    with (ROOT / "configs/algorithms.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    profile_payload = _require_table(payload["profiles"], context="profiles")
    profiles: dict[str, OptimizationProfileSettings] = {}
    for profile_name, profile_config in profile_payload.items():
        profiles[profile_name] = _load_profile_settings(
            _require_table(profile_config, context=f"profiles.{profile_name}"),
            context=f"profiles.{profile_name}",
        )
    return AlgorithmConfig(profiles=profiles)


def _optimizer_config(
    algorithm: AlgorithmName,
    profile: str,
    show_progress: bool = False,
    progress_label: str = "",
) -> OptimizerConfig:
    profile_settings = _load_algorithm_config().profile(profile)
    return profile_settings.optimizer_config(
        algorithm,
        show_progress=show_progress,
        progress_label=progress_label,
    )


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
        return floor_displacement_ratio(controlled, uncontrolled_cache, floor_index=-1)

    return objective, uncontrolled_cache


def _load_example_record(config: BuildingConfig):
    return load_record(config.example_record_name)


def _scaled_far_field_record(config: BuildingConfig, record_name: str):
    record = load_record(record_name)
    target = load_record(config.far_field_target_record_name)
    period = fundamental_period(config)
    target_sa = pseudo_spectral_acceleration(target, period)
    return scale_record_to_target_spectral_acceleration(record, target_sa, period)


def _optimize_algorithms_for_record(
    config: BuildingConfig, record, backend: str, profile: str, progress: bool = False
) -> tuple[DynamicResponse, dict[str, OptimizationResult], dict[str, DynamicResponse]]:
    objective, uncontrolled = _objective_factory(config, record, backend)
    optimizations: dict[str, OptimizationResult] = {}
    controlled: dict[str, DynamicResponse] = {}
    for algorithm in ALGORITHMS:
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
    record = _load_example_record(config)
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
    record = _load_example_record(config)
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


def run_far_field(
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
        record, scale_factor = _scaled_far_field_record(config, record_name)
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
                "scale_factor": float(scale_factor),
                "profile": profile,
            }
            for index, value in enumerate(story_reduction, start=1):
                row[f"story_{index}"] = float(value)
            rows.append(row)
    publish_simple_table("far_field_simulated", rows)
    return {"mode": "simulate", "rows": rows}


def publish_simple_table(stem: str, rows: list[dict[str, object]]) -> None:
    from .reporting import ensure_result_dirs, write_csv

    paths = ensure_result_dirs(ROOT)
    write_csv(rows, paths["tables"] / f"{stem}.csv")
