import math
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

from .backends import analyze_with_backend
from .examples import get_example_config, with_tmd_mass
from .io import load_record, load_wind_bundle
from .optimizers import OptimizerConfig, run_optimizer
from .reference import get_reference_params
from .reporting import publish_multi_hazard_run, publish_run
from .spectra import (
    fundamental_period,
    pseudo_spectral_acceleration,
    scale_record_to_target_spectral_acceleration,
)
from .types import (
    AlgorithmConfig,
    AlgorithmName,
    BuildingConfig,
    DynamicResponse,
    ExampleRun,
    GAOptimizerSettings,
    HazardBundle,
    HazardCase,
    GlobalOptimizerSettings,
    HPWOptimizerSettings,
    MultiHazardRun,
    OptimizationProfileSettings,
    OptimizationResult,
    PSOOptimizerSettings,
    TMDParameters,
    WOAOptimizerSettings,
)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ALGORITHMS: tuple[AlgorithmName, ...] = ("gahpw",)


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


def _load_ga_optimizer_settings(
    payload: dict[str, object], *, context: str
) -> GAOptimizerSettings:
    return GAOptimizerSettings(
        population=_require_int(payload["population"], context=f"{context}.population"),
        iterations=_require_int(payload["iterations"], context=f"{context}.iterations"),
        crossover_rate=_require_float(
            payload["crossover_rate"],
            context=f"{context}.crossover_rate",
        ),
        mutation_rate=_require_float(
            payload["mutation_rate"],
            context=f"{context}.mutation_rate",
        ),
        tournament_size=_require_int(
            payload["tournament_size"],
            context=f"{context}.tournament_size",
        ),
        elite_count=_require_int(
            payload["elite_count"], context=f"{context}.elite_count"
        ),
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
        ga=_load_ga_optimizer_settings(
            _require_table(payload["ga"], context=f"{context}.ga"),
            context=f"{context}.ga",
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
        gahpw=_load_ga_optimizer_settings(
            _require_table(payload["gahpw"], context=f"{context}.gahpw"),
            context=f"{context}.gahpw",
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
    return replace(
        profile_settings.optimizer_config(
            algorithm,
            show_progress=show_progress,
            progress_label=progress_label,
        ),
        integer_indices=(0,),
    )


def _bounds(config: BuildingConfig) -> np.ndarray:
    return np.array(
        [
            [1.0, float(config.n_stories)],
            [config.tmd_mass_bounds_ton[0], config.tmd_mass_bounds_ton[1]],
            [config.stiffness_bounds_kn_per_m[0], config.stiffness_bounds_kn_per_m[1]],
            [config.damping_bounds_kns_per_m[0], config.damping_bounds_kns_per_m[1]],
        ],
        dtype=float,
    )


def _position_to_params(config: BuildingConfig, position: np.ndarray) -> TMDParameters:
    floor = int(np.clip(np.rint(position[0]), 1, config.n_stories))
    return TMDParameters(
        mass_ton=float(position[1]),
        stiffness_kn_per_m=float(position[2]),
        damping_kns_per_m=float(position[3]),
        installation_floor=floor,
    )


def _global_peak_displacement_ratio(
    controlled: DynamicResponse, uncontrolled: DynamicResponse
) -> float:
    controlled_peak = float(np.max(controlled.peak_story_displacements_m))
    uncontrolled_peak = float(np.max(uncontrolled.peak_story_displacements_m))
    return controlled_peak / max(uncontrolled_peak, 1.0e-12)


def _damper_cost(params: TMDParameters) -> float:
    return (
        8.0 * 1000 * params.mass_ton
        + 150.0 * params.damping_kns_per_m
        + 2.0 * math.sqrt(params.stiffness_kn_per_m * params.mass_ton)
        + 100_000.0
    )


class _ObjectiveEvaluator:
    """Picklable single-evaluation callable for use in worker processes."""

    def __init__(
        self,
        config: BuildingConfig,
        excitation,
        backend: str,
        uncontrolled: DynamicResponse,
    ):
        self._config = config
        self._excitation = excitation
        self._backend = backend
        self._uncontrolled = uncontrolled

    def __call__(self, position: np.ndarray) -> float:
        params = _position_to_params(self._config, position)
        controlled = analyze_with_backend(
            self._config, self._excitation, params=params, backend=self._backend
        )
        displacement_ratio = _global_peak_displacement_ratio(
            controlled, self._uncontrolled
        )
        return displacement_ratio + 1.0e-7 * _damper_cost(params)


class _MultiHazardObjectiveEvaluator:
    """Picklable multi-hazard objective evaluator."""

    def __init__(
        self,
        config: BuildingConfig,
        hazard_bundle: HazardBundle,
        backend: str,
        uncontrolled: dict[str, DynamicResponse],
    ):
        self._config = config
        self._hazard_bundle = hazard_bundle
        self._backend = backend
        self._uncontrolled = uncontrolled

    def __call__(self, position: np.ndarray) -> float:
        params = _position_to_params(self._config, position)
        total = 0.0
        for case in self._hazard_bundle.cases:
            controlled = analyze_with_backend(
                self._config,
                case.excitation,
                params=params,
                backend=self._backend,
            )
            total += case.weight * _global_peak_displacement_ratio(
                controlled, self._uncontrolled[case.name]
            )
        return total + 1.0e-7 * _damper_cost(params)


_worker_evaluator: Callable[[np.ndarray], float] | None = None


def _init_worker(evaluator: Callable[[np.ndarray], float]) -> None:
    global _worker_evaluator
    _worker_evaluator = evaluator


def _worker_evaluate(position: np.ndarray) -> float:
    assert _worker_evaluator is not None
    return _worker_evaluator(position)


class BatchObjective:
    """Objective function with caching and parallel batch evaluation."""

    def __init__(
        self, evaluator: Callable[[np.ndarray], float], max_workers: int | None = None
    ):
        self._evaluator = evaluator
        self._cache: dict[bytes, float] = {}
        self._max_workers = max_workers or os.cpu_count() or 1
        self._pool: ProcessPoolExecutor | None = None

    def __call__(self, position: np.ndarray) -> float:
        key = position.tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._evaluator(position)
        self._cache[key] = result
        return result

    def batch(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate a batch of positions in parallel, utilizing the cache."""
        keys = [pos.tobytes() for pos in positions]
        results: list[float | None] = [self._cache.get(k) for k in keys]

        uncached = [(i, positions[i]) for i, r in enumerate(results) if r is None]
        if not uncached:
            return np.array(results, dtype=float)

        uncached_indices, uncached_positions = zip(*uncached)

        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_init_worker,
                initargs=(self._evaluator,),
            )

        computed = list(self._pool.map(_worker_evaluate, uncached_positions))
        for idx, val in zip(uncached_indices, computed):
            results[idx] = val
            self._cache[keys[idx]] = val

        return np.array(results, dtype=float)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def _objective_factory(config: BuildingConfig, excitation, backend: str):
    uncontrolled_cache = analyze_with_backend(
        config, excitation, params=None, backend=backend
    )
    evaluator = _ObjectiveEvaluator(config, excitation, backend, uncontrolled_cache)
    objective = BatchObjective(evaluator)
    return objective, uncontrolled_cache


def _multi_hazard_objective_factory(
    config: BuildingConfig, hazard_bundle: HazardBundle, backend: str
) -> tuple[BatchObjective, dict[str, DynamicResponse]]:
    uncontrolled = {
        case.name: analyze_with_backend(
            config, case.excitation, params=None, backend=backend
        )
        for case in hazard_bundle.cases
    }
    evaluator = _MultiHazardObjectiveEvaluator(
        config, hazard_bundle, backend, uncontrolled
    )
    return BatchObjective(evaluator), uncontrolled


def _load_example_record(config: BuildingConfig):
    return load_record(config.example_record_name)


def _default_wind_bundle_name(config: BuildingConfig) -> str:
    return f"{config.name}_dev"


def _build_multi_hazard_bundle(
    config: BuildingConfig, *, wind_bundle_name: str | None = None
) -> HazardBundle:
    seismic_cases = (
        HazardCase(
            name=config.example_record_name,
            family="seismic",
            excitation=_load_example_record(config),
        ),
    )
    wind_cases = tuple(
        HazardCase(name=excitation.name, family="wind", excitation=excitation)
        for excitation in load_wind_bundle(
            config, wind_bundle_name or _default_wind_bundle_name(config)
        )
    )
    weighted_cases: list[HazardCase] = []
    for case in seismic_cases:
        weighted_cases.append(
            HazardCase(
                name=case.name,
                family=case.family,
                excitation=case.excitation,
                weight=0.5 / len(seismic_cases),
                metadata=case.metadata,
            )
        )
    for case in wind_cases:
        weighted_cases.append(
            HazardCase(
                name=case.name,
                family=case.family,
                excitation=case.excitation,
                weight=0.5 / len(wind_cases),
                metadata=case.metadata,
            )
        )
    return HazardBundle(
        name=wind_bundle_name or _default_wind_bundle_name(config),
        cases=tuple(weighted_cases),
        metadata={"hazards": ("seismic", "wind")},
    )


def _scaled_far_field_record(config: BuildingConfig, record_name: str):
    record = load_record(record_name)
    target = load_record(config.far_field_target_record_name)
    period = fundamental_period(config)
    target_sa = pseudo_spectral_acceleration(target, period)
    return scale_record_to_target_spectral_acceleration(record, target_sa, period)


def _optimize_algorithms_for_record(
    config: BuildingConfig, excitation, backend: str, profile: str, progress: bool = False
) -> tuple[DynamicResponse, dict[str, OptimizationResult], dict[str, DynamicResponse]]:
    objective, uncontrolled = _objective_factory(config, excitation, backend)
    optimizations: dict[str, OptimizationResult] = {}
    controlled: dict[str, DynamicResponse] = {}
    try:
        for algorithm in WORKFLOW_ALGORITHMS:
            label = f"{excitation.name}:{algorithm.upper()}"
            result = run_optimizer(
                algorithm,
                objective,
                _bounds(config),
                _optimizer_config(
                    algorithm, profile, show_progress=progress, progress_label=label
                ),
            )
            params = _position_to_params(config, result.best_position)
            optimizations[algorithm] = result
            controlled[algorithm] = analyze_with_backend(
                config, excitation, params=params, backend=backend
            )
    finally:
        objective.shutdown()
    return uncontrolled, optimizations, controlled


def _optimize_hazard_bundle(
    config: BuildingConfig,
    hazard_bundle: HazardBundle,
    backend: str,
    profile: str,
    progress: bool = False,
) -> tuple[
    OptimizationResult,
    dict[str, DynamicResponse],
    dict[str, DynamicResponse],
    list[dict[str, object]],
]:
    objective, uncontrolled = _multi_hazard_objective_factory(
        config, hazard_bundle, backend
    )
    try:
        algorithm = WORKFLOW_ALGORITHMS[0]
        result = run_optimizer(
            algorithm,
            objective,
            _bounds(config),
            _optimizer_config(
                algorithm,
                profile,
                show_progress=progress,
                progress_label=f"{hazard_bundle.name}:{algorithm.upper()}",
            ),
        )
    finally:
        objective.shutdown()

    params = _position_to_params(config, result.best_position)
    controlled = {
        case.name: analyze_with_backend(
            config, case.excitation, params=params, backend=backend
        )
        for case in hazard_bundle.cases
    }
    case_objectives = []
    for case in hazard_bundle.cases:
        ratio = _global_peak_displacement_ratio(controlled[case.name], uncontrolled[case.name])
        case_objectives.append(
            {
                "case": case.name,
                "family": case.family,
                "weight": float(case.weight),
                "displacement_ratio": float(ratio),
                "weighted_contribution": float(case.weight * ratio),
            }
        )
    return result, uncontrolled, controlled, case_objectives


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


def _table_slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace(":", "_")
        .replace("/", "_")
    )


def _single_response_tables(
    uncontrolled: DynamicResponse, controlled: DynamicResponse
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    controlled_map = {"with_tmd": controlled}
    return (
        _displacement_table(uncontrolled, controlled_map),
        _reduction_table(uncontrolled, controlled_map),
    )


def _multi_hazard_tables(
    config: BuildingConfig,
    hazard_bundle: HazardBundle,
    optimization: OptimizationResult,
    uncontrolled: dict[str, DynamicResponse],
    controlled: dict[str, DynamicResponse],
    case_objectives: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    params = _position_to_params(config, optimization.best_position)
    tables: dict[str, list[dict[str, object]]] = {
        "aggregate_summary": [
            {
                "hazard_bundle": hazard_bundle.name,
                "floor": params.installation_floor,
                "mass_ton": float(params.mass_ton),
                "kd": float(params.stiffness_kn_per_m),
                "cd": float(params.damping_kns_per_m),
                "objective": float(optimization.best_value),
                "damper_cost": float(_damper_cost(params)),
                "iterations": int(optimization.iterations),
                "runtime_s": float(optimization.runtime_s),
            }
        ],
        "objective_breakdown": case_objectives,
    }
    for case in hazard_bundle.cases:
        slug = _table_slug(case.name)
        displacement, reduction = _single_response_tables(
            uncontrolled[case.name], controlled[case.name]
        )
        tables[f"{slug}_displacement"] = displacement
        tables[f"{slug}_reduction"] = reduction
    return tables


def run_example(
    name: str,
    backend: str = "auto",
    profile: str = "full",
    progress: bool = False,
    hazards: str = "seismic",
    wind_bundle: str | None = None,
) -> ExampleRun | MultiHazardRun:
    config = get_example_config(name)
    if hazards == "seismic":
        notes = [
            "objective: minimize global peak displacement ratio plus damper cost",
            "decision vector: [installation_floor, mass_ton, stiffness_kn_per_m, damping_kns_per_m]",
            "workflow optimizer: mixed-integer GA+HPW hybrid",
        ]
        record = _load_example_record(config)
        uncontrolled, optimizations, controlled = _optimize_algorithms_for_record(
            config, record, backend, profile, progress=progress
        )
        tables = _example_table_payload(config, uncontrolled, controlled)

        run = ExampleRun(
            example=config,
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

    if hazards != "seismic,wind":
        raise ValueError(
            "Unsupported hazards selection. Use 'seismic' or 'seismic,wind'."
        )

    if backend == "auto":
        backend = "numpy"

    hazard_bundle = _build_multi_hazard_bundle(config, wind_bundle_name=wind_bundle)
    notes = [
        "objective: minimize weighted multi-hazard displacement ratio plus damper cost",
        "decision vector: [installation_floor, mass_ton, stiffness_kn_per_m, damping_kns_per_m]",
        "workflow optimizer: mixed-integer GA+HPW hybrid",
        f"hazards: {hazards}",
        f"wind_bundle: {hazard_bundle.name}",
    ]
    optimization, uncontrolled, controlled, case_objectives = _optimize_hazard_bundle(
        config, hazard_bundle, backend, profile, progress=progress
    )
    run = MultiHazardRun(
        example=config,
        backend=backend,
        mode="simulate",
        hazard_bundle=hazard_bundle,
        optimization=optimization,
        uncontrolled=uncontrolled,
        controlled=controlled,
        case_objectives=case_objectives,
        tables=_multi_hazard_tables(
            config, hazard_bundle, optimization, uncontrolled, controlled, case_objectives
        ),
        figures={},
        notes=notes,
    )
    run.figures = publish_multi_hazard_run(ROOT, run)
    return run


def run_mass_sweep(backend: str = "auto") -> dict[str, object]:
    config = get_example_config("example1")
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
    config = get_example_config("example1")
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
                "floor": int(np.rint(result.best_position[0])),
                "mass_ton": float(result.best_position[1]),
                "kd": float(result.best_position[2]),
                "cd": float(result.best_position[3]),
                "objective": float(result.best_value),
                "damper_cost": float(
                    _damper_cost(_position_to_params(config, result.best_position))
                ),
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
