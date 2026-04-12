import json
import math
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .backends import analyze_with_backend
from .examples import get_example_config, with_tmd_mass
from .integration import newmark_linear
from .io import load_record
from .models import TON_TO_KG, build_scaled_uncontrolled_mck
from .optimizers import OptimizerConfig, run_optimizer
from .reference import get_reference_params
from .reporting import ensure_result_dirs, publish_run, write_csv
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
    GlobalOptimizerSettings,
    HPWOptimizerSettings,
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
        record,
        backend: str,
        uncontrolled: DynamicResponse,
    ):
        self._config = config
        self._record = record
        self._backend = backend
        self._uncontrolled = uncontrolled

    def __call__(self, position: np.ndarray) -> float:
        params = _position_to_params(self._config, position)
        controlled = analyze_with_backend(
            self._config, self._record, params=params, backend=self._backend
        )
        displacement_ratio = _global_peak_displacement_ratio(
            controlled, self._uncontrolled
        )
        return displacement_ratio + 1.0e-7 * _damper_cost(params)


_worker_evaluator: _ObjectiveEvaluator | None = None


def _init_worker(evaluator: _ObjectiveEvaluator) -> None:
    global _worker_evaluator
    _worker_evaluator = evaluator


def _worker_evaluate(position: np.ndarray) -> float:
    assert _worker_evaluator is not None
    return _worker_evaluator(position)


class BatchObjective:
    """Objective function with caching and parallel batch evaluation."""

    def __init__(self, evaluator: _ObjectiveEvaluator, max_workers: int | None = None):
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


def _objective_factory(config: BuildingConfig, record, backend: str):
    uncontrolled_cache = analyze_with_backend(
        config, record, params=None, backend=backend
    )
    evaluator = _ObjectiveEvaluator(config, record, backend, uncontrolled_cache)
    objective = BatchObjective(evaluator)
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
    try:
        for algorithm in WORKFLOW_ALGORITHMS:
            label = f"{record.name}:{algorithm.upper()}"
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
                config, record, params=params, backend=backend
            )
    finally:
        objective.shutdown()
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
) -> ExampleRun:
    config = get_example_config(name)
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
    paths = ensure_result_dirs(ROOT)
    write_csv(rows, paths["tables"] / f"{stem}.csv")


def _load_target_deflection_table(
    path: Path, target_column: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(path)
    required = {"story", "without_tmd"}
    missing = required.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Deflection table is missing required columns: {missing_list}")

    frame = frame[frame["story"].astype(str).str.lower() != "mean"].copy()
    frame["story"] = pd.to_numeric(frame["story"], errors="raise").astype(int)
    frame = frame.sort_values("story")

    candidate_columns = [
        column for column in frame.columns if column not in {"story", "without_tmd"}
    ]
    if not candidate_columns:
        raise ValueError(
            "Deflection table must contain at least one controlled-response column."
        )

    selected = target_column or candidate_columns[0]
    if selected not in frame.columns:
        choices = ", ".join(candidate_columns)
        raise ValueError(
            f"Target column '{selected}' was not found in the table. Choices: {choices}"
        )

    return (
        frame["story"].to_numpy(dtype=int),
        frame["without_tmd"].to_numpy(dtype=float),
        frame[selected].to_numpy(dtype=float),
        selected,
    )


def _gather_gahpw_params_from_manifest(example_name: str) -> TMDParameters | None:
    path = ROOT / "results/metadata/run_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("example") != example_name:
        return None

    optimization = payload.get("optimization")
    if isinstance(optimization, dict):
        position = optimization.get("best_position")
        if isinstance(position, list) and len(position) >= 4:
            return TMDParameters(
                mass_ton=float(position[1]),
                stiffness_kn_per_m=float(position[2]),
                damping_kns_per_m=float(position[3]),
                installation_floor=int(np.rint(float(position[0]))),
            )

    optimizations = payload.get("optimizations")
    if isinstance(optimizations, dict):
        gahpw = optimizations.get("gahpw")
        if isinstance(gahpw, dict):
            position = gahpw.get("best_position")
            if isinstance(position, list) and len(position) >= 4:
                return TMDParameters(
                    mass_ton=float(position[1]),
                    stiffness_kn_per_m=float(position[2]),
                    damping_kns_per_m=float(position[3]),
                    installation_floor=int(np.rint(float(position[0]))),
                )
    return None


def _gather_gahpw_params_from_report(example_name: str) -> TMDParameters | None:
    path = ROOT / "results/summary/report.md"
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if f"# Run Summary: {example_name}" not in content:
        return None

    match = re.search(
        r"-\s+GAHPW:\s+best=.*?,\s+floor=(?P<floor>\d+),\s+mass=(?P<mass>[0-9.]+),\s+kd=(?P<kd>[0-9.]+),\s+cd=(?P<cd>[0-9.]+)",
        content,
    )
    if match is None:
        return None
    return TMDParameters(
        mass_ton=float(match.group("mass")),
        stiffness_kn_per_m=float(match.group("kd")),
        damping_kns_per_m=float(match.group("cd")),
        installation_floor=int(match.group("floor")),
    )


def _infer_tmd_params(
    example_name: str, target_column: str
) -> tuple[TMDParameters | None, str | None]:
    if target_column in {"pso", "woa", "hpw"}:
        return get_reference_params(example_name, target_column), "paper_reference"
    if target_column == "gahpw":
        manifest_params = _gather_gahpw_params_from_manifest(example_name)
        if manifest_params is not None:
            return manifest_params, "results_manifest"
        report_params = _gather_gahpw_params_from_report(example_name)
        if report_params is not None:
            return report_params, "summary_report"
    return None, None


def _scaled_response(
    config: BuildingConfig, record, area_scale_factor: float
) -> DynamicResponse:
    return newmark_linear(
        *build_scaled_uncontrolled_mck(config, area_scale_factor),
        record,
    )


def _profile_match_metrics(
    simulated: np.ndarray, target: np.ndarray
) -> tuple[float, float, float]:
    residual = simulated - target
    scale = np.maximum(np.abs(target), 1.0e-12)
    score = float(np.sqrt(np.mean((residual / scale) ** 2)))
    mean_abs_error = float(np.mean(np.abs(residual)))
    max_abs_error = float(np.max(np.abs(residual)))
    return score, mean_abs_error, max_abs_error


def estimate_equivalent_upgrade(
    name: str,
    table_path: str | Path,
    *,
    target_column: str | None = None,
    s_min: float = 1.0,
    s_max: float = 4.0,
    coarse_steps: int = 81,
    refine_steps: int = 41,
    refine_rounds: int = 3,
    upgrade_mass_cost_usd_per_kg: float = 8.0,
    upgrade_fixed_cost_usd: float = 0.0,
    tmd_params: TMDParameters | None = None,
) -> dict[str, object]:
    if s_min <= 0.0 or s_max <= 0.0:
        raise ValueError("Search bounds for s must be positive.")
    if s_min > s_max:
        raise ValueError("s_min must be less than or equal to s_max.")
    if coarse_steps < 2 or refine_steps < 2:
        raise ValueError("Search step counts must be at least 2.")
    if refine_rounds < 0:
        raise ValueError("refine_rounds must be non-negative.")

    config = get_example_config(name)
    record = _load_example_record(config)
    stories, baseline, target, selected_column = _load_target_deflection_table(
        Path(table_path), target_column=target_column
    )
    if len(stories) != config.n_stories:
        raise ValueError(
            f"Deflection table has {len(stories)} stories but {name} expects {config.n_stories}."
        )

    inferred_params_source = None
    if tmd_params is None:
        tmd_params, inferred_params_source = _infer_tmd_params(name, selected_column)

    evaluations: dict[float, dict[str, object]] = {}
    lower = s_min
    upper = s_max
    best: dict[str, object] | None = None

    for round_index in range(refine_rounds + 1):
        steps = coarse_steps if round_index == 0 else refine_steps
        for area_scale_factor in np.linspace(lower, upper, steps):
            s_value = float(area_scale_factor)
            candidate = evaluations.get(s_value)
            if candidate is None:
                response = _scaled_response(config, record, s_value)
                score, mean_abs_error, max_abs_error = _profile_match_metrics(
                    response.peak_story_displacements_m,
                    target,
                )
                candidate = {
                    "s": s_value,
                    "score": score,
                    "mean_abs_error_m": mean_abs_error,
                    "max_abs_error_m": max_abs_error,
                    "global_peak_m": float(np.max(response.peak_story_displacements_m)),
                    "top_story_peak_m": float(response.peak_story_displacements_m[-1]),
                    "response": response,
                }
                evaluations[s_value] = candidate
            if best is None or float(candidate["score"]) < float(best["score"]):
                best = candidate

        assert best is not None
        spacing = (upper - lower) / max(steps - 1, 1)
        half_window = max(4.0 * spacing, 1.0e-6)
        lower = max(s_min, float(best["s"]) - half_window)
        upper = min(s_max, float(best["s"]) + half_window)

    assert best is not None
    best_response = best["response"]
    matched = best_response.peak_story_displacements_m

    comparison_rows: list[dict[str, object]] = []
    for story, baseline_value, target_value, matched_value in zip(
        stories,
        baseline,
        target,
        matched,
        strict=True,
    ):
        abs_error = abs(float(matched_value) - float(target_value))
        rel_error = abs_error / max(abs(float(target_value)), 1.0e-12)
        comparison_rows.append(
            {
                "story": int(story),
                "without_tmd": float(baseline_value),
                "target": float(target_value),
                "matched": float(matched_value),
                "abs_error_m": abs_error,
                "rel_error": rel_error,
            }
        )

    search_rows = [
        {
            "s": float(candidate["s"]),
            "score": float(candidate["score"]),
            "mean_abs_error_m": float(candidate["mean_abs_error_m"]),
            "max_abs_error_m": float(candidate["max_abs_error_m"]),
            "global_peak_m": float(candidate["global_peak_m"]),
            "top_story_peak_m": float(candidate["top_story_peak_m"]),
        }
        for candidate in sorted(evaluations.values(), key=lambda item: float(item["s"]))
    ]

    output_stem = f"{name}_{selected_column}_equivalent_upgrade"
    paths = ensure_result_dirs(ROOT)
    comparison_path = write_csv(
        comparison_rows,
        paths["tables"] / f"{output_stem}_comparison.csv",
    )
    search_path = write_csv(
        search_rows,
        paths["tables"] / f"{output_stem}_search.csv",
    )

    base_mass_ton = float(sum(config.story_masses_ton))
    added_mass_ton = max(float(best["s"]) - 1.0, 0.0) * base_mass_ton
    added_mass_kg = added_mass_ton * TON_TO_KG
    upgrade_cost_estimate_usd = (
        upgrade_fixed_cost_usd + upgrade_mass_cost_usd_per_kg * added_mass_kg
    )

    tmd_cost_estimate_usd = None
    cost_savings_usd = None
    if tmd_params is not None:
        tmd_cost_estimate_usd = float(_damper_cost(tmd_params))
        cost_savings_usd = float(upgrade_cost_estimate_usd - tmd_cost_estimate_usd)

    hit_lower_bound = math.isclose(float(best["s"]), s_min, rel_tol=0.0, abs_tol=1.0e-9)
    hit_upper_bound = math.isclose(float(best["s"]), s_max, rel_tol=0.0, abs_tol=1.0e-9)

    notes = [
        "solver: newmark_linear",
        "matrix scaling assumptions: M(s)=s*M0, K(s)=s^2*K0, C(s)=s^1.5*C0",
        "search objective: minimize RMS relative error between target and scaled-building peak story displacements",
        f"structural upgrade cost proxy: fixed + ({upgrade_mass_cost_usd_per_kg:.3f} USD/kg) * added mass",
    ]
    if inferred_params_source is None and tmd_params is None:
        notes.append(
            "TMD cost was not estimated because no TMD parameters were supplied or inferred from saved results."
        )
    elif inferred_params_source is not None:
        notes.append(f"TMD cost parameters inferred from: {inferred_params_source}")
    if hit_lower_bound or hit_upper_bound:
        notes.append("Best s landed on a search bound; widen the range if you need a broader estimate.")

    return {
        "mode": "estimate_upgrade",
        "example": config.name,
        "record": record.name,
        "table_path": str(Path(table_path)),
        "target_column": selected_column,
        "matched_scale_factor": float(best["s"]),
        "score": float(best["score"]),
        "mean_abs_error_m": float(best["mean_abs_error_m"]),
        "max_abs_error_m": float(best["max_abs_error_m"]),
        "target_global_peak_m": float(np.max(target)),
        "matched_global_peak_m": float(np.max(matched)),
        "target_top_story_peak_m": float(target[-1]),
        "matched_top_story_peak_m": float(matched[-1]),
        "base_building_mass_ton": base_mass_ton,
        "equivalent_building_mass_ton": float(best["s"]) * base_mass_ton,
        "added_building_mass_ton": added_mass_ton,
        "upgrade_cost_estimate_usd": float(upgrade_cost_estimate_usd),
        "tmd_cost_estimate_usd": tmd_cost_estimate_usd,
        "estimated_cost_savings_usd": cost_savings_usd,
        "generated": {
            "comparison_table": str(comparison_path),
            "search_table": str(search_path),
        },
        "notes": notes,
    }
