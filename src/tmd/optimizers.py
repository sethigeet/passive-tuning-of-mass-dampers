import math
import time
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

from .types import Array, OptimizationResult, OptimizerConfig

Objective = Callable[[Array], float]


def _evaluate_batch(objective: Objective, positions: Array) -> Array:
    """Evaluate objective for all positions, using parallel batch if available."""
    if hasattr(objective, "batch"):
        return objective.batch(positions)  # type: ignore
    return np.array([objective(p) for p in positions])


def _iteration_range(config: OptimizerConfig):
    if not config.show_progress:
        return range(config.iterations)
    return tqdm(
        range(config.iterations),
        total=config.iterations,
        desc=config.progress_label or "opt",
        leave=False,
        dynamic_ncols=True,
    )


def _repair_positions(
    positions: Array, bounds: Array, integer_indices: tuple[int, ...]
) -> Array:
    repaired = np.clip(positions, bounds[:, 0], bounds[:, 1])
    if integer_indices:
        repaired[:, integer_indices] = np.rint(repaired[:, integer_indices])
    return repaired


def _repair_position(
    position: Array, bounds: Array, integer_indices: tuple[int, ...]
) -> Array:
    return _repair_positions(
        np.asarray(position, dtype=float).reshape(1, -1), bounds, integer_indices
    )[0]


def _random_population(
    bounds: Array,
    population: int,
    rng: np.random.Generator,
    integer_indices: tuple[int, ...],
) -> Array:
    positions = rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(population, bounds.shape[0])
    )
    return _repair_positions(positions, bounds, integer_indices)


def _converged(history: list[float], tolerance: float, window: int) -> bool:
    if len(history) < window:
        return False
    segment = history[-window:]
    return max(segment) - min(segment) <= tolerance


def _tournament_select(
    positions: Array,
    values: Array,
    rng: np.random.Generator,
    tournament_size: int,
) -> Array:
    contenders = rng.integers(0, len(positions), size=max(tournament_size, 2))
    winner = contenders[int(np.argmin(values[contenders]))]
    return positions[winner].copy()


def _crossover(
    parent_a: Array,
    parent_b: Array,
    rng: np.random.Generator,
    integer_indices: tuple[int, ...],
) -> Array:
    child = parent_a.copy()
    integer_set = set(integer_indices)
    for index in range(len(child)):
        if index in integer_set:
            child[index] = parent_a[index] if rng.random() < 0.5 else parent_b[index]
        else:
            alpha = rng.random()
            child[index] = alpha * parent_a[index] + (1.0 - alpha) * parent_b[index]
    return child


def _mutate(
    child: Array,
    bounds: Array,
    rng: np.random.Generator,
    config: OptimizerConfig,
    integer_indices: tuple[int, ...],
) -> Array:
    integer_set = set(integer_indices)
    for index in range(len(child)):
        if rng.random() >= config.mutation_rate:
            continue
        low = bounds[index, 0]
        high = bounds[index, 1]
        if index in integer_set:
            child[index] = rng.integers(int(math.ceil(low)), int(math.floor(high)) + 1)
            continue
        span = high - low
        child[index] = child[index] + rng.normal(0.0, 0.1 * span)
    return _repair_position(child, bounds, integer_indices)


def optimize_ga(
    objective: Objective, bounds: Array, config: OptimizerConfig
) -> OptimizationResult:
    rng = np.random.default_rng(config.seed)
    integer_indices = tuple(config.integer_indices)
    positions = _random_population(bounds, config.population, rng, integer_indices)
    values = _evaluate_batch(objective, positions)
    best_idx = int(np.argmin(values))
    best_position = positions[best_idx].copy()
    best_value = float(values[best_idx])
    history = [best_value]
    start = time.perf_counter()
    elite_count = min(max(config.elite_count, 1), config.population)

    for _ in _iteration_range(config):
        elite_indices = np.argsort(values)[:elite_count]
        next_positions = [positions[index].copy() for index in elite_indices]
        while len(next_positions) < config.population:
            parent_a = _tournament_select(
                positions, values, rng, config.tournament_size
            )
            parent_b = _tournament_select(
                positions, values, rng, config.tournament_size
            )
            if rng.random() < config.crossover_rate:
                child = _crossover(parent_a, parent_b, rng, integer_indices)
            else:
                child = parent_a.copy()
            next_positions.append(_mutate(child, bounds, rng, config, integer_indices))
        positions = np.array(next_positions)
        values = _evaluate_batch(objective, positions)
        best_idx = int(np.argmin(values))
        if values[best_idx] < best_value:
            best_value = float(values[best_idx])
            best_position = positions[best_idx].copy()
        history.append(best_value)
        if _converged(history, config.convergence_tolerance, config.convergence_window):
            break

    runtime = time.perf_counter() - start
    return OptimizationResult(
        algorithm="ga",
        best_position=best_position,
        best_value=best_value,
        history=history,
        iterations=len(history) - 1,
        runtime_s=runtime,
        seed=config.seed,
    )


def optimize_gahpw(
    objective: Objective, bounds: Array, config: OptimizerConfig
) -> OptimizationResult:
    rng = np.random.default_rng(config.seed)
    integer_indices = tuple(config.integer_indices)
    dimensions = bounds.shape[0]
    positions = _random_population(bounds, config.population, rng, integer_indices)
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_values = _evaluate_batch(objective, positions)
    best_idx = int(np.argmin(personal_values))
    global_best = personal_best[best_idx].copy()
    global_value = float(personal_values[best_idx])
    history = [global_value]
    start = time.perf_counter()
    elite_count = min(max(config.elite_count, 1), config.population)
    iteration_scale = max(config.iterations - 1, 1)

    for iteration in _iteration_range(config):
        elite_indices = np.argsort(personal_values)[:elite_count]
        next_positions = [personal_best[index].copy() for index in elite_indices]
        while len(next_positions) < config.population:
            parent_a = _tournament_select(
                personal_best, personal_values, rng, config.tournament_size
            )
            parent_b = _tournament_select(
                personal_best, personal_values, rng, config.tournament_size
            )
            if rng.random() < config.crossover_rate:
                child = _crossover(parent_a, parent_b, rng, integer_indices)
            else:
                child = parent_a.copy()
            next_positions.append(_mutate(child, bounds, rng, config, integer_indices))
        positions = np.array(next_positions)

        values = _evaluate_batch(objective, positions)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_idx = int(np.argmin(values))
        if values[best_idx] < global_value:
            global_value = float(values[best_idx])
            global_best = positions[best_idx].copy()

        inertia = config.inertia_start + (
            (config.inertia_end - config.inertia_start) * iteration / iteration_scale
        )
        r1 = rng.random((config.population, dimensions))
        r2 = rng.random((config.population, dimensions))
        velocities = (
            inertia * velocities
            + config.c1 * r1 * (personal_best - positions)
            + config.c2 * r2 * (global_best - positions)
        )
        positions = _repair_positions(positions + velocities, bounds, integer_indices)
        values = _evaluate_batch(objective, positions)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_idx = int(np.argmin(personal_values))
        if personal_values[best_idx] < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()

        a = 2.0 - 2.0 * iteration / iteration_scale
        for i in range(config.population):
            r = rng.random(dimensions)
            a_vec = 2.0 * a * r - a
            c_vec = 2.0 * rng.random(dimensions)
            p = rng.random()
            spiral_offset = rng.uniform(-1.0, 1.0)
            if p < 0.5:
                if np.linalg.norm(a_vec, ord=np.inf) < 1.0:
                    d = np.abs(c_vec * global_best - positions[i])
                    new_pos = global_best - a_vec * d
                else:
                    random_agent = positions[rng.integers(0, config.population)]
                    d = np.abs(c_vec * random_agent - positions[i])
                    new_pos = random_agent - a_vec * d
            else:
                d = np.abs(global_best - positions[i])
                new_pos = (
                    d
                    * math.exp(config.b * spiral_offset)
                    * math.cos(2.0 * math.pi * spiral_offset)
                    + global_best
                )
            positions[i] = _repair_position(new_pos, bounds, integer_indices)

        values = _evaluate_batch(objective, positions)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_idx = int(np.argmin(personal_values))
        if personal_values[best_idx] < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()

        history.append(global_value)
        if _converged(history, config.convergence_tolerance, config.convergence_window):
            break

    runtime = time.perf_counter() - start
    return OptimizationResult(
        algorithm="gahpw",
        best_position=global_best,
        best_value=global_value,
        history=history,
        iterations=len(history) - 1,
        runtime_s=runtime,
        seed=config.seed,
    )


def optimize_pso(
    objective: Objective, bounds: Array, config: OptimizerConfig
) -> OptimizationResult:
    rng = np.random.default_rng(config.seed)
    integer_indices = tuple(config.integer_indices)
    dimensions = bounds.shape[0]
    positions = _random_population(bounds, config.population, rng, integer_indices)
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_values = _evaluate_batch(objective, positions)
    best_idx = int(np.argmin(personal_values))
    global_best = personal_best[best_idx].copy()
    global_value = float(personal_values[best_idx])
    history = [global_value]
    start = time.perf_counter()
    iteration_scale = max(config.iterations - 1, 1)

    for iteration in _iteration_range(config):
        inertia = config.inertia_start + (
            (config.inertia_end - config.inertia_start) * iteration / iteration_scale
        )
        r1 = rng.random((config.population, dimensions))
        r2 = rng.random((config.population, dimensions))
        velocities = (
            inertia * velocities
            + config.c1 * r1 * (personal_best - positions)
            + config.c2 * r2 * (global_best - positions)
        )
        positions = _repair_positions(positions + velocities, bounds, integer_indices)
        values = _evaluate_batch(objective, positions)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_idx = int(np.argmin(personal_values))
        if personal_values[best_idx] < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()
        history.append(global_value)
        if _converged(history, config.convergence_tolerance, config.convergence_window):
            break

    runtime = time.perf_counter() - start
    return OptimizationResult(
        algorithm="pso",
        best_position=global_best,
        best_value=global_value,
        history=history,
        iterations=len(history) - 1,
        runtime_s=runtime,
        seed=config.seed,
    )


def optimize_woa(
    objective: Objective, bounds: Array, config: OptimizerConfig
) -> OptimizationResult:
    rng = np.random.default_rng(config.seed)
    integer_indices = tuple(config.integer_indices)
    dimensions = bounds.shape[0]
    positions = _random_population(bounds, config.population, rng, integer_indices)
    values = _evaluate_batch(objective, positions)
    best_idx = int(np.argmin(values))
    best_position = positions[best_idx].copy()
    best_value = float(values[best_idx])
    history = [best_value]
    start = time.perf_counter()
    iteration_scale = max(config.iterations - 1, 1)

    for iteration in _iteration_range(config):
        a = 2.0 - 2.0 * iteration / iteration_scale
        for i in range(config.population):
            r = rng.random(dimensions)
            a_vec = 2.0 * a * r - a
            c_vec = 2.0 * rng.random(dimensions)
            p = rng.random()
            spiral_offset = rng.uniform(-1.0, 1.0)
            if p < 0.5:
                if np.linalg.norm(a_vec, ord=np.inf) < 1.0:
                    d = np.abs(c_vec * best_position - positions[i])
                    new_pos = best_position - a_vec * d
                else:
                    random_agent = positions[rng.integers(0, config.population)]
                    d = np.abs(c_vec * random_agent - positions[i])
                    new_pos = random_agent - a_vec * d
            else:
                d = np.abs(best_position - positions[i])
                new_pos = (
                    d
                    * math.exp(config.b * spiral_offset)
                    * math.cos(2.0 * math.pi * spiral_offset)
                    + best_position
                )
            positions[i] = _repair_position(new_pos, bounds, integer_indices)
        values = _evaluate_batch(objective, positions)
        best_idx = int(np.argmin(values))
        if values[best_idx] < best_value:
            best_value = float(values[best_idx])
            best_position = positions[best_idx].copy()
        history.append(best_value)
        if _converged(history, config.convergence_tolerance, config.convergence_window):
            break

    runtime = time.perf_counter() - start
    return OptimizationResult(
        algorithm="woa",
        best_position=best_position,
        best_value=best_value,
        history=history,
        iterations=len(history) - 1,
        runtime_s=runtime,
        seed=config.seed,
    )


def optimize_hpw(
    objective: Objective, bounds: Array, config: OptimizerConfig
) -> OptimizationResult:
    rng = np.random.default_rng(config.seed)
    integer_indices = tuple(config.integer_indices)
    dimensions = bounds.shape[0]
    positions = _random_population(bounds, config.population, rng, integer_indices)
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_values = _evaluate_batch(objective, positions)
    best_idx = int(np.argmin(personal_values))
    global_best = personal_best[best_idx].copy()
    global_value = float(personal_values[best_idx])
    history = [global_value]
    start = time.perf_counter()
    iteration_scale = max(config.iterations - 1, 1)

    for iteration in _iteration_range(config):
        inertia = config.inertia_start + (
            (config.inertia_end - config.inertia_start) * iteration / iteration_scale
        )
        r1 = rng.random((config.population, dimensions))
        r2 = rng.random((config.population, dimensions))
        velocities = (
            inertia * velocities
            + config.c1 * r1 * (personal_best - positions)
            + config.c2 * r2 * (global_best - positions)
        )
        positions = _repair_positions(positions + velocities, bounds, integer_indices)
        values = _evaluate_batch(objective, positions)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_idx = int(np.argmin(personal_values))
        if personal_values[best_idx] < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()

        a = 2.0 - 2.0 * iteration / iteration_scale
        for i in range(config.population):
            r = rng.random(dimensions)
            a_vec = 2.0 * a * r - a
            c_vec = 2.0 * rng.random(dimensions)
            p = rng.random()
            spiral_offset = rng.uniform(-1.0, 1.0)
            if p < 0.5:
                if np.linalg.norm(a_vec, ord=np.inf) < 1.0:
                    d = np.abs(c_vec * global_best - positions[i])
                    new_pos = global_best - a_vec * d
                else:
                    random_agent = positions[rng.integers(0, config.population)]
                    d = np.abs(c_vec * random_agent - positions[i])
                    new_pos = random_agent - a_vec * d
            else:
                d = np.abs(global_best - positions[i])
                new_pos = (
                    d
                    * math.exp(config.b * spiral_offset)
                    * math.cos(2.0 * math.pi * spiral_offset)
                    + global_best
                )
            positions[i] = _repair_position(new_pos, bounds, integer_indices)

        values = _evaluate_batch(objective, positions)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_idx = int(np.argmin(personal_values))
        if personal_values[best_idx] < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()
        history.append(global_value)
        if _converged(history, config.convergence_tolerance, config.convergence_window):
            break

    runtime = time.perf_counter() - start
    return OptimizationResult(
        algorithm="hpw",
        best_position=global_best,
        best_value=global_value,
        history=history,
        iterations=len(history) - 1,
        runtime_s=runtime,
        seed=config.seed,
    )


def run_optimizer(
    algorithm: str, objective: Objective, bounds: Array, config: OptimizerConfig
) -> OptimizationResult:
    match algorithm:
        case "ga":
            return optimize_ga(objective, bounds, config)
        case "gahpw":
            return optimize_gahpw(objective, bounds, config)
        case "pso":
            return optimize_pso(objective, bounds, config)
        case "woa":
            return optimize_woa(objective, bounds, config)
        case "hpw":
            return optimize_hpw(objective, bounds, config)
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
