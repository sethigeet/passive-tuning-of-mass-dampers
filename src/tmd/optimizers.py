from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

from .types import Array, OptimizationResult, OptimizerConfig

Objective = Callable[[Array], float]


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


def _bounded(position: Array, bounds: Array) -> Array:
    low = bounds[:, 0]
    high = bounds[:, 1]
    return np.clip(position, low, high)


def _converged(history: list[float], tolerance: float, window: int) -> bool:
    if len(history) < window:
        return False
    segment = history[-window:]
    return max(segment) - min(segment) <= tolerance


def optimize_pso(
    objective: Objective, bounds: Array, config: OptimizerConfig
) -> OptimizationResult:
    rng = np.random.default_rng(config.seed)
    dimensions = bounds.shape[0]
    positions = rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(config.population, dimensions)
    )
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_values = np.array([objective(p) for p in positions])
    best_idx = int(np.argmin(personal_values))
    global_best = personal_best[best_idx].copy()
    global_value = float(personal_values[best_idx])
    history = [global_value]
    start = time.perf_counter()

    for iteration in _iteration_range(config):
        inertia = config.inertia_start + (
            (config.inertia_end - config.inertia_start)
            * iteration
            / max(config.iterations - 1, 1)
        )
        r1 = rng.random((config.population, dimensions))
        r2 = rng.random((config.population, dimensions))
        velocities = (
            inertia * velocities
            + config.c1 * r1 * (personal_best - positions)
            + config.c2 * r2 * (global_best - positions)
        )
        positions = _bounded(positions + velocities, bounds)
        values = np.array([objective(p) for p in positions])
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
    dimensions = bounds.shape[0]
    positions = rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(config.population, dimensions)
    )
    values = np.array([objective(p) for p in positions])
    best_idx = int(np.argmin(values))
    best_position = positions[best_idx].copy()
    best_value = float(values[best_idx])
    history = [best_value]
    start = time.perf_counter()

    for iteration in _iteration_range(config):
        a = 2.0 - 2.0 * iteration / max(config.iterations - 1, 1)
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
            positions[i] = _bounded(new_pos, bounds)
        values = np.array([objective(p) for p in positions])
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
    dimensions = bounds.shape[0]
    positions = rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(config.population, dimensions)
    )
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_values = np.array([objective(p) for p in positions])
    best_idx = int(np.argmin(personal_values))
    global_best = personal_best[best_idx].copy()
    global_value = float(personal_values[best_idx])
    history = [global_value]
    start = time.perf_counter()

    for iteration in _iteration_range(config):
        inertia = config.inertia_start + (
            (config.inertia_end - config.inertia_start)
            * iteration
            / max(config.iterations - 1, 1)
        )
        r1 = rng.random((config.population, dimensions))
        r2 = rng.random((config.population, dimensions))
        velocities = (
            inertia * velocities
            + config.c1 * r1 * (personal_best - positions)
            + config.c2 * r2 * (global_best - positions)
        )
        positions = _bounded(positions + velocities, bounds)
        values = np.array([objective(p) for p in positions])
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        best_idx = int(np.argmin(personal_values))
        if personal_values[best_idx] < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()

        a = 2.0 - 2.0 * iteration / max(config.iterations - 1, 1)
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
            positions[i] = _bounded(new_pos, bounds)

        values = np.array([objective(p) for p in positions])
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
        case "pso":
            return optimize_pso(objective, bounds, config)
        case "woa":
            return optimize_woa(objective, bounds, config)
        case "hpw":
            return optimize_hpw(objective, bounds, config)
        case _:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
