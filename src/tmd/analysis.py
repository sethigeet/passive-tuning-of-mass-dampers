from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import build_controlled_mck, build_uncontrolled_mck, influence_vector
from .types import Array, BuildingConfig, DynamicResponse, Record, TMDParameters


def newmark_linear(
    m: Array,
    c: Array,
    k: Array,
    record: Record,
    gamma: float = 0.5,
    beta: float = 0.25,
) -> DynamicResponse:
    n = m.shape[0]
    dt = record.dt
    r = influence_vector(n)
    u = np.zeros((len(record.time), n), dtype=float)
    v = np.zeros_like(u)
    a = np.zeros_like(u)

    inv_m = np.linalg.inv(m)
    external = -np.outer(record.accel_mps2, m @ r)
    a[0] = inv_m @ (external[0] - c @ v[0] - k @ u[0])

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    k_eff = k + a0 * m + a1 * c

    for i in range(1, len(record.time)):
        p_eff = (
            external[i]
            + m @ (a0 * u[i - 1] + a2 * v[i - 1] + a3 * a[i - 1])
            + c @ (a1 * u[i - 1] + a4 * v[i - 1] + a5 * a[i - 1])
        )
        u[i] = np.linalg.solve(k_eff, p_eff)
        a[i] = a0 * (u[i] - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

    peaks = np.max(np.abs(u), axis=0)
    objective = float(peaks[-1])
    return DynamicResponse(
        time=record.time,
        relative_displacements_m=u,
        relative_velocities_mps=v,
        relative_accelerations_mps2=a,
        peak_story_displacements_m=peaks,
        objective_value=objective,
        metadata={"solver": "newmark_linear"},
    )


def analyze_uncontrolled(config: BuildingConfig, record: Record) -> DynamicResponse:
    return newmark_linear(*build_uncontrolled_mck(config), record)


def analyze_controlled(
    config: BuildingConfig, params: TMDParameters, record: Record
) -> DynamicResponse:
    response = newmark_linear(*build_controlled_mck(config, params), record)
    story_disp = response.relative_displacements_m[:, : config.n_stories]
    response.relative_displacements_m = story_disp
    response.relative_velocities_mps = response.relative_velocities_mps[
        :, : config.n_stories
    ]
    response.relative_accelerations_mps2 = response.relative_accelerations_mps2[
        :, : config.n_stories
    ]
    response.peak_story_displacements_m = np.max(np.abs(story_disp), axis=0)
    response.objective_value = float(response.peak_story_displacements_m[-1])
    return response


def top_floor_displacement_ratio(
    controlled: DynamicResponse, uncontrolled: DynamicResponse
) -> float:
    return float(
        controlled.peak_story_displacements_m[-1]
        / uncontrolled.peak_story_displacements_m[-1]
    )


def write_record_csv(record: Record, destination: Path) -> None:
    payload = np.column_stack((record.time, record.accel_mps2 / 9.80665))
    np.savetxt(destination, payload, delimiter=",", header="time,accel_g", comments="")
