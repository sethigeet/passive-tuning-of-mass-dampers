import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .models import influence_vector
from .types import Array, DynamicResponse, Record


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
    k_eff_lu = lu_factor(k_eff)

    for i in range(1, len(record.time)):
        p_eff = (
            external[i]
            + m @ (a0 * u[i - 1] + a2 * v[i - 1] + a3 * a[i - 1])
            + c @ (a1 * u[i - 1] + a4 * v[i - 1] + a5 * a[i - 1])
        )
        u[i] = lu_solve(k_eff_lu, p_eff)
        a[i] = a0 * (u[i] - u[i - 1]) - a2 * v[i - 1] - a3 * a[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

    peaks = np.max(np.abs(u), axis=0)
    objective = float(np.max(peaks))
    return DynamicResponse(
        time=record.time,
        relative_displacements_m=u,
        relative_velocities_mps=v,
        relative_accelerations_mps2=a,
        peak_story_displacements_m=peaks,
        objective_value=objective,
        metadata={"solver": "newmark_linear"},
    )
