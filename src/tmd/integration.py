import numpy as np
from numba import njit

from .types import Array, DynamicResponse


@njit(cache=True)
def _newmark_loop(m, c, k_eff_inv, external, inv_m, a0, a1, a2, a3, a4, a5, dt, gamma):
    n_steps = external.shape[0]
    n_dof = m.shape[0]
    u = np.zeros((n_steps, n_dof))
    v = np.zeros((n_steps, n_dof))
    acc = np.zeros((n_steps, n_dof))

    # Initial acceleration (u[0]=0, v[0]=0 so c/k terms vanish)
    acc[0] = inv_m @ external[0]

    for i in range(1, n_steps):
        m_contrib = a0 * u[i - 1] + a2 * v[i - 1] + a3 * acc[i - 1]
        c_contrib = a1 * u[i - 1] + a4 * v[i - 1] + a5 * acc[i - 1]
        p_eff = external[i] + m @ m_contrib + c @ c_contrib
        u[i] = k_eff_inv @ p_eff
        acc[i] = a0 * (u[i] - u[i - 1]) - a2 * v[i - 1] - a3 * acc[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * acc[i - 1] + gamma * acc[i])

    return u, v, acc


def newmark_linear(
    m: Array,
    c: Array,
    k: Array,
    time: Array,
    external: Array,
    gamma: float = 0.5,
    beta: float = 0.25,
) -> DynamicResponse:
    n = m.shape[0]
    if external.shape[1] != n:
        raise ValueError("External force history width must match system DOF count.")
    if len(time) < 2:
        raise ValueError("Excitation time history must contain at least two samples.")
    dt = float(time[1] - time[0])
    inv_m = np.linalg.inv(m)

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    k_eff = k + a0 * m + a1 * c
    k_eff_inv = np.linalg.inv(k_eff)

    u, v, a = _newmark_loop(
        m, c, k_eff_inv, external, inv_m,
        a0, a1, a2, a3, a4, a5, dt, gamma,
    )

    peaks = np.max(np.abs(u), axis=0)
    objective = float(np.max(peaks))
    return DynamicResponse(
        time=time,
        relative_displacements_m=u,
        relative_velocities_mps=v,
        relative_accelerations_mps2=a,
        peak_story_displacements_m=peaks,
        objective_value=objective,
        metadata={"solver": "newmark_linear"},
    )
