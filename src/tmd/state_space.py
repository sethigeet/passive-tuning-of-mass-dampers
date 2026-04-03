import numpy as np
from scipy import linalg

from .types import Array


def second_order_to_state_space(
    m: Array, c: Array, k: Array, input_dof: int = 0
) -> tuple[Array, Array, Array, Array]:
    n = m.shape[0]
    zeros = np.zeros_like(m)
    identity = np.eye(n)
    a_top = np.hstack((zeros, identity))
    m_inv = np.linalg.inv(m)
    a_bottom = np.hstack((-m_inv @ k, -m_inv @ c))
    a = np.vstack((a_top, a_bottom))
    force = np.zeros((n, 1), dtype=float)
    force[input_dof, 0] = 1.0
    b = np.vstack((np.zeros((n, 1)), m_inv @ force))
    c_out = np.hstack((identity, zeros))
    d = np.zeros((n, 1))
    return a, b, c_out, d


def displacement_transfer_function(
    a: Array, b: Array, c_out: Array, d: Array, omega: Array
) -> Array:
    response = np.zeros((len(omega), c_out.shape[0]), dtype=complex)
    identity = np.eye(a.shape[0], dtype=complex)
    for index, w in enumerate(omega):
        jw = 1j * w
        h = c_out @ linalg.solve(jw * identity - a, b) + d
        response[index, :] = h[:, 0]
    return response


def state_space_objective(
    controlled: tuple[Array, Array, Array],
    uncontrolled: tuple[Array, Array, Array],
    omega: Array,
    first_floor_displacement_ratio: float | None = None,
) -> float:
    ac, bc, cc, dc = second_order_to_state_space(*controlled)
    au, bu, cu, du = second_order_to_state_space(*uncontrolled)
    hc = displacement_transfer_function(ac, bc, cc, dc, omega)
    hu = displacement_transfer_function(au, bu, cu, du, omega)
    transfer_ratio = np.max(np.abs(hc[:, 0])) / max(np.max(np.abs(hu[:, 0])), 1e-12)
    if first_floor_displacement_ratio is None:
        return float(transfer_ratio)
    return float(
        transfer_ratio + max(first_floor_displacement_ratio, 0.0)
    )
