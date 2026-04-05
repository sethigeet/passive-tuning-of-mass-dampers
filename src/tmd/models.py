from dataclasses import asdict

import numpy as np

from .types import Array, BuildingConfig, TMDParameters

TON_TO_KG = 1000.0
KN_TO_N = 1000.0


def _story_tridiagonal(values: tuple[float, ...]) -> Array:
    n = len(values)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        current = values[i]
        below = values[i + 1] if i + 1 < n else 0.0
        matrix[i, i] = current + below
        if i + 1 < n:
            matrix[i, i + 1] = -below
            matrix[i + 1, i] = -below
    return matrix


def build_uncontrolled_mck(config: BuildingConfig) -> tuple[Array, Array, Array]:
    masses = np.diag(np.array(config.story_masses_ton, dtype=float) * TON_TO_KG)
    stiffness = _story_tridiagonal(
        tuple(v * KN_TO_N for v in config.story_stiffness_kn_per_m)
    )
    damping = _story_tridiagonal(
        tuple(v * KN_TO_N for v in config.story_damping_kns_per_m)
    )
    return masses, damping, stiffness


def resolve_tmd_installation_floor(config: BuildingConfig, params: TMDParameters) -> int:
    floor = config.n_stories if params.installation_floor is None else params.installation_floor
    if isinstance(floor, bool) or not isinstance(floor, int):
        raise ValueError("TMD installation floor must be an integer story index.")
    if floor < 1 or floor > config.n_stories:
        raise ValueError(
            f"TMD installation floor must be between 1 and {config.n_stories}."
        )
    return floor


def build_controlled_mck(
    config: BuildingConfig, params: TMDParameters
) -> tuple[Array, Array, Array]:
    m, c, k = build_uncontrolled_mck(config)
    n = config.n_stories
    m_aug = np.zeros((n + 1, n + 1), dtype=float)
    c_aug = np.zeros((n + 1, n + 1), dtype=float)
    k_aug = np.zeros((n + 1, n + 1), dtype=float)

    m_aug[:n, :n] = m
    c_aug[:n, :n] = c
    k_aug[:n, :n] = k
    m_aug[n, n] = params.mass_ton * TON_TO_KG

    kd = params.stiffness_kn_per_m * KN_TO_N
    cd = params.damping_kns_per_m * KN_TO_N

    floor_index = resolve_tmd_installation_floor(config, params) - 1
    k_aug[floor_index, floor_index] += kd
    k_aug[n, n] += kd
    k_aug[floor_index, n] -= kd
    k_aug[n, floor_index] -= kd

    c_aug[floor_index, floor_index] += cd
    c_aug[n, n] += cd
    c_aug[floor_index, n] -= cd
    c_aug[n, floor_index] -= cd
    return m_aug, c_aug, k_aug


def influence_vector(size: int) -> Array:
    return np.ones(size, dtype=float)


def normalize_config(config: BuildingConfig) -> dict[str, object]:
    return asdict(config)
