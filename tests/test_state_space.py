import numpy as np
import pytest

from tmd.benchmarks import get_benchmark
from tmd.models import build_controlled_mck, build_uncontrolled_mck
from tmd.state_space import second_order_to_state_space, state_space_objective
from tmd.types import TMDParameters


def test_state_space_block_dimensions_match_second_order_system():
    config = get_benchmark("example1")
    a, b, c_out, d = second_order_to_state_space(*build_uncontrolled_mck(config))
    assert a.shape == (20, 20)
    assert b.shape == (20, 1)
    assert c_out.shape == (10, 20)
    assert d.shape == (10, 1)


def test_state_space_objective_is_positive():
    config = get_benchmark("example1")
    params = TMDParameters(config.tmd_mass_ton, 4136.0, 117.5)
    omega = np.linspace(0.1, 10.0, 64)
    value = state_space_objective(
        build_controlled_mck(config, params),
        build_uncontrolled_mck(config),
        omega,
    )
    assert value > 0.0


def test_state_space_uses_first_floor_force_input():
    config = get_benchmark("example1")
    m, c, k = build_uncontrolled_mck(config)
    _, b, _, _ = second_order_to_state_space(m, c, k)
    assert b[10, 0] == pytest.approx(1.0 / m[0, 0])
    assert np.allclose(b[11:, 0], 0.0)
