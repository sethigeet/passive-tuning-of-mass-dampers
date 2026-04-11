import numpy as np

from tmd.examples import get_example_config
from tmd.models import build_controlled_mck, build_uncontrolled_mck, pad_story_forces
from tmd.state_space import second_order_to_state_space, state_space_objective
from tmd.types import TMDParameters


def test_state_space_block_dimensions_match_second_order_system():
    config = get_example_config("example1")
    a, b, c_out, d = second_order_to_state_space(*build_uncontrolled_mck(config))
    assert a.shape == (20, 20)
    assert b.shape == (20, 1)
    assert c_out.shape == (10, 20)
    assert d.shape == (10, 1)


def test_state_space_objective_is_positive():
    config = get_example_config("example1")
    params = TMDParameters(config.tmd_mass_ton, 4136.0, 117.5)
    omega = np.linspace(0.1, 10.0, 64)
    value = state_space_objective(
        build_controlled_mck(config, params),
        build_uncontrolled_mck(config),
        omega,
    )
    assert value > 0.0


def test_state_space_uses_base_excitation_influence_vector():
    config = get_example_config("example1")
    m, c, k = build_uncontrolled_mck(config)
    _, b, _, _ = second_order_to_state_space(m, c, k)
    assert np.allclose(b[:10, 0], 0.0)
    assert np.allclose(b[10:, 0], -1.0)


def test_pad_story_forces_adds_zero_force_column_for_controlled_tmd_dof():
    config = get_example_config("example1")
    forces = np.full((4, config.n_stories), 12.5)

    padded = pad_story_forces(config, forces, config.n_stories + 1)

    assert padded.shape == (4, config.n_stories + 1)
    assert np.allclose(padded[:, : config.n_stories], forces)
    assert np.allclose(padded[:, -1], 0.0)
