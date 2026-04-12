import numpy as np

from tmd.examples import get_example_config
from tmd.models import (
    build_controlled_mck,
    build_scaled_uncontrolled_mck,
    build_uncontrolled_mck,
)
from tmd.types import TMDParameters


def test_uncontrolled_matrices_have_expected_shape_and_symmetry():
    config = get_example_config("example1")
    m, c, k = build_uncontrolled_mck(config)
    assert m.shape == (10, 10)
    assert c.shape == (10, 10)
    assert k.shape == (10, 10)
    assert np.allclose(m, m.T)
    assert np.allclose(c, c.T)
    assert np.allclose(k, k.T)


def test_controlled_matrices_add_tmd_coupling_at_selected_floor_and_tmd():
    config = get_example_config("example1")
    params = TMDParameters(config.tmd_mass_ton, 4136.0, 117.5, installation_floor=4)
    _, c, k = build_controlled_mck(config, params)
    floor_index = 3
    tmd = config.n_stories
    assert k[floor_index, tmd] < 0.0
    assert k[tmd, floor_index] < 0.0
    assert c[floor_index, tmd] < 0.0
    assert c[tmd, floor_index] < 0.0
    assert np.allclose(k[:floor_index, tmd], 0.0)
    assert np.allclose(k[floor_index + 1 : tmd, tmd], 0.0)


def test_scaled_uncontrolled_matrices_follow_area_scale_assumptions():
    config = get_example_config("example1")
    m0, c0, k0 = build_uncontrolled_mck(config)
    s = 1.5

    m, c, k = build_scaled_uncontrolled_mck(config, s)

    assert np.allclose(m, s * m0)
    assert np.allclose(c, (s**1.5) * c0)
    assert np.allclose(k, (s**2) * k0)
