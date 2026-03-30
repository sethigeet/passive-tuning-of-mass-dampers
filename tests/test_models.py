import numpy as np

from tmd.benchmarks import get_benchmark
from tmd.models import build_controlled_mck, build_uncontrolled_mck
from tmd.types import TMDParameters


def test_uncontrolled_matrices_have_expected_shape_and_symmetry():
    config = get_benchmark("example1")
    m, c, k = build_uncontrolled_mck(config)
    assert m.shape == (10, 10)
    assert c.shape == (10, 10)
    assert k.shape == (10, 10)
    assert np.allclose(m, m.T)
    assert np.allclose(c, c.T)
    assert np.allclose(k, k.T)


def test_controlled_matrices_add_tmd_coupling_only_at_roof_and_tmd():
    config = get_benchmark("example1")
    params = TMDParameters(config.tmd_mass_ton, 4136.0, 117.5)
    _, c, k = build_controlled_mck(config, params)
    roof = config.n_stories - 1
    tmd = config.n_stories
    assert k[roof, tmd] < 0.0
    assert k[tmd, roof] < 0.0
    assert c[roof, tmd] < 0.0
    assert c[tmd, roof] < 0.0
