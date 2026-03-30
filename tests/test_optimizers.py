import numpy as np

from tmd.optimizers import OptimizerConfig, optimize_hpw, optimize_pso, optimize_woa


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


BOUNDS = np.array([[-5.0, 5.0], [-5.0, 5.0]])


def test_pso_respects_bounds():
    result = optimize_pso(
        sphere, BOUNDS, OptimizerConfig(population=12, iterations=20, seed=1)
    )
    assert np.all(result.best_position >= BOUNDS[:, 0])
    assert np.all(result.best_position <= BOUNDS[:, 1])


def test_woa_respects_bounds():
    result = optimize_woa(
        sphere, BOUNDS, OptimizerConfig(population=12, iterations=20, seed=1)
    )
    assert np.all(result.best_position >= BOUNDS[:, 0])
    assert np.all(result.best_position <= BOUNDS[:, 1])


def test_hpw_improves_over_initial_guess():
    result = optimize_hpw(
        sphere, BOUNDS, OptimizerConfig(population=12, iterations=20, seed=1)
    )
    assert result.best_value <= result.history[0]
