import pytest
import numpy as np

from tmd.optimizers import (
    OptimizerConfig,
    optimize_ga,
    optimize_gahpw,
    optimize_hpw,
    optimize_pso,
    optimize_woa,
    run_optimizer,
)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


BOUNDS = np.array([[-5.0, 5.0], [-5.0, 5.0]])
INTEGER_BOUNDS = np.array([[1.0, 10.0], [-5.0, 5.0], [-5.0, 5.0]])


def shifted_sphere(x: np.ndarray) -> float:
    target = np.array([1.5, -2.0])
    return float(np.sum((x - target) ** 2))


def mixed_integer_objective(x: np.ndarray) -> float:
    return float((x[0] - 4.0) ** 2 + (x[1] - 1.5) ** 2 + (x[2] + 0.5) ** 2)


def _assert_nonincreasing_history(history: list[float]) -> None:
    assert len(history) >= 2
    assert all(curr <= prev for prev, curr in zip(history, history[1:]))


def _assert_deterministic(
    optimizer, objective, bounds: np.ndarray, config: OptimizerConfig
) -> None:
    first = optimizer(objective, bounds, config)
    second = optimizer(objective, bounds, config)
    assert np.allclose(first.best_position, second.best_position)
    assert first.best_value == pytest.approx(second.best_value)
    assert first.history == pytest.approx(second.history)
    assert first.iterations == second.iterations
    assert first.seed == second.seed


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


def test_ga_respects_bounds_and_integer_gene():
    result = optimize_ga(
        sphere,
        INTEGER_BOUNDS,
        OptimizerConfig(
            population=12,
            iterations=20,
            seed=1,
            integer_indices=(0,),
        ),
    )
    assert np.all(result.best_position >= INTEGER_BOUNDS[:, 0])
    assert np.all(result.best_position <= INTEGER_BOUNDS[:, 1])
    assert result.best_position[0] == np.rint(result.best_position[0])


def test_gahpw_respects_bounds_and_integer_gene():
    result = optimize_gahpw(
        sphere,
        INTEGER_BOUNDS,
        OptimizerConfig(
            population=12,
            iterations=20,
            seed=1,
            integer_indices=(0,),
        ),
    )
    assert np.all(result.best_position >= INTEGER_BOUNDS[:, 0])
    assert np.all(result.best_position <= INTEGER_BOUNDS[:, 1])
    assert result.best_position[0] == np.rint(result.best_position[0])


@pytest.mark.parametrize(
    ("algorithm", "optimizer"),
    [
        ("pso", optimize_pso),
        ("woa", optimize_woa),
        ("hpw", optimize_hpw),
        ("ga", optimize_ga),
        ("gahpw", optimize_gahpw),
    ],
)
def test_optimizers_are_deterministic_for_fixed_seed(algorithm, optimizer):
    bounds = BOUNDS if algorithm in {"pso", "woa", "hpw"} else INTEGER_BOUNDS
    objective = sphere if algorithm in {"pso", "woa", "hpw"} else mixed_integer_objective
    config = OptimizerConfig(
        population=16,
        iterations=25,
        seed=7,
        integer_indices=(0,) if algorithm in {"ga", "gahpw"} else (),
    )
    _assert_deterministic(optimizer, objective, bounds, config)


@pytest.mark.parametrize(
    ("algorithm", "optimizer", "threshold"),
    [
        ("pso", optimize_pso, 1.0e-3),
        ("woa", optimize_woa, 5.0e-3),
        ("hpw", optimize_hpw, 1.0e-3),
    ],
)
def test_continuous_optimizers_converge_near_shifted_sphere_minimum(
    algorithm, optimizer, threshold
):
    result = optimizer(
        shifted_sphere,
        BOUNDS,
        OptimizerConfig(
            population=24,
            iterations=60,
            seed=3,
            convergence_window=20,
            convergence_tolerance=1.0e-8,
        ),
    )
    _assert_nonincreasing_history(result.history)
    assert result.algorithm == algorithm
    assert result.best_value < threshold
    assert np.allclose(result.best_position, np.array([1.5, -2.0]), atol=0.12)


@pytest.mark.parametrize(
    ("algorithm", "optimizer"),
    [
        ("ga", optimize_ga),
        ("gahpw", optimize_gahpw),
    ],
)
def test_mixed_integer_optimizers_find_integer_optimum(algorithm, optimizer):
    result = optimizer(
        mixed_integer_objective,
        INTEGER_BOUNDS,
        OptimizerConfig(
            population=24,
            iterations=60,
            seed=4,
            integer_indices=(0,),
        ),
    )
    _assert_nonincreasing_history(result.history)
    assert result.algorithm == algorithm
    assert result.best_position[0] == np.rint(result.best_position[0])
    assert result.best_position[0] == pytest.approx(4.0)
    assert np.allclose(result.best_position[1:], np.array([1.5, -0.5]), atol=0.15)
    assert result.best_value < 5.0e-3


@pytest.mark.parametrize("algorithm", ["ga", "pso", "woa", "hpw", "gahpw"])
def test_run_optimizer_dispatches_to_requested_algorithm(algorithm):
    bounds = BOUNDS if algorithm in {"pso", "woa", "hpw"} else INTEGER_BOUNDS
    objective = sphere if algorithm in {"pso", "woa", "hpw"} else mixed_integer_objective
    result = run_optimizer(
        algorithm,
        objective,
        bounds,
        OptimizerConfig(
            population=12,
            iterations=20,
            seed=9,
            integer_indices=(0,) if algorithm in {"ga", "gahpw"} else (),
        ),
    )
    assert result.algorithm == algorithm
    assert result.iterations == len(result.history) - 1


def test_run_optimizer_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unsupported algorithm: bogus"):
        run_optimizer(
            "bogus",
            sphere,
            BOUNDS,
            OptimizerConfig(population=12, iterations=20, seed=1),
        )
