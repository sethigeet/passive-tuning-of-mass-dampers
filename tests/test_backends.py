import numpy as np
import pytest

from tmd.backends import analyze_with_backend, availability
from tmd.examples import get_example_config
from tmd.io import synthetic_record
from tmd.reference import get_reference_params


def _assert_matching_response_fields(opensees_response, numpy_response) -> None:
    assert np.allclose(opensees_response.time, numpy_response.time)
    assert np.allclose(
        opensees_response.relative_displacements_m,
        numpy_response.relative_displacements_m,
        rtol=1.0e-3,
        atol=1.0e-6,
    )
    assert np.allclose(
        opensees_response.relative_velocities_mps,
        numpy_response.relative_velocities_mps,
        rtol=1.0e-3,
        atol=1.0e-6,
    )
    assert np.allclose(
        opensees_response.relative_accelerations_mps2,
        numpy_response.relative_accelerations_mps2,
        rtol=1.0e-3,
        atol=1.0e-6,
    )
    assert np.allclose(
        opensees_response.peak_story_displacements_m,
        numpy_response.peak_story_displacements_m,
        rtol=1.0e-3,
        atol=1.0e-6,
    )
    assert opensees_response.objective_value == pytest.approx(
        numpy_response.objective_value,
        rel=1.0e-3,
        abs=1.0e-6,
    )


def test_opensees_backend_tracks_numpy_backend_for_uncontrolled_short_record():
    if not availability("opensees").available:
        pytest.skip("OpenSees backend unavailable")

    config = get_example_config("example1")
    record = synthetic_record("short", duration_s=0.5, dt=0.05)

    numpy_response = analyze_with_backend(config, record, backend="numpy")
    opensees_response = analyze_with_backend(config, record, backend="opensees")

    _assert_matching_response_fields(opensees_response, numpy_response)


def test_opensees_backend_tracks_numpy_backend_for_controlled_short_record():
    if not availability("opensees").available:
        pytest.skip("OpenSees backend unavailable")

    config = get_example_config("example1")
    record = synthetic_record("short", duration_s=0.5, dt=0.05)
    params = get_reference_params("example1", "pso")

    numpy_response = analyze_with_backend(
        config, record, params=params, backend="numpy"
    )
    opensees_response = analyze_with_backend(
        config, record, params=params, backend="opensees"
    )

    _assert_matching_response_fields(opensees_response, numpy_response)
