import numpy as np

from tmd.benchmarks import get_benchmark
from tmd.io import synthetic_record
from tmd.opensees_model import availability, analyze_with_backend
from tmd.reference import get_reference_params


def test_opensees_backend_tracks_numpy_backend_for_short_record():
    if not availability().available:
        return

    config = get_benchmark("example1")
    record = synthetic_record("short", duration_s=0.5, dt=0.05)
    params = get_reference_params("example1", "pso")

    numpy_response = analyze_with_backend(
        config, record, params=params, backend="numpy"
    )
    opensees_response = analyze_with_backend(
        config, record, params=params, backend="opensees"
    )

    assert np.allclose(
        opensees_response.peak_story_displacements_m,
        numpy_response.peak_story_displacements_m,
        rtol=1.0e-3,
        atol=1.0e-6,
    )
