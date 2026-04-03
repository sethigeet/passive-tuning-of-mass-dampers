import pytest

from tmd.benchmarks import get_benchmark
from tmd.io import load_record
from tmd.spectra import (
    fundamental_period,
    pseudo_spectral_acceleration,
    scale_record_to_target_spectral_acceleration,
)


def test_far_field_record_scaling_matches_target_spectral_acceleration():
    config = get_benchmark("example1")
    period = fundamental_period(config)
    target = load_record(config.far_field_target_record_name)
    target_sa = pseudo_spectral_acceleration(target, period)
    scaled, factor = scale_record_to_target_spectral_acceleration(
        load_record("northridge"), target_sa, period
    )

    assert factor > 0.0
    assert pseudo_spectral_acceleration(scaled, period) == pytest.approx(
        target_sa, rel=1.0e-6
    )
