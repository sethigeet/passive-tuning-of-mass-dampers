import numpy as np

from tmd.examples import get_example_config
from tmd.io import load_wind_bundle
from tmd.wind import SyntheticWindCaseConfig, synthesize_wind_excitation


def _case_config(name: str = "wind_test") -> SyntheticWindCaseConfig:
    return SyntheticWindCaseConfig(
        name=name,
        duration_s=60.0,
        dt=0.1,
        seed=123,
        coherence_decay=10.0,
        reference_speed_mps=30.0,
        rms_force_n_base=1000.0,
        rms_force_n_top=5000.0,
        peak_frequency_ratio_base=0.95,
        peak_frequency_ratio_top=1.15,
        bandwidth_ratio=0.3,
    )


def test_synthetic_wind_generation_is_deterministic_for_fixed_seed():
    config = get_example_config("example1")
    case = _case_config()

    first = synthesize_wind_excitation(config, case)
    second = synthesize_wind_excitation(config, case)

    assert np.allclose(first.time, second.time)
    assert np.allclose(first.floor_forces_n, second.floor_forces_n)


def test_synthetic_wind_generation_matches_rms_profile_and_correlation_order():
    config = get_example_config("example1")
    case = _case_config()

    excitation = synthesize_wind_excitation(config, case)

    rms = np.sqrt(np.mean(excitation.floor_forces_n**2, axis=0))
    expected = np.linspace(case.rms_force_n_base, case.rms_force_n_top, config.n_stories)
    correlations = np.corrcoef(excitation.floor_forces_n.T)

    assert excitation.floor_forces_n.shape == (len(excitation.time), config.n_stories)
    assert np.allclose(rms, expected, rtol=0.15, atol=1.0)
    assert correlations[0, 1] > correlations[0, -1]


def test_load_wind_bundle_builds_expected_number_of_cases():
    config = get_example_config("example1")

    bundle = load_wind_bundle(config, "example1_dev")

    assert len(bundle) == 2
    assert bundle[0].floor_forces_n.shape[1] == config.n_stories
