import numpy as np

from tmd.analysis import analyze_controlled, analyze_uncontrolled
from tmd.benchmarks import get_benchmark, with_tmd_mass
from tmd.io import load_record
from tmd.reference import get_reference_params


PAPER_DISPLACEMENTS = {
    "example1": {
        "without_tmd": np.array(
            [0.031, 0.060, 0.087, 0.112, 0.133, 0.151, 0.166, 0.177, 0.184, 0.188]
        ),
        "pso": np.array(
            [0.0191, 0.0375, 0.0547, 0.0682, 0.0826, 0.0946, 0.1044, 0.1139, 0.1191, 0.1222]
        ),
        "woa": np.array(
            [0.0185, 0.0361, 0.0524, 0.0673, 0.0812, 0.0933, 0.1033, 0.1110, 0.1164, 0.1191]
        ),
        "hpw": np.array(
            [0.0181, 0.0355, 0.0519, 0.0668, 0.0808, 0.0934, 0.1035, 0.1112, 0.1167, 0.1192]
        ),
    },
    "example2": {
        "without_tmd": np.array(
            [0.041, 0.088, 0.129, 0.166, 0.197, 0.222, 0.252, 0.286, 0.313, 0.327]
        ),
        "pso": np.array(
            [0.0291, 0.0624, 0.0923, 0.1195, 0.1391, 0.1552, 0.1822, 0.2182, 0.2441, 0.2565]
        ),
        "woa": np.array(
            [0.0320, 0.0687, 0.0996, 0.1272, 0.1497, 0.1667, 0.1786, 0.1928, 0.2069, 0.2153]
        ),
        "hpw": np.array(
            [0.0303, 0.0650, 0.0941, 0.1206, 0.1461, 0.1675, 0.1849, 0.1987, 0.2092, 0.2171]
        ),
    },
}

PAPER_MASS_SWEEP_TOP = np.array([0.125, 0.116, 0.114, 0.115, 0.122, 0.130, 0.135])


def _reference_story_displacements(benchmark_name: str) -> dict[str, np.ndarray]:
    config = get_benchmark(benchmark_name)
    record = load_record(config.example_record_name)
    uncontrolled = analyze_uncontrolled(config, record)
    values = {"without_tmd": uncontrolled.peak_story_displacements_m}
    for algorithm in ("pso", "woa", "hpw"):
        values[algorithm] = analyze_controlled(
            config, get_reference_params(benchmark_name, algorithm), record
        ).peak_story_displacements_m
    return values


def test_example1_reference_displacements_track_paper_table():
    values = _reference_story_displacements("example1")
    assert np.max(np.abs(values["without_tmd"] - PAPER_DISPLACEMENTS["example1"]["without_tmd"])) < 0.01
    assert np.max(np.abs(values["pso"] - PAPER_DISPLACEMENTS["example1"]["pso"])) < 0.03
    assert np.max(np.abs(values["woa"] - PAPER_DISPLACEMENTS["example1"]["woa"])) < 0.01
    assert np.max(np.abs(values["hpw"] - PAPER_DISPLACEMENTS["example1"]["hpw"])) < 0.01


def test_example2_reference_displacements_track_paper_table():
    values = _reference_story_displacements("example2")
    assert np.max(np.abs(values["without_tmd"] - PAPER_DISPLACEMENTS["example2"]["without_tmd"])) < 0.04
    assert np.max(np.abs(values["pso"] - PAPER_DISPLACEMENTS["example2"]["pso"])) < 0.06
    assert np.max(np.abs(values["woa"] - PAPER_DISPLACEMENTS["example2"]["woa"])) < 0.02
    assert np.max(np.abs(values["hpw"] - PAPER_DISPLACEMENTS["example2"]["hpw"])) < 0.01


def test_example1_mass_sweep_keeps_paper_shape():
    config = get_benchmark("example1")
    record = load_record(config.example_record_name)
    masses = (90.0, 96.0, 100.0, 104.0, 108.0, 112.0, 116.0)
    top_floor = []
    for mass in masses:
        params = get_reference_params("example1", "pso", mass_ton=mass)
        response = analyze_controlled(with_tmd_mass(config, mass), params, record)
        top_floor.append(response.peak_story_displacements_m[-1])
    top_floor = np.asarray(top_floor)

    assert np.max(np.abs(top_floor - PAPER_MASS_SWEEP_TOP)) < 0.031
    assert int(np.argmin(top_floor)) in {2, 3}
    assert top_floor[0] > top_floor[2]
    assert top_floor[-1] > top_floor[4]
