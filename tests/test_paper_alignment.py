import pytest

from tmd.analysis import analyze_uncontrolled
from tmd.benchmarks import get_benchmark
from tmd.io import load_record


def test_example1_uncontrolled_response_matches_paper_scale():
    config = get_benchmark("example1")
    response = analyze_uncontrolled(config, load_record(config.example_record_name))

    assert response.peak_story_displacements_m[0] == pytest.approx(0.031, abs=0.005)
    assert response.peak_story_displacements_m[-1] == pytest.approx(0.188, abs=0.01)


def test_example2_uncontrolled_response_matches_paper_scale():
    config = get_benchmark("example2")
    response = analyze_uncontrolled(config, load_record(config.example_record_name))

    assert response.peak_story_displacements_m[0] == pytest.approx(0.041, abs=0.005)
    assert response.peak_story_displacements_m[-1] == pytest.approx(0.327, abs=0.02)
