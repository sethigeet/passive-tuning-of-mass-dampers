import json
from types import SimpleNamespace

from tmd import cli
from tmd.benchmarks import get_benchmark
from tmd.io import load_record, synthetic_record
from tmd.workflows import run_example


def test_prepared_el_centro_record_loads():
    record = load_record("el_centro")
    assert record.dt > 0.0
    assert len(record.time) > 100


def test_prepared_far_field_record_loads():
    record = load_record("northridge")
    assert record.dt > 0.0
    assert len(record.time) > 100


def test_example1_fast_workflow_runs(monkeypatch):
    monkeypatch.setattr("tmd.workflows.publish_run", lambda root, run: {})
    monkeypatch.setattr(
        "tmd.workflows.load_record",
        lambda name: synthetic_record(name, duration_s=1.0, dt=0.05),
    )

    run = run_example("example1", backend="numpy", profile="fast", progress=False)

    assert run.benchmark.name == "example1"
    assert run.mode == "simulate"
    assert set(run.optimizations) == {"pso", "woa", "hpw"}
    assert set(run.controlled) == {"pso", "woa", "hpw"}
    assert run.uncontrolled is not None
    assert run.uncontrolled.peak_story_displacements_m[-1] > 0.0


def test_example2_fast_workflow_runs(monkeypatch):
    monkeypatch.setattr("tmd.workflows.publish_run", lambda root, run: {})
    monkeypatch.setattr(
        "tmd.workflows.load_record",
        lambda name: synthetic_record(name, duration_s=1.0, dt=0.05),
    )

    run = run_example("example2", backend="numpy", profile="fast", progress=False)

    assert run.benchmark.name == "example2"
    assert run.mode == "simulate"
    assert set(run.optimizations) == {"pso", "woa", "hpw"}
    assert set(run.controlled) == {"pso", "woa", "hpw"}
    assert run.uncontrolled is not None
    assert run.uncontrolled.peak_story_displacements_m[-1] > 0.0


def test_example2_uses_benchmark_specific_record(monkeypatch):
    loaded: list[str] = []
    monkeypatch.setattr("tmd.workflows.publish_run", lambda root, run: {})
    monkeypatch.setattr(
        "tmd.workflows.load_record",
        lambda name: (
            loaded.append(name) or synthetic_record(name, duration_s=0.5, dt=0.05)
        ),
    )

    run_example("example2", backend="numpy", profile="fast", progress=False)

    assert loaded[0] == get_benchmark("example2").example_record_name


def test_cli_run_subcommand_emits_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["tmd", "run", "example1", "--backend", "numpy", "--profile", "fast"],
    )
    monkeypatch.setattr(
        cli,
        "run_example",
        lambda *args, **kwargs: SimpleNamespace(
            benchmark=SimpleNamespace(name="example1"), mode="simulate"
        ),
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"benchmark": "example1", "mode": "simulate"}
