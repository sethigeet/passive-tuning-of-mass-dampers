import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tmd import cli
from tmd.examples import get_example_config
from tmd.io import load_record, synthetic_record
from tmd.workflows import estimate_equivalent_upgrade, run_example


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

    assert run.example.name == "example1"
    assert run.mode == "simulate"
    assert set(run.optimizations) == {"gahpw"}
    assert set(run.controlled) == {"gahpw"}
    assert run.uncontrolled is not None
    assert run.uncontrolled.peak_story_displacements_m[-1] > 0.0
    assert len(run.optimizations["gahpw"].best_position) == 4
    assert run.optimizations["gahpw"].best_position[0] == round(
        run.optimizations["gahpw"].best_position[0]
    )


def test_example2_fast_workflow_runs(monkeypatch):
    monkeypatch.setattr("tmd.workflows.publish_run", lambda root, run: {})
    monkeypatch.setattr(
        "tmd.workflows.load_record",
        lambda name: synthetic_record(name, duration_s=1.0, dt=0.05),
    )

    run = run_example("example2", backend="numpy", profile="fast", progress=False)

    assert run.example.name == "example2"
    assert run.mode == "simulate"
    assert set(run.optimizations) == {"gahpw"}
    assert set(run.controlled) == {"gahpw"}
    assert run.uncontrolled is not None
    assert run.uncontrolled.peak_story_displacements_m[-1] > 0.0


def test_example2_uses_example_specific_record(monkeypatch):
    loaded: list[str] = []
    monkeypatch.setattr("tmd.workflows.publish_run", lambda root, run: {})
    monkeypatch.setattr(
        "tmd.workflows.load_record",
        lambda name: (
            loaded.append(name) or synthetic_record(name, duration_s=0.5, dt=0.05)
        ),
    )

    run_example("example2", backend="numpy", profile="fast", progress=False)

    assert loaded[0] == get_example_config("example2").example_record_name


def test_cli_run_subcommand_emits_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["tmd", "run", "example1", "--backend", "numpy", "--profile", "fast"],
    )
    monkeypatch.setattr(
        cli,
        "run_example",
        lambda *args, **kwargs: SimpleNamespace(
            example=SimpleNamespace(name="example1"), mode="simulate"
        ),
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"example": "example1", "mode": "simulate"}


def test_estimate_equivalent_upgrade_returns_scale_near_one_for_baseline_target(
    monkeypatch, tmp_path
):
    from tmd.integration import newmark_linear
    from tmd.models import build_uncontrolled_mck

    monkeypatch.setattr(
        "tmd.workflows.load_record",
        lambda name: synthetic_record(name, duration_s=0.5, dt=0.05),
    )

    config = get_example_config("example1")
    record = synthetic_record(config.example_record_name, duration_s=0.5, dt=0.05)
    response = newmark_linear(*build_uncontrolled_mck(config), record)
    table_path = tmp_path / "example1_table3.csv"
    pd.DataFrame(
        {
            "story": range(1, config.n_stories + 1),
            "without_tmd": response.peak_story_displacements_m,
            "gahpw": response.peak_story_displacements_m,
        }
    ).to_csv(table_path, index=False)

    payload = estimate_equivalent_upgrade(
        "example1",
        table_path,
        s_min=1.0,
        s_max=2.0,
        coarse_steps=21,
        refine_steps=21,
        refine_rounds=1,
    )

    assert payload["example"] == "example1"
    assert payload["target_column"] == "gahpw"
    assert payload["matched_scale_factor"] == pytest.approx(1.0, abs=0.05)
    assert payload["score"] < 1.0e-6
