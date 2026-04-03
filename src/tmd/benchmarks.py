from dataclasses import replace
from pathlib import Path

from .types import BuildingConfig

_BENCHMARKS = {
    "example1": BuildingConfig(
        name="example1",
        story_masses_ton=(360.0,) * 10,
        story_stiffness_kn_per_m=(650000.0,) * 10,
        story_damping_kns_per_m=(6200.0,) * 10,
        tmd_mass_ton=108.0,
        stiffness_bounds_kn_per_m=(0.0, 5000.0),
        damping_bounds_kns_per_m=(0.0, 1000.0),
        required_records=("el_centro",),
        example_record_name="el_centro",
        far_field_target_record_name="el_centro",
    ),
    "example2": BuildingConfig(
        name="example2",
        story_masses_ton=(
            179.0,
            170.0,
            161.0,
            152.0,
            143.0,
            134.0,
            125.0,
            116.0,
            107.0,
            98.0,
        ),
        story_stiffness_kn_per_m=(
            62470.0,
            52260.0,
            56140.0,
            53020.0,
            49910.0,
            46790.0,
            43670.0,
            40550.0,
            37430.0,
            34310.0,
        ),
        story_damping_kns_per_m=(
            805.863,
            674.154,
            724.206,
            683.958,
            643.839,
            603.591,
            563.095,
            523.098,
            482.847,
            442.592,
        ),
        tmd_mass_ton=108.0,
        stiffness_bounds_kn_per_m=(0.0, 500.0),
        damping_bounds_kns_per_m=(0.0, 150.0),
        required_records=("el_centro_2",),
        example_record_name="el_centro_2",
        far_field_target_record_name="el_centro",
    ),
}


def get_benchmark(name: str) -> BuildingConfig:
    try:
        return _BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {name}") from exc


def record_candidates(record_name: str) -> list[Path]:
    slug = record_name.lower().replace(",", "").replace(" ", "_")
    return [
        Path("data/raw") / f"{slug}.csv",
        Path("data/raw") / f"{slug}.at2",
    ]


def with_tmd_mass(config: BuildingConfig, mass_ton: float) -> BuildingConfig:
    return replace(config, tmd_mass_ton=mass_ton)
