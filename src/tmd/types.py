from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TMDParameters:
    mass_ton: float
    stiffness_kn_per_m: float
    damping_kns_per_m: float


@dataclass(frozen=True)
class BuildingConfig:
    name: str
    story_masses_ton: tuple[float, ...]
    story_stiffness_kn_per_m: tuple[float, ...]
    story_damping_kns_per_m: tuple[float, ...]
    tmd_mass_ton: float
    stiffness_bounds_kn_per_m: tuple[float, float]
    damping_bounds_kns_per_m: tuple[float, float]
    required_records: tuple[str, ...] = ("el_centro",)

    @property
    def n_stories(self) -> int:
        return len(self.story_masses_ton)


@dataclass(frozen=True)
class Record:
    name: str
    time: Array
    accel_mps2: Array
    source_path: Path | None = None

    @property
    def dt(self) -> float:
        if len(self.time) < 2:
            raise ValueError("Ground motion record must contain at least two samples.")
        return float(self.time[1] - self.time[0])


@dataclass
class DynamicResponse:
    time: Array
    relative_displacements_m: Array
    relative_velocities_mps: Array
    relative_accelerations_mps2: Array
    peak_story_displacements_m: Array
    objective_value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    algorithm: str
    best_position: Array
    best_value: float
    history: list[float]
    iterations: int
    runtime_s: float
    seed: int


@dataclass
class BenchmarkRun:
    benchmark: BuildingConfig
    backend: str
    mode: str
    uncontrolled: DynamicResponse | None
    controlled: dict[str, DynamicResponse]
    optimizations: dict[str, OptimizationResult]
    tables: dict[str, Any]
    figures: dict[str, Path]
    notes: list[str] = field(default_factory=list)
