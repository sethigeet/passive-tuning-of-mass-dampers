from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Array = np.ndarray

AlgorithmName = Literal["ga", "pso", "woa", "hpw", "gahpw"]
ALGORITHMS: tuple[AlgorithmName, ...] = ("ga", "pso", "woa", "hpw", "gahpw")
HazardFamily = Literal["seismic", "wind"]


@dataclass(frozen=True)
class OptimizerConfig:
    population: int
    iterations: int
    seed: int
    convergence_tolerance: float = 1.0e-6
    convergence_window: int = 10
    c1: float = 2.0
    c2: float = 2.0
    inertia_start: float = 1.0
    inertia_end: float = 0.0
    b: float = 1.0
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elite_count: int = 1
    integer_indices: tuple[int, ...] = ()
    show_progress: bool = False
    progress_label: str = ""


@dataclass(frozen=True)
class GlobalOptimizerSettings:
    seed: int
    convergence_window: int
    convergence_tolerance: float


@dataclass(frozen=True)
class PSOOptimizerSettings:
    population: int
    iterations: int
    c1: float
    c2: float
    inertia_start: float
    inertia_end: float

    def to_optimizer_config(
        self,
        global_settings: GlobalOptimizerSettings,
        *,
        show_progress: bool = False,
        progress_label: str = "",
    ) -> OptimizerConfig:
        return OptimizerConfig(
            population=self.population,
            iterations=self.iterations,
            seed=global_settings.seed,
            convergence_tolerance=global_settings.convergence_tolerance,
            convergence_window=global_settings.convergence_window,
            c1=self.c1,
            c2=self.c2,
            inertia_start=self.inertia_start,
            inertia_end=self.inertia_end,
            show_progress=show_progress,
            progress_label=progress_label,
        )


@dataclass(frozen=True)
class WOAOptimizerSettings:
    population: int
    iterations: int
    b: float

    def to_optimizer_config(
        self,
        global_settings: GlobalOptimizerSettings,
        *,
        show_progress: bool = False,
        progress_label: str = "",
    ) -> OptimizerConfig:
        return OptimizerConfig(
            population=self.population,
            iterations=self.iterations,
            seed=global_settings.seed,
            convergence_tolerance=global_settings.convergence_tolerance,
            convergence_window=global_settings.convergence_window,
            b=self.b,
            show_progress=show_progress,
            progress_label=progress_label,
        )


@dataclass(frozen=True)
class HPWOptimizerSettings:
    population: int
    iterations: int
    c1: float
    c2: float
    inertia_start: float
    inertia_end: float
    b: float

    def to_optimizer_config(
        self,
        global_settings: GlobalOptimizerSettings,
        *,
        show_progress: bool = False,
        progress_label: str = "",
    ) -> OptimizerConfig:
        return OptimizerConfig(
            population=self.population,
            iterations=self.iterations,
            seed=global_settings.seed,
            convergence_tolerance=global_settings.convergence_tolerance,
            convergence_window=global_settings.convergence_window,
            c1=self.c1,
            c2=self.c2,
            inertia_start=self.inertia_start,
            inertia_end=self.inertia_end,
            b=self.b,
            show_progress=show_progress,
            progress_label=progress_label,
        )


@dataclass(frozen=True)
class GAOptimizerSettings:
    population: int
    iterations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int
    elite_count: int

    def to_optimizer_config(
        self,
        global_settings: GlobalOptimizerSettings,
        *,
        show_progress: bool = False,
        progress_label: str = "",
    ) -> OptimizerConfig:
        return OptimizerConfig(
            population=self.population,
            iterations=self.iterations,
            seed=global_settings.seed,
            convergence_tolerance=global_settings.convergence_tolerance,
            convergence_window=global_settings.convergence_window,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            tournament_size=self.tournament_size,
            elite_count=self.elite_count,
            show_progress=show_progress,
            progress_label=progress_label,
        )


@dataclass(frozen=True)
class OptimizationProfileSettings:
    global_settings: GlobalOptimizerSettings
    ga: GAOptimizerSettings
    pso: PSOOptimizerSettings
    woa: WOAOptimizerSettings
    hpw: HPWOptimizerSettings
    gahpw: GAOptimizerSettings

    def optimizer_config(
        self,
        algorithm: AlgorithmName,
        *,
        show_progress: bool = False,
        progress_label: str = "",
    ) -> OptimizerConfig:
        match algorithm:
            case "ga":
                return self.ga.to_optimizer_config(
                    self.global_settings,
                    show_progress=show_progress,
                    progress_label=progress_label,
                )
            case "pso":
                return self.pso.to_optimizer_config(
                    self.global_settings,
                    show_progress=show_progress,
                    progress_label=progress_label,
                )
            case "woa":
                return self.woa.to_optimizer_config(
                    self.global_settings,
                    show_progress=show_progress,
                    progress_label=progress_label,
                )
            case "hpw":
                return self.hpw.to_optimizer_config(
                    self.global_settings,
                    show_progress=show_progress,
                    progress_label=progress_label,
                )
            case "gahpw":
                base = self.gahpw.to_optimizer_config(
                    self.global_settings,
                    show_progress=show_progress,
                    progress_label=progress_label,
                )
                return OptimizerConfig(
                    population=base.population,
                    iterations=base.iterations,
                    seed=base.seed,
                    convergence_tolerance=base.convergence_tolerance,
                    convergence_window=base.convergence_window,
                    c1=self.hpw.c1,
                    c2=self.hpw.c2,
                    inertia_start=self.hpw.inertia_start,
                    inertia_end=self.hpw.inertia_end,
                    b=self.hpw.b,
                    crossover_rate=base.crossover_rate,
                    mutation_rate=base.mutation_rate,
                    tournament_size=base.tournament_size,
                    elite_count=base.elite_count,
                    show_progress=show_progress,
                    progress_label=progress_label,
                )


@dataclass(frozen=True)
class AlgorithmConfig:
    profiles: dict[str, OptimizationProfileSettings]

    def profile(self, name: str) -> OptimizationProfileSettings:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ValueError(f"Unknown optimization profile: {name}") from exc


@dataclass(frozen=True)
class TMDParameters:
    mass_ton: float
    stiffness_kn_per_m: float
    damping_kns_per_m: float
    installation_floor: int | None = None


@dataclass(frozen=True)
class BuildingConfig:
    name: str
    story_masses_ton: tuple[float, ...]
    story_stiffness_kn_per_m: tuple[float, ...]
    story_damping_kns_per_m: tuple[float, ...]
    tmd_mass_ton: float
    tmd_mass_bounds_ton: tuple[float, float]
    stiffness_bounds_kn_per_m: tuple[float, float]
    damping_bounds_kns_per_m: tuple[float, float]
    required_records: tuple[str, ...]
    example_record_name: str
    far_field_target_record_name: str

    @property
    def n_stories(self) -> int:
        return len(self.story_masses_ton)


@dataclass(frozen=True)
class BaseAccelerationExcitation:
    name: str
    time: Array
    accel_mps2: Array
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        if len(self.time) < 2:
            raise ValueError("Ground motion record must contain at least two samples.")
        return float(self.time[1] - self.time[0])


Record = BaseAccelerationExcitation


@dataclass(frozen=True)
class FloorForceExcitation:
    name: str
    time: Array
    floor_forces_n: Array
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        if len(self.time) < 2:
            raise ValueError("Wind excitation must contain at least two samples.")
        return float(self.time[1] - self.time[0])


Excitation = BaseAccelerationExcitation | FloorForceExcitation


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
class ExampleRun:
    example: BuildingConfig
    backend: str
    mode: str
    uncontrolled: DynamicResponse | None
    controlled: dict[str, DynamicResponse]
    optimizations: dict[str, OptimizationResult]
    tables: dict[str, Any]
    figures: dict[str, Path]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HazardCase:
    name: str
    family: HazardFamily
    excitation: Excitation
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HazardBundle:
    name: str
    cases: tuple[HazardCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiHazardRun:
    example: BuildingConfig
    backend: str
    mode: str
    hazard_bundle: HazardBundle
    optimization: OptimizationResult
    uncontrolled: dict[str, DynamicResponse]
    controlled: dict[str, DynamicResponse]
    case_objectives: list[dict[str, Any]]
    tables: dict[str, Any]
    figures: dict[str, Path]
    notes: list[str] = field(default_factory=list)
