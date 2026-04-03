from dataclasses import dataclass

import numpy as np

from .analysis import analyze_controlled, analyze_uncontrolled
from .types import BuildingConfig, DynamicResponse, Record, TMDParameters

try:
    from openseespy import opensees as ops
except Exception:  # pragma: no cover - optional dependency at runtime
    ops = None


@dataclass(frozen=True)
class OpenSeesAvailability:
    available: bool
    reason: str | None = None


def availability() -> OpenSeesAvailability:
    if ops is None:
        return OpenSeesAvailability(
            False, "openseespy is not installed in the active environment"
        )
    return OpenSeesAvailability(True, None)


def _build_opensees_model(config: BuildingConfig, params: TMDParameters | None) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(0, 0.0)
    ops.fix(0, 1)
    for story in range(1, config.n_stories + 1):
        ops.node(story, float(story))
        ops.mass(story, config.story_masses_ton[story - 1] * 1000.0)
    for story in range(1, config.n_stories + 1):
        spring_tag = 1000 + story
        dash_tag = 2000 + story
        mat_tag = 3000 + story
        ops.uniaxialMaterial(
            "Elastic", spring_tag, config.story_stiffness_kn_per_m[story - 1] * 1000.0
        )
        ops.uniaxialMaterial(
            "Viscous", dash_tag, config.story_damping_kns_per_m[story - 1] * 1000.0, 1.0
        )
        ops.uniaxialMaterial("Parallel", mat_tag, spring_tag, dash_tag)
        ops.element("twoNodeLink", 4000 + story, story - 1, story, "-mat", mat_tag, "-dir", 1)
    if params is not None:
        tmd_node = config.n_stories + 1
        ops.node(tmd_node, float(tmd_node))
        ops.mass(tmd_node, params.mass_ton * 1000.0)
        spring_tag = 5001
        dash_tag = 5002
        mat_tag = 5003
        ops.uniaxialMaterial("Elastic", spring_tag, params.stiffness_kn_per_m * 1000.0)
        ops.uniaxialMaterial(
            "Viscous", dash_tag, params.damping_kns_per_m * 1000.0, 1.0
        )
        ops.uniaxialMaterial("Parallel", mat_tag, spring_tag, dash_tag)
        ops.element("twoNodeLink", 5004, config.n_stories, tmd_node, "-mat", mat_tag, "-dir", 1)


def _run_opensees_transient(
    config: BuildingConfig, record: Record, params: TMDParameters | None
) -> DynamicResponse:
    _build_opensees_model(config, params)
    ts_values = list(record.accel_mps2.tolist())
    ops.timeSeries("Path", 1, "-dt", record.dt, "-values", *ts_values)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 20)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    nodes = list(range(1, config.n_stories + 1))
    if params is not None:
        nodes.append(config.n_stories + 1)
    displacements = np.zeros((len(record.time), len(nodes)), dtype=float)
    velocities = np.zeros_like(displacements)
    accelerations = np.zeros_like(displacements)
    for step in range(len(record.time)):
        if step > 0:
            code = ops.analyze(1, record.dt)
            if code != 0:
                raise RuntimeError(
                    f"OpenSees analysis failed at step {step} with code {code}"
                )
        for index, node in enumerate(nodes):
            displacements[step, index] = ops.nodeDisp(node, 1)
            velocities[step, index] = ops.nodeVel(node, 1)
            accelerations[step, index] = ops.nodeAccel(node, 1)
    story_disp = displacements[:, : config.n_stories]
    story_vel = velocities[:, : config.n_stories]
    story_acc = accelerations[:, : config.n_stories]
    peaks = np.max(np.abs(story_disp), axis=0)
    return DynamicResponse(
        time=record.time,
        relative_displacements_m=story_disp,
        relative_velocities_mps=story_vel,
        relative_accelerations_mps2=story_acc,
        peak_story_displacements_m=peaks,
        objective_value=float(peaks[-1]),
        metadata={"solver": "openseespy"},
    )


def analyze_with_backend(
    config: BuildingConfig,
    record: Record,
    params: TMDParameters | None = None,
    backend: str = "auto",
) -> DynamicResponse:
    selected = backend
    if backend == "auto":
        selected = "opensees" if availability().available else "numpy"
    if selected == "opensees":
        if not availability().available:
            raise RuntimeError(availability().reason)
        return _run_opensees_transient(config, record, params)
    if selected == "numpy":
        return (
            analyze_uncontrolled(config, record)
            if params is None
            else analyze_controlled(config, params, record)
        )
    raise ValueError(f"Unsupported backend: {backend}")
