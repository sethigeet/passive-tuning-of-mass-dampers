import numpy as np

from .base import BackendAvailability
from ..models import resolve_tmd_installation_floor
from ..types import (
    BaseAccelerationExcitation,
    BuildingConfig,
    DynamicResponse,
    Excitation,
    FloorForceExcitation,
    TMDParameters,
)

try:
    from openseespy import opensees as ops
except Exception:  # pragma: no cover - optional dependency at runtime
    ops = None


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
        ops.element(
            "twoNodeLink", 4000 + story, story - 1, story, "-mat", mat_tag, "-dir", 1
        )
    if params is not None:
        tmd_node = config.n_stories + 1
        installation_floor = resolve_tmd_installation_floor(config, params)
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
        ops.element(
            "twoNodeLink",
            5004,
            installation_floor,
            tmd_node,
            "-mat",
            mat_tag,
            "-dir",
            1,
        )


def _run_opensees_transient(
    config: BuildingConfig, excitation: Excitation, params: TMDParameters | None
) -> DynamicResponse:
    _build_opensees_model(config, params)
    if isinstance(excitation, BaseAccelerationExcitation):
        ts_values = list(excitation.accel_mps2.tolist())
        ops.timeSeries("Path", 1, "-dt", excitation.dt, "-values", *ts_values)
        ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    elif isinstance(excitation, FloorForceExcitation):
        for story in range(1, config.n_stories + 1):
            ts_tag = 100 + story
            pattern_tag = 200 + story
            values = list(excitation.floor_forces_n[:, story - 1].tolist())
            ops.timeSeries("Path", ts_tag, "-dt", excitation.dt, "-values", *values)
            ops.pattern("Plain", pattern_tag, ts_tag)
            ops.load(story, 1.0)
    else:
        raise TypeError(f"Unsupported excitation: {type(excitation)!r}")
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
    displacements = np.zeros((len(excitation.time), len(nodes)), dtype=float)
    velocities = np.zeros_like(displacements)
    accelerations = np.zeros_like(displacements)
    for step in range(len(excitation.time)):
        if step > 0:
            code = ops.analyze(1, excitation.dt)
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
        time=excitation.time,
        relative_displacements_m=story_disp,
        relative_velocities_mps=story_vel,
        relative_accelerations_mps2=story_acc,
        peak_story_displacements_m=peaks,
        objective_value=float(np.max(peaks)),
        metadata={"solver": "openseespy"},
    )


class OpenSeesBackend:
    name = "opensees"

    def availability(self) -> BackendAvailability:
        if ops is None:
            return BackendAvailability(
                False, "openseespy is not installed in the active environment"
            )
        return BackendAvailability(True, None)

    def analyze(
        self,
        config: BuildingConfig,
        excitation: Excitation,
        params: TMDParameters | None = None,
    ) -> DynamicResponse:
        status = self.availability()
        if not status.available:
            raise RuntimeError(status.reason)
        return _run_opensees_transient(config, excitation, params)


opensees_backend = OpenSeesBackend()
