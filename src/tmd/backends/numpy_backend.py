import numpy as np

from .base import BackendAvailability
from ..integration import newmark_linear
from ..models import assemble_external_force, build_controlled_mck, build_uncontrolled_mck
from ..types import BuildingConfig, DynamicResponse, Excitation, TMDParameters


def _analyze_uncontrolled(
    config: BuildingConfig, excitation: Excitation
) -> DynamicResponse:
    m, c, k = build_uncontrolled_mck(config)
    return newmark_linear(
        m,
        c,
        k,
        time=excitation.time,
        external=assemble_external_force(config, m, excitation),
    )


def _analyze_controlled(
    config: BuildingConfig, params: TMDParameters, excitation: Excitation
) -> DynamicResponse:
    m, c, k = build_controlled_mck(config, params)
    response = newmark_linear(
        m,
        c,
        k,
        time=excitation.time,
        external=assemble_external_force(config, m, excitation),
    )
    story_disp = response.relative_displacements_m[:, : config.n_stories]
    response.relative_displacements_m = story_disp
    response.relative_velocities_mps = response.relative_velocities_mps[
        :, : config.n_stories
    ]
    response.relative_accelerations_mps2 = response.relative_accelerations_mps2[
        :, : config.n_stories
    ]
    response.peak_story_displacements_m = np.max(np.abs(story_disp), axis=0)
    response.objective_value = float(np.max(response.peak_story_displacements_m))
    return response


class NumPyBackend:
    name = "numpy"

    def availability(self) -> BackendAvailability:
        return BackendAvailability(True, None)

    def analyze(
        self,
        config: BuildingConfig,
        excitation: Excitation,
        params: TMDParameters | None = None,
    ) -> DynamicResponse:
        if params is None:
            return _analyze_uncontrolled(config, excitation)
        return _analyze_controlled(config, params, excitation)


numpy_backend = NumPyBackend()
