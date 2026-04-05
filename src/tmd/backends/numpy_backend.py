import numpy as np

from .base import BackendAvailability
from ..integration import newmark_linear
from ..models import build_controlled_mck, build_uncontrolled_mck
from ..types import BuildingConfig, DynamicResponse, Record, TMDParameters


def _analyze_uncontrolled(config: BuildingConfig, record: Record) -> DynamicResponse:
    return newmark_linear(*build_uncontrolled_mck(config), record)


def _analyze_controlled(
    config: BuildingConfig, params: TMDParameters, record: Record
) -> DynamicResponse:
    response = newmark_linear(*build_controlled_mck(config, params), record)
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
        record: Record,
        params: TMDParameters | None = None,
    ) -> DynamicResponse:
        if params is None:
            return _analyze_uncontrolled(config, record)
        return _analyze_controlled(config, params, record)


numpy_backend = NumPyBackend()
