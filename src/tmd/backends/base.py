from dataclasses import dataclass
from typing import Protocol

from ..types import BuildingConfig, DynamicResponse, Excitation, TMDParameters


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str | None = None


class AnalysisBackend(Protocol):
    name: str

    def availability(self) -> BackendAvailability: ...

    def analyze(
        self,
        config: BuildingConfig,
        excitation: Excitation,
        params: TMDParameters | None = None,
    ) -> DynamicResponse: ...
