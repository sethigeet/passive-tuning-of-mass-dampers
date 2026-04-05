from .base import AnalysisBackend, BackendAvailability
from .numpy_backend import numpy_backend
from .opensees_backend import opensees_backend
from ..types import BuildingConfig, DynamicResponse, Record, TMDParameters

BACKENDS: dict[str, AnalysisBackend] = {
    numpy_backend.name: numpy_backend,
    opensees_backend.name: opensees_backend,
}


def get_backend(name: str) -> AnalysisBackend:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend: {name}") from exc


def availability(name: str = "opensees") -> BackendAvailability:
    return get_backend(name).availability()


def analyze_with_backend(
    config: BuildingConfig,
    record: Record,
    params: TMDParameters | None = None,
    backend: str = "auto",
) -> DynamicResponse:
    selected = backend
    if backend == "auto":
        selected = (
            opensees_backend.name
            if opensees_backend.availability().available
            else numpy_backend.name
        )
    return get_backend(selected).analyze(config, record, params=params)


__all__ = [
    "AnalysisBackend",
    "BACKENDS",
    "BackendAvailability",
    "analyze_with_backend",
    "availability",
    "get_backend",
]
