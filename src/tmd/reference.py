from __future__ import annotations

from dataclasses import replace

from .types import TMDParameters

_PAPER_OPTIMA = {
    "example1": {
        "pso": TMDParameters(108.0, 4136.0, 117.50),
        "woa": TMDParameters(108.0, 3365.0, 67.58),
        "hpw": TMDParameters(108.0, 3336.0, 70.08),
    },
    "example2": {
        "pso": TMDParameters(108.0, 493.50, 119.85),
        "woa": TMDParameters(108.0, 253.36, 29.61),
        "hpw": TMDParameters(108.0, 289.87, 25.33),
    },
}


def get_reference_params(
    benchmark_name: str, algorithm: str, mass_ton: float | None = None
) -> TMDParameters:
    try:
        params = _PAPER_OPTIMA[benchmark_name][algorithm]
    except KeyError as exc:
        raise ValueError(
            f"Unknown paper reference pair: benchmark={benchmark_name}, algorithm={algorithm}"
        ) from exc
    if mass_ton is None:
        return params
    return replace(params, mass_ton=mass_ton)
