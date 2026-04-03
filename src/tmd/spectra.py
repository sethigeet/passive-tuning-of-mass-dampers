import math

import numpy as np
from scipy import linalg

from .analysis import newmark_linear
from .models import build_uncontrolled_mck
from .types import BuildingConfig, Record


def fundamental_angular_frequency(config: BuildingConfig) -> float:
    m, _, k = build_uncontrolled_mck(config)
    eigenvalues = linalg.eigvalsh(k, m)
    positive = eigenvalues[eigenvalues > 0.0]
    if len(positive) == 0:
        raise ValueError(f"Benchmark {config.name} has no positive eigenvalues.")
    return float(math.sqrt(positive[0]))


def fundamental_period(config: BuildingConfig) -> float:
    return float(2.0 * math.pi / fundamental_angular_frequency(config))


def pseudo_spectral_acceleration(
    record: Record, period_s: float, damping_ratio: float = 0.05
) -> float:
    if period_s <= 0.0:
        raise ValueError("Spectral period must be positive.")
    wn = 2.0 * math.pi / period_s
    damping = 2.0 * damping_ratio * wn
    response = newmark_linear(
        m=np.array([[1.0]], dtype=float),
        c=np.array([[damping]], dtype=float),
        k=np.array([[wn * wn]], dtype=float),
        record=record,
    )
    return float((wn * wn) * response.peak_story_displacements_m[0])


def scale_record(record: Record, factor: float, name: str | None = None) -> Record:
    return Record(
        name=name or record.name,
        time=record.time.copy(),
        accel_mps2=record.accel_mps2 * factor,
        source_path=record.source_path,
    )


def scale_record_to_target_spectral_acceleration(
    record: Record,
    target_sa_mps2: float,
    period_s: float,
    damping_ratio: float = 0.05,
) -> tuple[Record, float]:
    source_sa = pseudo_spectral_acceleration(record, period_s, damping_ratio)
    if source_sa <= 0.0:
        raise ValueError(f"Record {record.name} has non-positive spectral acceleration.")
    scale_factor = target_sa_mps2 / source_sa
    return scale_record(record, scale_factor, name=record.name), float(scale_factor)
