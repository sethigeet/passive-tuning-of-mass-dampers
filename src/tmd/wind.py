from dataclasses import dataclass

import numpy as np

from .spectra import fundamental_angular_frequency
from .types import BuildingConfig, FloorForceExcitation


@dataclass(frozen=True)
class SyntheticWindCaseConfig:
    name: str
    duration_s: float
    dt: float
    seed: int
    coherence_decay: float
    reference_speed_mps: float
    rms_force_n_base: float
    rms_force_n_top: float
    peak_frequency_ratio_base: float
    peak_frequency_ratio_top: float
    bandwidth_ratio: float
    direction: str = "crosswind"


def _linear_profile(start: float, stop: float, count: int) -> np.ndarray:
    if count == 1:
        return np.array([stop], dtype=float)
    return np.linspace(start, stop, count, dtype=float)


def _auto_spectra(
    freqs_hz: np.ndarray,
    rms_force_n: np.ndarray,
    peak_freq_hz: np.ndarray,
    bandwidth_ratio: float,
) -> np.ndarray:
    if len(freqs_hz) < 2:
        raise ValueError("Wind synthesis needs at least two positive frequency bins.")
    delta_f = float(freqs_hz[1] - freqs_hz[0])
    spectra = np.zeros((len(rms_force_n), len(freqs_hz)), dtype=float)
    for story in range(len(rms_force_n)):
        center = max(float(peak_freq_hz[story]), delta_f)
        width = max(center * bandwidth_ratio, delta_f)
        shape = np.exp(-0.5 * ((freqs_hz - center) / width) ** 2)
        area = float(np.sum(shape) * delta_f)
        spectra[story] = shape * (rms_force_n[story] ** 2) / max(area, 1.0e-12)
    return spectra


def _cross_spectral_density(
    freqs_hz: np.ndarray,
    heights_m: np.ndarray,
    auto_spectra: np.ndarray,
    coherence_decay: float,
    reference_speed_mps: float,
) -> np.ndarray:
    n_stories = len(heights_m)
    n_freqs = len(freqs_hz)
    psd = np.zeros((n_freqs, n_stories, n_stories), dtype=complex)
    for freq_index, freq_hz in enumerate(freqs_hz):
        for i in range(n_stories):
            psd[freq_index, i, i] = auto_spectra[i, freq_index]
            for j in range(i + 1, n_stories):
                distance = abs(float(heights_m[i] - heights_m[j]))
                coherence = np.exp(
                    -coherence_decay * freq_hz * distance / max(reference_speed_mps, 1.0e-9)
                )
                value = coherence * np.sqrt(
                    auto_spectra[i, freq_index] * auto_spectra[j, freq_index]
                )
                psd[freq_index, i, j] = value
                psd[freq_index, j, i] = value
    return psd


def _factor_psd_matrix(psd_matrix: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (psd_matrix + psd_matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.clip(eigenvalues.real, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(clipped))


def synthesize_wind_excitation(
    config: BuildingConfig, case: SyntheticWindCaseConfig
) -> FloorForceExcitation:
    n_steps = int(round(case.duration_s / case.dt)) + 1
    if n_steps < 4:
        raise ValueError("Synthetic wind duration must produce at least four samples.")

    time = np.arange(n_steps, dtype=float) * case.dt
    positive_freqs = np.fft.rfftfreq(n_steps, d=case.dt)[1:]
    omega_1 = fundamental_angular_frequency(config)
    fundamental_hz = omega_1 / (2.0 * np.pi)
    heights_m = _linear_profile(1.0, float(config.n_stories), config.n_stories)
    rms_force_n = _linear_profile(
        case.rms_force_n_base, case.rms_force_n_top, config.n_stories
    )
    peak_freq_hz = _linear_profile(
        case.peak_frequency_ratio_base * fundamental_hz,
        case.peak_frequency_ratio_top * fundamental_hz,
        config.n_stories,
    )
    auto_spectra = _auto_spectra(
        positive_freqs, rms_force_n, peak_freq_hz, case.bandwidth_ratio
    )
    cross_psd = _cross_spectral_density(
        positive_freqs,
        heights_m,
        auto_spectra,
        case.coherence_decay,
        case.reference_speed_mps,
    )

    rng = np.random.default_rng(case.seed)
    spectrum = np.zeros((config.n_stories, len(positive_freqs) + 1), dtype=complex)
    delta_f = float(positive_freqs[1] - positive_freqs[0])
    for freq_index in range(len(positive_freqs)):
        factor = _factor_psd_matrix(cross_psd[freq_index])
        random_vector = (
            rng.normal(size=config.n_stories) + 1j * rng.normal(size=config.n_stories)
        ) / np.sqrt(2.0)
        spectrum[:, freq_index + 1] = factor @ random_vector * np.sqrt(delta_f)

    histories = np.fft.irfft(spectrum, n=n_steps, axis=1).T * n_steps
    histories -= np.mean(histories, axis=0, keepdims=True)

    current_rms = np.sqrt(np.mean(histories * histories, axis=0))
    scale = rms_force_n / np.maximum(current_rms, 1.0e-12)
    histories *= scale

    return FloorForceExcitation(
        name=case.name,
        time=time,
        floor_forces_n=histories,
        metadata={
            "source": "synthetic_psd",
            "direction": case.direction,
            "coherence_model": "davenport_style",
            "coherence_decay": case.coherence_decay,
            "reference_speed_mps": case.reference_speed_mps,
            "seed": case.seed,
        },
    )
