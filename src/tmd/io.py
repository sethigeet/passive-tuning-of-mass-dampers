import re
import tomllib
from pathlib import Path

import numpy as np

from .examples import record_candidates
from .types import BuildingConfig, FloorForceExcitation, Record
from .wind import SyntheticWindCaseConfig, synthesize_wind_excitation

ROOT = Path(__file__).resolve().parents[2]


def _record_aliases() -> dict[str, str]:
    path = ROOT / "configs/records.toml"
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return payload.get("aliases", {})


def load_record(name: str) -> Record:
    aliases = _record_aliases()
    if name in aliases:
        aliased = ROOT / aliases[name]
        if aliased.exists():
            if aliased.suffix.lower() == ".csv":
                data = np.genfromtxt(aliased, delimiter=",", names=True)
                return Record(
                    name=name,
                    time=np.asarray(data["time"], dtype=float),
                    accel_mps2=np.asarray(data["accel_g"], dtype=float) * 9.80665,
                    source_path=aliased,
                )
            if aliased.suffix.lower() == ".at2":
                return load_peer_at2(aliased, name)
    for candidate in record_candidates(name):
        if candidate.exists():
            if candidate.suffix.lower() == ".csv":
                data = np.genfromtxt(candidate, delimiter=",", names=True)
                return Record(
                    name=name,
                    time=np.asarray(data["time"], dtype=float),
                    accel_mps2=np.asarray(data["accel_g"], dtype=float) * 9.80665,
                    source_path=candidate,
                )
            if candidate.suffix.lower() == ".at2":
                return load_peer_at2(candidate, name)
    raise FileNotFoundError(f"No record found for {name} in data/raw")


def load_peer_at2(path: Path, name: str | None = None) -> Record:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    dt = None
    values: list[float] = []
    for line in lines:
        upper = line.upper()
        if "NPTS=" in upper and "DT=" in upper:
            cleaned = upper.replace(",", " ").replace("=", " ")
            parts = cleaned.split()
            dt = float(parts[parts.index("DT") + 1])
            continue
        if "NPTS" in upper and "DT" in upper:
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)
            if len(numbers) >= 2:
                dt = float(numbers[1])
                continue
        try:
            values.extend(float(token) for token in line.split())
        except ValueError:
            continue
    if dt is None:
        raise ValueError(f"Could not parse DT from AT2 file: {path}")
    accel_g = np.asarray(values, dtype=float)
    time = np.arange(len(accel_g), dtype=float) * dt
    return Record(
        name=name or path.stem,
        time=time,
        accel_mps2=accel_g * 9.80665,
        source_path=path,
    )


def synthetic_record(name: str, duration_s: float = 20.0, dt: float = 0.02) -> Record:
    time = np.arange(0.0, duration_s + dt, dt)
    accel_g = 0.25 * np.sin(2.0 * np.pi * 1.15 * time) * np.exp(-0.08 * time)
    return Record(name=name, time=time, accel_mps2=accel_g * 9.80665)


def _require_table(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a TOML table.")
    return {str(key): item for key, item in value.items()}


def _require_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a float.")
    return float(value)


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer.")
    return int(value)


def _require_str(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string.")
    return value


def _load_wind_payload() -> dict[str, object]:
    with (ROOT / "configs/wind.toml").open("rb") as handle:
        return tomllib.load(handle)


def load_wind_bundle(
    config: BuildingConfig, bundle_name: str
) -> tuple[FloorForceExcitation, ...]:
    payload = _load_wind_payload()
    bundle_table = _require_table(payload["bundles"], context="bundles")
    try:
        bundle = _require_table(bundle_table[bundle_name], context=f"bundles.{bundle_name}")
    except KeyError as exc:
        raise ValueError(f"Unknown wind bundle: {bundle_name}") from exc

    case_payload = bundle.get("cases")
    if not isinstance(case_payload, list) or not case_payload:
        raise ValueError(f"bundles.{bundle_name}.cases must be a non-empty array.")

    excitations: list[FloorForceExcitation] = []
    for index, item in enumerate(case_payload):
        context = f"bundles.{bundle_name}.cases[{index}]"
        case = _require_table(item, context=context)
        source = _require_str(case.get("source", "synthetic_psd"), context=f"{context}.source")
        if source != "synthetic_psd":
            raise ValueError(f"Unsupported wind case source: {source}")
        spec = SyntheticWindCaseConfig(
            name=_require_str(case["name"], context=f"{context}.name"),
            duration_s=_require_float(case["duration_s"], context=f"{context}.duration_s"),
            dt=_require_float(case["dt"], context=f"{context}.dt"),
            seed=_require_int(case["seed"], context=f"{context}.seed"),
            coherence_decay=_require_float(
                case["coherence_decay"], context=f"{context}.coherence_decay"
            ),
            reference_speed_mps=_require_float(
                case["reference_speed_mps"], context=f"{context}.reference_speed_mps"
            ),
            rms_force_n_base=_require_float(
                case["rms_force_n_base"], context=f"{context}.rms_force_n_base"
            ),
            rms_force_n_top=_require_float(
                case["rms_force_n_top"], context=f"{context}.rms_force_n_top"
            ),
            peak_frequency_ratio_base=_require_float(
                case["peak_frequency_ratio_base"],
                context=f"{context}.peak_frequency_ratio_base",
            ),
            peak_frequency_ratio_top=_require_float(
                case["peak_frequency_ratio_top"],
                context=f"{context}.peak_frequency_ratio_top",
            ),
            bandwidth_ratio=_require_float(
                case["bandwidth_ratio"], context=f"{context}.bandwidth_ratio"
            ),
            direction=_require_str(case.get("direction", "crosswind"), context=f"{context}.direction"),
        )
        excitations.append(synthesize_wind_excitation(config, spec))
    return tuple(excitations)
