from __future__ import annotations

import re
import tomllib
from pathlib import Path

import numpy as np

from .benchmarks import record_candidates
from .types import Record

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
