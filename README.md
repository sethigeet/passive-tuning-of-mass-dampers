# Passive Tuning of Mass Dampers

This repository implements a Python reproduction pipeline for the paper:

- [Finding Optimum Parameters of Passive Tuned Mass Damper by PSO, WOA, and Hybrid PSO-WOA (HPW) Algorithms](https://doi.org/10.22115/scce.2023.352340.1489)

The project is organized around:

- structural matrix assembly for uncontrolled and TMD-controlled shear buildings
- state-space verification utilities
- transient response analysis with a NumPy/SciPy solver and an optional OpenSeesPy backend
- PSO, WOA, and HPW optimizers
- benchmark metadata for the simulation scenarios
- figure, table, and report generation

## Quick start

```bash
uv sync
uv run python scripts/prepare_ground_motions.py
uv run python -m tmd validate --profile fast
uv run python -m tmd reproduce example1 --profile full
uv run python -m tmd reproduce example2 --profile full
uv run python -m tmd reproduce mass-sweep
uv run python -m tmd reproduce far-field --profile fast
uv run python -m tmd reproduce far-field --profile full
uv run python -m tmd reproduce all --profile fast
```

## Data

Use the setup script to download and preprocess the FEMA P695 archive into canonical CSV files:

```bash
uv run python scripts/prepare_ground_motions.py
```

That command will:

- download the public FEMA P695 far-field archive
- extract the original AT2 files
- convert the selected records into canonical CSV files under `data/processed/records/`
- write `configs/records.toml` so the pipeline resolves the prepared files automatically

The CLI now assumes real simulation data is available. If the required records are missing, commands fail instead of falling back to replayed paper tables.

Optimization profiles:

- `fast`: lightweight calibration/debug runs
- `full`: slower paper-style runs with the larger swarm settings
