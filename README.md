# Passive Tuning of Mass Dampers

Python tooling for reproducing and extending the workflows from:

- [Finding Optimum Parameters of Passive Tuned Mass Damper by PSO, WOA, and Hybrid PSO-WOA (HPW) Algorithms](https://doi.org/10.22115/scce.2023.352340.1489)

The repository contains:

- 10-story benchmark building definitions for the two paper examples
- linear transient response analysis with a NumPy/SciPy Newmark solver
- an OpenSeesPy transient backend exposed through the same workflow API
- PSO, WOA, and hybrid PSO-WOA optimizers
- record preprocessing for the FEMA P695 far-field archive
- CSV/figure/report generation under `results/`

## Requirements

- Python 3.11+
- `uv`

Install the project environment:

```bash
uv sync --extra dev
```

## Data Preparation

The workflows expect prepared ground-motion files to exist locally. Download and preprocess them with:

```bash
uv run python scripts/prepare_ground_motions.py
```

This script:

- downloads the FEMA P695 far-field archive into `data/raw/downloads/`
- extracts the original AT2 files into `data/raw/fema_p695/`
- converts the project’s selected records into canonical CSV files under `data/processed/records/`
- writes `configs/records.toml` so record aliases resolve automatically

The benchmark workflows use these prepared aliases:

- `el_centro`
- `el_centro_2`
- `northridge`
- `duzce_turkey`
- `hector_mine`
- `kobe_japan`
- `landers`
- `manjil_iran`

## CLI

The package installs a `tmd` entry point and also supports `python -m tmd`.

```bash
uv run tmd --help
uv run python -m tmd --help
```

Available commands:

- `run example1`
- `run example2`
- `run mass-sweep`
- `run far-field`
- `run all`

Workflow meanings:

- `example1` runs the paper's first 10-story benchmark under its reference record and optimizes TMD stiffness and damping with PSO, WOA, and HPW.
- `example2` runs the paper's second 10-story benchmark under its own reference record and performs the same TMD optimization study.
- `mass-sweep` keeps the Example 1 reference PSO tuning and varies only the TMD mass to reproduce the mass-sensitivity study.
- `far-field` reruns the Example 1 optimization workflow across the selected FEMA P695 far-field records after scaling them to the target spectral acceleration.
- `all` runs `example1`, `example2`, `mass-sweep`, and `far-field` in sequence.

Common options:

- `--profile {fast,full}` for optimization-heavy workflows
- `--backend {auto,numpy,opensees}`
- `--no-progress` to disable `tqdm` output

Backend behavior:

- `numpy` uses the in-repo Newmark implementation
- `opensees` uses OpenSeesPy explicitly
- `auto` selects OpenSeesPy when it is installed, otherwise NumPy

Because `openseespy` is currently a normal project dependency in `pyproject.toml`, a fresh `uv sync` will usually make `auto` resolve to `opensees`. Use `--backend numpy` when you want the pure Python/SciPy path explicitly.

## Typical Runs

Quick verification:

```bash
uv run pytest -q
```

Run the two built-in benchmark examples:

```bash
uv run python -m tmd run example1 --profile full --backend numpy
uv run python -m tmd run example2 --profile full --backend numpy
```

Run the mass sweep:

```bash
uv run python -m tmd run mass-sweep --backend numpy
```

Run the far-field study:

```bash
uv run python -m tmd run far-field --profile fast --backend numpy
uv run python -m tmd run far-field --profile full --backend numpy
```

Run the bundled suite:

```bash
uv run python -m tmd run all --profile fast --backend numpy
```

The CLI prints JSON summaries to stdout and writes generated artifacts under `results/`.

## Tests

Run the test suite with:

```bash
uv run pytest -q
```

The current tests cover core model assembly, optimizer behavior, state-space utilities, record loading, lightweight example workflow execution, and CLI command dispatch.
