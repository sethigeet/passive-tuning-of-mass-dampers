# Passive Tuning of Mass Dampers

Python tooling for reproducing and extending the workflows from:

- [Finding Optimum Parameters of Passive Tuned Mass Damper by PSO, WOA, and Hybrid PSO-WOA (HPW) Algorithms](https://doi.org/10.22115/scce.2023.352340.1489)

The repository contains:

- 10-story example building definitions for the two paper examples
- linear transient response analysis with a NumPy/SciPy Newmark solver
- an OpenSeesPy transient backend exposed through the same workflow API
- a mixed-integer GA+HPW hybrid optimizer for joint TMD floor-placement and parameter search
- PSO, WOA, and hybrid PSO-WOA optimizers for continuous or repaired mixed-integer studies
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

The example workflows use these prepared aliases:

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
- `estimate-upgrade example1 <table.csv>`
- `estimate-upgrade example2 <table.csv>`

Workflow meanings:

- `example1` runs the first 10-story example under its reference record and solves a mixed-integer search over `[floor, mass, stiffness, damping]` with the built-in GA+HPW hybrid.
- `example2` runs the second 10-story example under its own reference record and solves the same mixed-integer TMD placement-and-tuning problem.
- `mass-sweep` keeps the Example 1 reference PSO tuning and varies only the TMD mass to reproduce the mass-sensitivity study.
- `far-field` reruns the Example 1 mixed-integer optimization workflow across the selected FEMA P695 far-field records after scaling them to the target spectral acceleration.
- `all` runs `example1`, `example2`, `mass-sweep`, and `far-field` in sequence.
- `estimate-upgrade` reads a deflection table such as `results/tables/example1_table3.csv` or `results/tables/example2_table11.csv` and searches for the column area scale factor `s` that makes the uncontrolled building match the chosen TMD response using the in-repo Newmark solver.

Common options:

- `--profile {fast,full}` for optimization-heavy workflows
- `--backend {auto,numpy,opensees}`
- `--no-progress` to disable `tqdm` output

`estimate-upgrade` has its own options:

- `--target-column` to choose the controlled-response column from the table. If omitted, the first non-`without_tmd` column is used.
- `--s-min`, `--s-max`, `--coarse-steps`, `--refine-steps`, and `--refine-rounds` to control the scalar search over `s`
- `--upgrade-mass-cost-usd-per-kg` and `--upgrade-fixed-cost-usd` to tune the structural-upgrade cost proxy
- `--tmd-mass-ton`, `--tmd-stiffness-kn-per-m`, and `--tmd-damping-kns-per-m` to override the TMD cost inputs when you do not want the command to infer them from saved results

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

Run the two built-in examples:

```bash
uv run python -m tmd run example1 --profile full --backend numpy
uv run python -m tmd run example2 --profile full --backend numpy
```

Each run optimizes:

- TMD installation floor `p`
- TMD mass `m_d`
- TMD stiffness `k_d`
- TMD damping `c_d`

The workflow objective is to minimize the global peak-displacement ratio plus the damper cost approximation:

```tex
(max_{i,t} |x_i(t)| with TMD / max_i,t |x_i(t)| without TMD) + \alpha * (8m + 150c + 2sqrt(km) + 100,000)
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

Estimate the equivalent structural upgrade for a saved displacement table:

```bash
uv run python -m tmd estimate-upgrade example1 results/tables/example1_table3.csv --target-column gahpw
uv run python -m tmd estimate-upgrade example2 results/tables/example2_table11.csv --target-column gahpw
```

The structural-upgrade search uses the explicit scaling assumptions:

- `M(s) = s M0`
- `K(s) = s^2 K0`
- `C(s) = s^1.5 C0`

The command writes comparison and search CSVs under `results/tables/` and prints a JSON summary including the matched `s`, error metrics, added structural mass, structural-upgrade cost proxy, inferred TMD cost when available, and the estimated cost savings.

The CLI prints JSON summaries to stdout and writes generated artifacts under `results/`.

## Tests

Run the test suite with:

```bash
uv run pytest -q
```

The current tests cover core model assembly, optimizer behavior, state-space utilities, record loading, lightweight example workflow execution, and CLI command dispatch.
