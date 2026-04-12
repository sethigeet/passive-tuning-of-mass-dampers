import argparse
import json

from .types import TMDParameters
from .workflows import (
    estimate_equivalent_upgrade,
    run_example,
    run_far_field,
    run_mass_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tmd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "target", choices=["example1", "example2", "mass-sweep", "far-field", "all"]
    )
    run_parser.add_argument(
        "--backend", choices=["auto", "numpy", "opensees"], default="auto"
    )
    run_parser.add_argument("--profile", choices=["fast", "full"], default="full")
    run_parser.add_argument("--no-progress", action="store_true")

    estimate_parser = subparsers.add_parser("estimate-upgrade")
    estimate_parser.add_argument("target", choices=["example1", "example2"])
    estimate_parser.add_argument("table")
    estimate_parser.add_argument("--target-column")
    estimate_parser.add_argument("--s-min", type=float, default=1.0)
    estimate_parser.add_argument("--s-max", type=float, default=4.0)
    estimate_parser.add_argument("--coarse-steps", type=int, default=81)
    estimate_parser.add_argument("--refine-steps", type=int, default=41)
    estimate_parser.add_argument("--refine-rounds", type=int, default=3)
    estimate_parser.add_argument(
        "--upgrade-mass-cost-usd-per-kg",
        type=float,
        default=8.0,
    )
    estimate_parser.add_argument(
        "--upgrade-fixed-cost-usd",
        type=float,
        default=0.0,
    )
    estimate_parser.add_argument("--tmd-mass-ton", type=float)
    estimate_parser.add_argument("--tmd-stiffness-kn-per-m", type=float)
    estimate_parser.add_argument("--tmd-damping-kns-per-m", type=float)

    args = parser.parse_args()
    show_progress = not getattr(args, "no_progress", False)
    if args.command == "run":
        if args.target == "example1":
            payload = run_example(
                "example1",
                backend=args.backend,
                profile=args.profile,
                progress=show_progress,
            )
            print(
                json.dumps(
                    {"example": payload.example.name, "mode": payload.mode},
                    indent=2,
                )
            )
        elif args.target == "example2":
            payload = run_example(
                "example2",
                backend=args.backend,
                profile=args.profile,
                progress=show_progress,
            )
            print(
                json.dumps(
                    {"example": payload.example.name, "mode": payload.mode},
                    indent=2,
                )
            )
        elif args.target == "mass-sweep":
            print(json.dumps(run_mass_sweep(backend=args.backend), indent=2))
        elif args.target == "far-field":
            print(
                json.dumps(
                    run_far_field(
                        backend=args.backend,
                        profile=args.profile,
                        progress=show_progress,
                    ),
                    indent=2,
                )
            )
        elif args.target == "all":
            payload = {
                "example1": {
                    "mode": run_example(
                        "example1",
                        backend=args.backend,
                        profile=args.profile,
                        progress=show_progress,
                    ).mode
                },
                "example2": {
                    "mode": run_example(
                        "example2",
                        backend=args.backend,
                        profile=args.profile,
                        progress=show_progress,
                    ).mode
                },
                "mass_sweep": run_mass_sweep(backend=args.backend)["mode"],
                "far_field": run_far_field(
                    backend=args.backend, profile=args.profile, progress=show_progress
                )["mode"],
            }
            print(json.dumps(payload, indent=2))
        return
    if args.command == "estimate-upgrade":
        tmd_values = (
            args.tmd_mass_ton,
            args.tmd_stiffness_kn_per_m,
            args.tmd_damping_kns_per_m,
        )
        if any(value is not None for value in tmd_values) and not all(
            value is not None for value in tmd_values
        ):
            parser.error(
                "--tmd-mass-ton, --tmd-stiffness-kn-per-m, and --tmd-damping-kns-per-m must be provided together."
            )
        tmd_params = None
        if all(value is not None for value in tmd_values):
            tmd_params = TMDParameters(
                mass_ton=args.tmd_mass_ton,
                stiffness_kn_per_m=args.tmd_stiffness_kn_per_m,
                damping_kns_per_m=args.tmd_damping_kns_per_m,
            )
        payload = estimate_equivalent_upgrade(
            args.target,
            args.table,
            target_column=args.target_column,
            s_min=args.s_min,
            s_max=args.s_max,
            coarse_steps=args.coarse_steps,
            refine_steps=args.refine_steps,
            refine_rounds=args.refine_rounds,
            upgrade_mass_cost_usd_per_kg=args.upgrade_mass_cost_usd_per_kg,
            upgrade_fixed_cost_usd=args.upgrade_fixed_cost_usd,
            tmd_params=tmd_params,
        )
        print(json.dumps(payload, indent=2))
        return
