import argparse
import json

from .workflows import run_example, run_far_field, run_mass_sweep


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

    args = parser.parse_args()
    show_progress = not args.no_progress
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
                    {"benchmark": payload.benchmark.name, "mode": payload.mode},
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
                    {"benchmark": payload.benchmark.name, "mode": payload.mode},
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
