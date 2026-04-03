import argparse
import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tmd.io import load_peer_at2

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw"
REFERENCE_ROOT = ROOT / "data/processed/records"
DOWNLOAD_DIR = RAW_ROOT / "downloads"
FEMA_DIR = RAW_ROOT / "fema_p695"
ZIP_PATH = DOWNLOAD_DIR / "ATC63_FF_original.zip"
FEMA_URL = "https://s3.us-west-2.amazonaws.com/static-assets.hbrisk.com/ground-motion-sets/2b_ATC-63_Far-Field_GroundMotionAccelTextFiles_Unscaled_Original.zip"
EXTRACTED_ROOT = (
    FEMA_DIR / "ATC-63_Far-Field_GroundMotionAccelTextFiles_Unscaled_Original"
)
MANIFEST_PATH = ROOT / "configs/records.toml"

RECORD_SPECS = {
    "el_centro": ("IMPVALL/H-DLT352.AT2", "Imperial Valley Differential Array"),
    "el_centro_2": ("IMPVALL/H-DLT262.AT2", "Imperial Valley Differential Array 2"),
    "northridge": ("NORTHR/MUL279.AT2", "Northridge Beverly Hills"),
    "duzce_turkey": ("DUZCE/BOL090.AT2", "Duzce Bolu"),
    "hector_mine": ("HECTOR/HEC090.AT2", "Hector Mine Hector"),
    "kobe_japan": ("KOBE/NIS090.AT2", "Kobe Nishi-Akashi"),
    "landers": ("LANDERS/YER270.AT2", "Landers Yermo Fire Station"),
    "manjil_iran": ("MANJIL/ABBAR--T.AT2", "Manjil Abbar"),
}


def download_archive(force: bool) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        return
    with urllib.request.urlopen(FEMA_URL) as response, ZIP_PATH.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_archive(force: bool) -> None:
    FEMA_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRACTED_ROOT.exists() and not force:
        return
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(FEMA_DIR)


def write_csv_exports() -> dict[str, dict[str, str | int | float]]:
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    aliases: dict[str, str] = {}
    summary: dict[str, dict[str, str | int | float]] = {}
    for alias, (relative_at2, label) in RECORD_SPECS.items():
        source = EXTRACTED_ROOT / relative_at2
        record = load_peer_at2(source, alias)
        destination = REFERENCE_ROOT / f"{alias}.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "accel_g"])
            for time_value, accel_mps2 in zip(
                record.time, record.accel_mps2, strict=True
            ):
                writer.writerow([f"{time_value:.6f}", f"{accel_mps2 / 9.80665:.12e}"])
        aliases[alias] = destination.relative_to(ROOT).as_posix()
        summary[alias] = {
            "label": label,
            "source_at2": source.relative_to(ROOT).as_posix(),
            "csv": aliases[alias],
            "npts": int(len(record.time)),
            "dt": float(record.dt),
            "duration_s": float(record.time[-1]),
        }
    write_record_manifest(aliases, summary)
    return summary


def write_record_manifest(
    aliases: dict[str, str], summary: dict[str, dict[str, str | int | float]]
) -> None:
    lines = ["[aliases]"]
    for name, path in aliases.items():
        lines.append(f'{name} = "{path}"')
    lines.append("")
    lines.append("[metadata]")
    for name in aliases:
        lines.append(f'{name} = "{summary[name]["label"]}"')
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "data/processed/records/manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess the FEMA P695 ground motions used by the project."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract the archive even if it already exists.",
    )
    args = parser.parse_args()

    download_archive(force=args.force)
    extract_archive(force=args.force)
    summary = write_csv_exports()
    print(json.dumps({"prepared": summary}, indent=2))


if __name__ == "__main__":
    main()
