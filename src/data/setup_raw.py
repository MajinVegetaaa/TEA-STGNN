"""Put the raw data into the data/raw/ layout the pipeline expects (see data/raw/README.md).

  python3 src/data/setup_raw.py --zip Dataset.zip [--tapr <folder>] [--retention <folder>]

Extracts the TEA files from Dataset.zip (2019-2023) into the numbered source folders, copies the TAPR and
retention downloads if their folders are given, then checks every file in configs/data_sources.yaml is present.
Files for other years (2018, 2024) are downloaded separately and placed at the paths the config lists. File contents and
names are never changed. Safe to re-run: existing files are left as they are.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile

import yaml
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def target_for(member: str) -> Path | None:
    """Map a Dataset.zip member (pro/...) to its place in data/raw/, or None to skip it."""
    p = PurePosixPath(member)
    parts = p.parts[1:] if p.parts and p.parts[0] == "pro" else p.parts
    if not parts or member.endswith("/"):
        return None
    name = parts[-1]
    if len(parts) == 2 and parts[0].isdigit():
        year = parts[0]
        if name.startswith("CCAD") or "Campus College Admissions" in name:
            return RAW / "01_sat_act" / year / name
        if name.startswith("CDROP") or "Campus Attendance" in name:
            return RAW / "02_attendance_dropout" / year / name
        return RAW / "unused" / "state_benchmarks" / year / name
    if parts[0] == "refernce":
        folder = "05_school_directory" if name.startswith("Directory") else "06_school_locations"
        return RAW / folder / name
    return RAW / "unused" / "previous_audit" / Path(*parts)


def required_files() -> list[Path]:
    """Every raw file listed in configs/data_sources.yaml for the years in use."""
    src = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text())
    req = [RAW / src["directory"], RAW / src["locations"]]
    for source in ("sat_act", "attendance_dropout", "demographics", "retention"):
        req += [RAW / src[source][y]["file"] for y in src["years"] if y in src[source]]
    return req


def copy_folder(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith(".") and not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=ROOT / "Dataset.zip")
    ap.add_argument("--tapr", type=Path, help="folder with the TAPR student-information files (names as in configs/data_sources.yaml)")
    ap.add_argument("--retention", type=Path, help="folder with campus-download-ret-*.xlsx")
    a = ap.parse_args()

    if a.zip.exists():
        with zipfile.ZipFile(a.zip) as z:
            for m in z.namelist():
                t = target_for(m)
                if t is None or t.exists() or "/__MACOSX/" in m or t.name == ".DS_Store":
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as fin, open(t, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        print(f"extracted {a.zip.name} into {RAW}")
    if a.tapr:
        copy_folder(a.tapr, RAW / "03_demographics_tapr")
    if a.retention:
        copy_folder(a.retention, RAW / "04_retention")

    missing = [p.relative_to(ROOT) for p in required_files() if not p.exists()]
    for p in missing:
        print("MISSING", p)
    print("all required raw files present" if not missing else f"{len(missing)} required files missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
