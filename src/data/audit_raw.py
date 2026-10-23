"""Integrity audit of the raw downloads: is every file the right dataset, for the right year, and not a copy?

  python3 src/data/audit_raw.py        (writes data/processed/reports/raw_audit.csv; exits 1 on any FAIL)

Checks, per source:
  1. no two files are byte-identical (a re-saved or mislabelled copy)
  2. each file is the dataset it claims to be (expected columns / title)
  3. each file's year is proven from its contents, not just its name
  4. values are not copied between consecutive years (the old Excel file's SAT problem)
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "reports" / "raw_audit.csv"
SRC = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text())
YEARS = [int(y) for y in SRC["years"]]
SKIP = lambda e: int(e.get("header_rows", 1)) - 1
rows: list[dict] = []


def rec(source, check, ok, detail=""):
    rows.append({"source": source, "check": check, "result": "PASS" if ok else ("INFO" if ok is None else "FAIL"),
                 "detail": detail})
    print(f"{'PASS' if ok else ('INFO' if ok is None else 'FAIL'):4}  [{source}] {check}" + (f"  ({detail})" if detail else ""))


def ids(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.strip('"').str.lstrip("'").str.zfill(9)


def read(path: Path, skip: int = 0) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, skiprows=skip)


def copy_rate(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """Share of campuses whose value is identical in both years, ignoring blanks, masks and 0/100."""
    x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna() & (x > 0) & (y > 0) & ~x.isin([100]) & ~y.isin([100])   # -1/-2 masks, 0 and 100 repeat legitimately
    return float((x[ok] == y[ok]).mean()) if ok.any() else np.nan, int(ok.sum())


def main() -> int:
    # 1. byte-identical files anywhere in data/raw (excluding unused/)
    files = [p for p in RAW.rglob("*") if p.is_file() and "unused" not in p.parts and p.suffix != ".md"
             and not p.name.startswith(".")]
    digest = {}
    for p in files:
        digest.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p.relative_to(RAW))
    dups = [v for v in digest.values() if len(v) > 1]
    rec("all", "no two raw files are byte-identical", not dups, f"{len(files)} files" + (f"; dups {dups}" if dups else ""))

    # 2-4. SAT/ACT and attendance/dropout: year from column-code suffix; copy check
    sat = {y: read(RAW / SRC["sat_act"][y]["file"], SKIP(SRC["sat_act"][y])) for y in YEARS}
    att = {y: read(RAW / SRC["attendance_dropout"][y]["file"], SKIP(SRC["attendance_dropout"][y])) for y in YEARS}
    for name, d, key in (("01_sat_act", sat, "CA0CSA"), ("02_attendance_dropout", att, "CA0AT")):
        for y, t in d.items():
            yy = {int(m.group(1)) for c in t.columns if (m := re.fullmatch(r"C..[A-Z0-9]+?(\d\d)[RNDA]", c))}
            # older (pre-2019) TEA files also carry the prior year's columns; the newer ones carry only their own
            rec(name, f"{y}: column codes carry year {y % 100} (plus at most the prior year)",
                y % 100 in yy and yy <= {y % 100, (y - 1) % 100}, f"years in codes {sorted(yy)}")
            rec(name, f"{y}: contains its key indicator {key}{y % 100}R", f"{key}{y % 100}R" in t.columns)
        for y in YEARS[1:]:
            a, b = d[y - 1].set_index(ids(d[y - 1].CAMPUS)), d[y].set_index(ids(d[y].CAMPUS))
            c = a.index.intersection(b.index)
            r, n = copy_rate(a.loc[c, f"{key}{(y - 1) % 100}R"], b.loc[c, f"{key}{y % 100}R"])
            rec(name, f"{y - 1}->{y}: values not copied between years", r < 0.2, f"{r:.1%} identical of {n}")

    # TAPR profile: right category, year proven by campus-list match with SAT/ACT year Y-1 (same TEA report)
    tapr = {y: read(RAW / SRC["demographics"][y]["file"], SKIP(SRC["demographics"][y])) for y in YEARS}
    sat_ids = {y: set(ids(t.CAMPUS)) for y, t in sat.items()}
    for y, f in SRC.get("not_yet_used", {}).get("sat_act", {}).items():     # extra years only used as year evidence
        sat_ids[int(y)] = set(ids(read(RAW / f).CAMPUS))
    for y, t in tapr.items():
        need = {"CPETALLC", "CPETECOP", "CPETLEPP", "CPETSPEP", "CPETHISP", "CPEMALLP"}
        rec("03_demographics_tapr", f"{y}: is the campus student-information download", need <= set(t.columns),
            f"{t.shape[1]} columns")
        tid = set(ids(t.CAMPUS))
        match = [k for k, v in sat_ids.items() if v == tid]
        rec("03_demographics_tapr", f"{y}: campus list equals SAT/ACT year {y - 1} (same TEA report => year {y})",
            match == [y - 1] if y - 1 in sat_ids else None, f"exact match with {match}")
    for y in YEARS[1:]:
        a, b = tapr[y - 1].set_index(ids(tapr[y - 1].CAMPUS)), tapr[y].set_index(ids(tapr[y].CAMPUS))
        c = a.index.intersection(b.index)
        r, n = copy_rate(a.loc[c, "CPETECOP"], b.loc[c, "CPETECOP"])
        rec("03_demographics_tapr", f"{y - 1}->{y}: % econ. disadvantaged not copied", r < 0.2, f"{r:.1%} identical of {n}")
        e = pd.to_numeric(b.loc[c, "CPETALLC"], errors="coerce").sum() / pd.to_numeric(a.loc[c, "CPETALLC"], errors="coerce").sum()
        rec("03_demographics_tapr", f"{y - 1}->{y}: total enrollment changes plausibly", 0.9 < e < 1.1, f"ratio {e:.3f}")

    # retention: title row names the school year; file (Y-1)(Y) -> year Y
    ret = {}
    for y in YEARS:
        f = SRC["retention"][y]["file"]
        x = pd.ExcelFile(RAW / f)
        title = str(x.parse("Overview", header=None).iloc[0, 0])
        want = f"{y - 1}-{str(y)[2:]}"
        rec("04_retention", f"{f}: title says school year {want} (=> year {y})", want in title, title[:80])
        sheet = [s for s in x.sheet_names if s.startswith("Campus_Retention")][0]
        t = x.parse(sheet, dtype=str, keep_default_na=False)
        rec("04_retention", f"{f}: campus-level, all-grades rows present", (t.GRADE == "AG").sum() > 8000,
            f"{int((t.GRADE == 'AG').sum())} AG rows")
        ret[y] = t[t.GRADE == "AG"].set_index("CAMPUS")
    for y in YEARS[1:]:
        a, b = ret[y - 1], ret[y]
        c = a.index.intersection(b.index)
        r, n = copy_rate(a.loc[c, "CAMP_ALLR"].str.replace(",", ""), b.loc[c, "CAMP_ALLR"].str.replace(",", ""))
        rec("04_retention", f"{y - 1}->{y}: retention rate not copied", r < 0.2, f"{r:.1%} identical of {n}")

    # directory and locations: right dataset
    d = read(RAW / SRC["directory"])
    rec("05_school_directory", "is the TEA school directory", {"School Number", "Grade Range", "Instruction Type",
                                                             "NCES School ID"} <= set(d.columns), f"{len(d)} rows")
    loc = pd.read_csv(RAW / SRC["locations"], dtype=str,
                      usecols=["NCESSCH", "STATE", "LAT", "LON", "SCHOOLYEAR"], encoding="utf-8-sig")
    rec("06_school_locations", "is NCES school locations with Texas schools", (loc.STATE == "TX").sum() > 9000,
        f"{int((loc.STATE == 'TX').sum())} TX schools; school year {loc.SCHOOLYEAR.mode().iloc[0]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT, index=False)
    print(f"\n{(rep.result == 'PASS').sum()} pass, {(rep.result == 'FAIL').sum()} fail, {(rep.result == 'INFO').sum()} info")
    return 1 if (rep.result == "FAIL").any() else 0


if __name__ == "__main__":
    sys.exit(main())
