"""Independent re-validation of data/processed/ outputs (does not reuse build_master code paths).

Run from the repo root:  python3 src/data/validate_processed.py
Exits non-zero if any check fails. Results are also written to data/processed/processed_validation.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "tea" / "pro"
REQUIRED = ["master_school_year.csv", "school_metadata.csv", "temporal_features.csv",
            "data_quality_report.csv"]
ID = {"campus_id": str, "district_id": str, "esc_region": str, "county_id": str, "nces_school_id": str}

results: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    for f in REQUIRED + ["../../DATA_PIPELINE.md"]:
        p = (P / f).resolve()
        check(f"exists and non-empty: {p.relative_to(ROOT)}", p.exists() and p.stat().st_size > 0)
    m = pd.read_csv(P / "master_school_year.csv", dtype=ID)
    md = pd.read_csv(P / "school_metadata.csv", dtype=ID)
    tf = pd.read_csv(P / "temporal_features.csv", dtype=ID)
    qa = pd.read_csv(P / "data_quality_report.csv")
    check("files readable", True, f"master {m.shape}, metadata {md.shape}, temporal {tf.shape}, qa {qa.shape}")

    # identifiers
    check("campus_id is 9 digits (master)", m.campus_id.str.fullmatch(r"\d{9}").all())
    check("campus_id is 9 digits (metadata)", md.campus_id.str.fullmatch(r"\d{9}").all())
    check("years are exactly 2019-2023", sorted(m.year.unique()) == [2019, 2020, 2021, 2022, 2023])
    check("master unique on (campus_id, year)", not m.duplicated(["campus_id", "year"]).any())
    check("metadata unique on campus_id", not md.campus_id.duplicated().any())
    check("master campuses == metadata campuses", set(m.campus_id) == set(md.campus_id),
          f"{m.campus_id.nunique()} vs {len(md)}")
    check("temporal keys == master keys",
          tf[["campus_id", "year"]].sort_values(["campus_id", "year"]).reset_index(drop=True)
          .equals(m[["campus_id", "year"]].sort_values(["campus_id", "year"]).reset_index(drop=True)))
    check("district_id == campus_id[:6]", (m.district_id == m.campus_id.str[:6]).all())
    j = m.merge(md[["campus_id", "district_id", "esc_region"]], on="campus_id", suffixes=("", "_md"))
    check("master district/region agree with metadata",
          (j.district_id == j.district_id_md).all() and (j.esc_region == j.esc_region_md).all())
    check("esc_region in 01..20", md.esc_region.isin([f"{i:02d}" for i in range(1, 21)]).all())

    # documented counts
    check("1,653 schools / 8,202 rows (as documented)", (md.shape[0], m.shape[0]) == (1653, 8202),
          f"{md.shape[0]} / {m.shape[0]}")
    per_year = m.groupby("year").size().to_dict()
    check("per-year counts as documented",
          per_year == {2019: 1623, 2020: 1633, 2021: 1643, 2022: 1652, 2023: 1651}, str(per_year))
    check("1,621 schools present in all 5 years", (m.groupby("campus_id").size() == 5).sum() == 1621)

    # values: ranges, suppression consistency, no fabricated values
    rng = {"sat_total": (400, 1600), "sat_erw": (200, 800), "sat_math": (200, 800), "act_composite": (1, 36),
           "act_english": (1, 36), "act_math": (1, 36), "act_science": (1, 36)}
    pct = ["sat_act_participation", "sat_act_above_criterion", "attendance_rate", "chronic_absenteeism",
           "dropout_9_12", "dropout_7_8"]
    for v, (lo, hi) in {**rng, **{p: (0, 100) for p in pct}}.items():
        check(f"range {v} [{lo},{hi}]", m[v].dropna().between(lo, hi).all())
        check(f"suppressed => missing: {v}", m.loc[m[f"{v}_supp"] > 0, v].isna().all())
    check("SAT total == ERW + Math", ((m.sat_total - m.sat_erw - m.sat_math).abs().dropna() <= 1).all())
    check("chronic absenteeism all missing in 2019 (not published)", m.loc[m.year == 2019, "chronic_absenteeism"].isna().all())

    # spot-check against raw files: 200 random cells re-read straight from raw
    rs = np.random.default_rng(0)
    raw_files = {2019: ("2019/CCAD.dat", "2019/CDROP_ATT.dat", 0), 2021: ("2021/CCAD.csv", "2021/CDROP_ATT.csv", 0),
                 2023: ("2023/2024 Campus College Admissions.csv", "2023/2024 Campus Attendance Absenteeism Dropout.csv", 1)}
    codes = {"sat_total": ("CSA", 0), "act_composite": ("CAA", 0), "sat_act_participation": ("CT", 0),
             "attendance_rate": ("AT", 1), "dropout_9_12": ("912DR", 1)}
    mism = n = 0
    for y, files in raw_files.items():
        raws = [pd.read_csv(RAW / f, dtype=str, keep_default_na=False, skiprows=files[2]) for f in files[:2]]
        for r in raws:
            r.index = r.CAMPUS.str.strip().str.lstrip("'").str.zfill(9)
        sample = m[m.year == y].sample(40, random_state=int(rs.integers(1e6)))
        for _, row in sample.iterrows():
            for v, (code, k) in codes.items():
                raw = raws[k].at[row.campus_id, f"CA0{code}{str(y)[2:]}R"].strip()
                exp = np.nan if raw in (".", "", "-1", "-2") else float(raw)
                got = row[v]
                n += 1
                mism += not ((np.isnan(exp) and np.isnan(got)) or exp == got)
    check("random cells match raw files", mism == 0, f"{n - mism}/{n} match")

    # temporal features: no future information
    w = m.pivot(index="campus_id", columns="year", values="act_composite")
    t = tf.set_index(["campus_id", "year"])
    ok = True
    for y in [2020, 2021, 2022, 2023]:
        exp = w[y - 1].reindex(t.xs(y, level="year").index)
        got = t.xs(y, level="year")["act_composite_lag1"]
        ok &= np.allclose(exp.fillna(-9), got.fillna(-9))
    check("lag1(t) equals value at t-1 for every year", ok)
    check("no lag features in 2019 (no history)", t.xs(2019, level="year").filter(like="_lag").isna().all().all())
    tr = tf.dropna(subset=["sat_total_trend3"])
    check("trend3 only where lag1, lag2, lag3 all exist",
          tr[["sat_total_lag1", "sat_total_lag2", "sat_total_lag3"]].notna().all().all() and set(tr.year) <= {2022, 2023})

    # geography
    check("every campus has coordinates", md[["latitude", "longitude"]].notna().all().all())
    check("coordinates inside Texas", md.latitude.between(25.5, 36.6).all() and md.longitude.between(-106.7, -93.4).all())
    check("NCES ID unique", not md.nces_school_id.duplicated().any())

    pd.DataFrame(results).to_csv(P / "processed_validation.csv", index=False)
    failed = [r for r in results if not r["passed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
