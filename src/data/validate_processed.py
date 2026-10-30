"""Independent re-validation of data/processed/ outputs (does not reuse build_master code paths).

Run from the repo root:  python3 src/data/validate_processed.py
Exits non-zero if any check fails. Results are also written to data/processed/processed_validation.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REQUIRED = ["master_school_year.csv", "school_metadata.csv", "temporal_features.csv",
            "reports/data_quality_report.csv"]
SRC = yaml.safe_load((ROOT / "configs" / "data_sources.yaml").read_text())
YEARS = [int(y) for y in SRC["years"]]
SKIP = lambda e: int(e.get("header_rows", 1)) - 1
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
    qa = pd.read_csv(P / "reports" / "data_quality_report.csv")
    check("files readable", True, f"master {m.shape}, metadata {md.shape}, temporal {tf.shape}, qa {qa.shape}")

    # identifiers
    check("campus_id is 9 digits (master)", m.campus_id.str.fullmatch(r"\d{9}").all())
    check("campus_id is 9 digits (metadata)", md.campus_id.str.fullmatch(r"\d{9}").all())
    check(f"years are exactly {YEARS[0]}-{YEARS[-1]} (configs/data_sources.yaml)", sorted(m.year.unique()) == YEARS)
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
    qa = qa.set_index(["check", "scope"]).value
    per_year = m.groupby("year").size()
    check("per-year school counts match the build report",
          all(int(qa[("schools in master", str(y))]) == per_year[y] for y in YEARS), per_year.to_dict().__str__())
    check("every metadata school appears in the master table", set(md.campus_id) == set(m.campus_id), f"{len(md)} schools")
    full = int((m.groupby("campus_id").size() == len(YEARS)).sum())
    check(f"schools present in all {len(YEARS)} years match the build report",
          full == int(qa[(f"schools present in all {len(YEARS)} years", "master")]), str(full))

    # values: ranges, suppression consistency, no fabricated values
    rng = {"sat_total": (400, 1600), "sat_erw": (200, 800), "sat_math": (200, 800), "act_composite": (1, 36),
           "act_english": (1, 36), "act_math": (1, 36), "act_science": (1, 36)}
    pct = ["sat_act_participation", "sat_act_above_criterion", "attendance_rate", "chronic_absenteeism",
           "dropout_9_12", "dropout_7_8"]
    for v, (lo, hi) in {**rng, **{p: (0, 100) for p in pct}}.items():
        check(f"range {v} [{lo},{hi}]", m[v].dropna().between(lo, hi).all())
        check(f"suppressed => missing: {v}", m.loc[m[f"{v}_supp"] > 0, v].isna().all())
    sat_gap = (m.sat_total - m.sat_erw - m.sat_math).abs().dropna()
    # TEA rounds each average separately; 3 of 1,231 school-years in 2018 are published 2-4 points apart
    check("SAT total == ERW + Math (within 1 for >= 99.5% of rows, within 5 for all)",
          (sat_gap <= 1).mean() >= 0.995 and (sat_gap <= 5).all(), f"{int((sat_gap > 1).sum())} rows differ by 2-5")
    check("chronic absenteeism missing before 2020 (not published)", m.loc[m.year < 2020, "chronic_absenteeism"].isna().all())

    # spot-check against raw files: 200 random cells re-read straight from raw
    rs = np.random.default_rng(0)
    raw_files = {y: (SRC["sat_act"][y]["file"], SRC["attendance_dropout"][y]["file"],
                     SKIP(SRC["sat_act"][y]), SKIP(SRC["attendance_dropout"][y])) for y in YEARS}
    codes = {"sat_total": ("CSA", 0), "act_composite": ("CAA", 0), "sat_act_participation": ("CT", 0),
             "attendance_rate": ("AT", 1), "dropout_9_12": ("912DR", 1)}
    mism = n = 0
    for y, files in raw_files.items():
        raws = [pd.read_csv(RAW / f, dtype=str, keep_default_na=False, skiprows=k) for f, k in ((files[0], files[2]), (files[1], files[3]))]
        for r in raws:
            r.index = r.CAMPUS.str.strip().str.lstrip("'").str.zfill(9)
        sample = m[m.year == y].sample(25, random_state=int(rs.integers(1e6)))
        for _, row in sample.iterrows():
            for v, (code, k) in codes.items():
                raw = raws[k].at[row.campus_id, f"CA0{code}{str(y)[2:]}R"].strip()
                exp = np.nan if raw in (".", "", "-1", "-2") else float(raw)
                got = row[v]
                n += 1
                mism += not ((np.isnan(exp) and np.isnan(got)) or exp == got)
    check("random cells match raw files", mism == 0, f"{n - mism}/{n} match")

    # demographics (TAPR profile, file Y -> year Y) and retention (file (Y-1)(Y) -> year Y), re-read from raw
    mism = n = 0
    for y in YEARS:
        e = SRC["demographics"][y]
        t = pd.read_csv(RAW / e["file"], dtype=str, keep_default_na=False, skiprows=SKIP(e))
        t.index = t.CAMPUS.str.strip().str.lstrip("'").str.zfill(9)
        for _, row in m[(m.year == y) & m.enrollment.notna()].sample(25, random_state=y).iterrows():
            for col, v in (("CPETALLC", "enrollment"), ("CPETECOP", "pct_econ_disadv"), ("CPETHISP", "pct_hispanic")):
                n += 1
                mism += float(t.at[row.campus_id, col].replace(",", "")) != row[v]
    check("demographic cells match raw TAPR files", mism == 0, f"{n - mism}/{n} match")
    for y in (YEARS[0], YEARS[-1]):
        x = pd.ExcelFile(RAW / SRC["retention"][y]["file"])
        r = x.parse([n for n in x.sheet_names if n.startswith("Campus_Retention")][0], dtype=str, keep_default_na=False)
        r = r[r.GRADE == "AG"].set_index("CAMPUS")
        smp = m[(m.year == y) & m.retention_rate.notna()].sample(40, random_state=0)
        ok = all(float(r.at[c, "CAMP_ALLR"]) == v for c, v in zip(smp.campus_id, smp.retention_rate))
        check(f"retention {y} cells match its raw file (AG rows)", ok, "40 sampled")
    check("retention present for every year that has a file",
          all(m.loc[m.year == y, "retention_rate"].notna().mean() > 0.95 for y in YEARS if y in SRC["retention"]))
    # TSI criteria (TAPR CGRAD1, class Y from the configured file), re-read from raw
    mism = n = 0
    for y in YEARS:
        e = SRC["tsi"][y]
        t = pd.read_csv(RAW / e["file"], dtype=str, keep_default_na=False, skiprows=SKIP(e))
        t.index = t.CAMPUS.str.strip().str.lstrip("'").str.zfill(9)
        for _, row in m[m.year == y].sample(25, random_state=y).iterrows():
            for x, v in (("R", "tsi_reading"), ("M", "tsi_math"), ("B", "tsi_both")):
                raw = t.at[row.campus_id, f"CACR{x}{str(y)[2:]}R"].strip() if row.campus_id in t.index else ""
                exp = np.nan if raw in (".", "", "-1", "-2") else float(raw)      # -1/-2: TEA small-group masks
                n += 1
                mism += not ((np.isnan(exp) and np.isnan(row[v])) or exp == row[v])
    check("TSI cells match raw TAPR files", mism == 0, f"{n - mism}/{n} match")
    tsi = m[["tsi_reading", "tsi_math", "tsi_both"]]
    check("TSI rates within [0, 100]", tsi.stack().between(0, 100).all())
    check("TSI both <= min(reading, math)",
          (m.tsi_both.dropna() <= tsi.loc[m.tsi_both.notna(), ["tsi_reading", "tsi_math"]].min(axis=1)).all())
    check("TSI present for >= 90% of campus-years every year",
          all(m.loc[m.year == y, "tsi_both"].notna().mean() >= 0.9 for y in YEARS),
          " ".join(f"{y}:{m.loc[m.year == y, 'tsi_both'].notna().mean():.0%}" for y in YEARS))
    race = m[[c for c in m.columns if c.startswith("pct_") and c not in
              ("pct_econ_disadv", "pct_english_learner", "pct_special_ed", "pct_at_risk")]].sum(axis=1, min_count=7)
    check("race shares sum to 100 +/- 1", race.dropna().between(99, 101).all())
    check("demographic lag1 equals previous-year value",
          np.allclose(tf.set_index(["campus_id", "year"]).xs(YEARS[-1], level="year")["pct_econ_disadv_lag1"]
                      .reindex(m[m.year == YEARS[-1]].campus_id).fillna(-9).to_numpy(),
                      m[m.year == YEARS[-2]].set_index("campus_id").pct_econ_disadv
                      .reindex(m[m.year == YEARS[-1]].campus_id).fillna(-9).to_numpy()))

    # Academic Risk Index: independent recomputation, range, direction, no future data
    import yaml
    sys.path.insert(0, str(ROOT / "src" / "data"))
    import risk_index as RI
    cfg = yaml.safe_load((ROOT / "configs/risk_index.yaml").read_text())
    w = {k: float(v) for k, v in cfg["weights"].items()}
    ri = m.risk_index.dropna()
    check("risk_index within [0, 100]", ri.between(0, 100).all(), f"n={len(ri)}")
    check("risk_index empty wherever weight coverage < threshold",
          m.loc[m.risk_index_coverage < cfg["min_weight_coverage"], "risk_index"].isna().all())
    rho = m[["risk_index", "dropout_9_12", "retention_rate", "act_composite"]].corr("spearman")["risk_index"]
    check("risk_index rises with dropout and retention, falls with ACT",
          rho.dropout_9_12 > 0 and rho.retention_rate > 0 and rho.act_composite < 0,
          f"rho dropout {rho.dropout_9_12:.2f}, retention {rho.retention_rate:.2f}, ACT {rho.act_composite:.2f}")
    # recompute 40 rows by hand: percentile within year, direction, prior-year decline, re-weighting
    comp = pd.read_csv(P / "risk_index_components.csv", dtype={"campus_id": str})
    bad = 0
    for _, r in m.dropna(subset=["risk_index"]).sample(40, random_state=3).iterrows():
        yr = m[m.year == r.year]
        c = {}
        for v in ("dropout_9_12", "retention_rate"):
            c[v] = (yr[v] < r[v]).sum() / yr[v].notna().sum() + ((yr[v] == r[v]).sum() + 1) / 2 / yr[v].notna().sum() if pd.notna(r[v]) else np.nan
        for v in ("sat_erw", "sat_math", "act_english", "act_math", "sat_act_above_criterion", "sat_act_participation"):
            c[v] = 1 - ((yr[v] < r[v]).sum() + ((yr[v] == r[v]).sum() + 1) / 2) / yr[v].notna().sum() if pd.notna(r[v]) else np.nan
        c["decline"] = comp.set_index(["campus_id", "year"]).at[(r.campus_id, r.year), "risk_c_decline"]
        num = sum(w[k] * c[k] for k in w if pd.notna(c[k]))
        den = sum(w[k] for k in w if pd.notna(c[k]))
        bad += not np.isclose(100 * num / den, r.risk_index, atol=1e-6)
    check("risk_index matches hand recomputation (40 rows)", bad == 0, f"{40 - bad}/40")
    # no future data: scrambling the last year's values must leave every earlier year's index unchanged
    last = YEARS[-1]
    scr = m.copy()
    for v in list(w)[:-1]:
        scr.loc[scr.year == last, v] = np.random.default_rng(0).permutation(scr.loc[scr.year == last, v].to_numpy())
    again = RI.compute(scr, YEARS)
    early = m.year < last
    check(f"risk_index for {YEARS[0]}-{last - 1} unaffected by {last} values",
          np.allclose(again.loc[early, "risk_index"].fillna(-1), m.loc[early, "risk_index"].fillna(-1)))

    # temporal features: no future information
    w = m.pivot(index="campus_id", columns="year", values="act_composite")
    t = tf.set_index(["campus_id", "year"])
    ok = True
    for y in YEARS[1:]:
        exp = w[y - 1].reindex(t.xs(y, level="year").index)
        got = t.xs(y, level="year")["act_composite_lag1"]
        ok &= np.allclose(exp.fillna(-9), got.fillna(-9))
    check("lag1(t) equals value at t-1 for every year", ok)
    check(f"no lag features in {YEARS[0]} (no history)", t.xs(YEARS[0], level="year").filter(like="_lag").isna().all().all())
    tr = tf.dropna(subset=["sat_total_trend3"])
    check("trend3 only where lag1, lag2, lag3 all exist",
          tr[["sat_total_lag1", "sat_total_lag2", "sat_total_lag3"]].notna().all().all() and set(tr.year) <= set(YEARS[3:]))

    # geography
    check("every campus has coordinates", md[["latitude", "longitude"]].notna().all().all())
    check("coordinates inside Texas", md.latitude.between(25.5, 36.6).all() and md.longitude.between(-106.7, -93.4).all())
    check("NCES ID unique", not md.nces_school_id.duplicated().any())

    pd.DataFrame(results).to_csv(P / "reports" / "processed_validation.csv", index=False)
    failed = [r for r in results if not r["passed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
