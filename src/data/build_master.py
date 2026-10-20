"""Build the clean 2019–2023 Texas high-school × year master dataset.

Reads only the raw TEA files under data/raw/tea/ (pro/, TAPR/, retention/; never modifies them) and writes:
  data/processed/master_school_year.csv
  data/processed/school_metadata.csv
  data/processed/temporal_features.csv
  data/processed/subgroup_school_year.csv
  data/processed/data_dictionary.csv
  data/processed/data_quality_report.csv

Run from the repo root:  python3 src/data/build_master.py
See DATA_PIPELINE.md for every rule applied here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "tea" / "pro"
OUT = ROOT / "data" / "processed"

YEARS = [2019, 2020, 2021, 2022, 2023]

# Folder year == data year. Verified: column codes end in the folder's 2-digit year, and the ACT values
# match the old combined file's same year (~80%) but no other year (~2%). The 2023 files are named
# "2024 ..." because they come from the 2023-24 TAPR publication cycle.
# (ccad file, cdrop file, header rows to skip: 2023 files have an English description row above the codes)
YEAR_FILES = {
    2019: ("2019/CCAD.dat", "2019/CDROP_ATT.dat", 0),
    2020: ("2020/CCAD.csv", "2020/CDROP_ATT.csv", 0),
    2021: ("2021/CCAD.csv", "2021/CDROP_ATT.csv", 0),
    2022: ("2022/CCAD.csv", "2022/CDROP_ATT.csv", 0),
    2023: ("2023/2024 Campus College Admissions.csv",
           "2023/2024 Campus Attendance Absenteeism Dropout.csv", 1),
}
# TAPR campus "Profile" (student membership). File Y = TAPR report (Y-1)-Y; its membership columns describe
# school year (Y-1)-Y = our year Y (TEA labels them "Campus Y Student Membership"). Verified: TAPR file Y has
# exactly the campus list of CCAD year Y-1, because both come from TAPR report Y.
TAPR_DIR = ROOT / "data" / "raw" / "tea" / "TAPR"
TAPR_FILES = {2019: "2019.dat", 2020: "2020.dat", 2021: "2021.csv", 2022: "2022.csv", 2023: "2023.csv"}
DEMOGRAPHICS = {   # TAPR column -> (clean name, description, unit)
    "CPETALLC": ("enrollment", "Student membership, all students (count)", "students"),
    "CPETECOP": ("pct_econ_disadv", "% economically disadvantaged", "percent"),
    "CPETLEPP": ("pct_english_learner", "% emergent bilingual / English learner", "percent"),
    "CPETSPEP": ("pct_special_ed", "% special education", "percent"),
    "CPETRSKP": ("pct_at_risk", "% at risk", "percent"),
    "CPETBLAP": ("pct_african_american", "% African American", "percent"),
    "CPETHISP": ("pct_hispanic", "% Hispanic", "percent"),
    "CPETWHIP": ("pct_white", "% White", "percent"),
    "CPETASIP": ("pct_asian", "% Asian", "percent"),
    "CPETINDP": ("pct_american_indian", "% American Indian", "percent"),
    "CPETPCIP": ("pct_pacific_islander", "% Pacific Islander", "percent"),
    "CPETTWOP": ("pct_two_or_more", "% two or more races", "percent"),
    "CPEMALLP": ("mobility_rate", "Mobility rate (TEA reports the PRIOR school year's mobility)", "percent"),
}
RACE_COLS = ["pct_african_american", "pct_hispanic", "pct_white", "pct_asian", "pct_american_indian",
             "pct_pacific_islander", "pct_two_or_more"]
# Grade-level retention: file "(Y-1)(Y)" = students enrolled in school year (Y-1)-Y who were retained in grade
# the following fall -> an outcome of our year Y. 2023-24 (year 2024) is outside the panel; 2018-19 (year 2019)
# was not downloaded, so 2019 retention is missing.
RET_DIR = ROOT / "data" / "raw" / "tea" / "retention"
RET_FILES = {2020: "campus-download-ret-1920.xlsx", 2021: "campus-download-ret-2021.xlsx",
             2022: "campus-download-ret-2122-0.xlsx", 2023: "campus-download-ret-2223.xlsx"}
DIRECTORY = "refernce/Directory.csv"
LOCATIONS = "refernce/Public_School_Locations_-_Current.csv"

# Indicator code (between group code and 2-digit year) -> (clean name, source file kind, description, unit).
# Only indicators published as a rate/average (suffix R) in all five years are kept, so every column means
# the same thing every year. ACT codes verified against the 2023 description row: CAA = ACT composite
# average, CAC = ACT Science (NOT composite, as the earlier audit dictionary assumed).
INDICATORS = {
    "CSA":   ("sat_total",            "ccad",  "SAT average total score (graduates who took SAT)", "score 400-1600"),
    "CSE":   ("sat_erw",              "ccad",  "SAT Evidence-Based Reading & Writing average", "score 200-800"),
    "CSM":   ("sat_math",             "ccad",  "SAT Math average", "score 200-800"),
    "CAA":   ("act_composite",        "ccad",  "ACT composite average", "score 1-36"),
    "CAE":   ("act_english",          "ccad",  "ACT English/ELA average", "score 1-36"),
    "CAM":   ("act_math",             "ccad",  "ACT Math average", "score 1-36"),
    "CAC":   ("act_science",          "ccad",  "ACT Science average", "score 1-36"),
    "CT":    ("sat_act_participation", "ccad", "% of graduates who took SAT or ACT", "percent"),
    "CC":    ("sat_act_above_criterion", "ccad", "% of SAT/ACT test-takers at or above TEA college-ready criterion", "percent"),
    "AT":    ("attendance_rate",      "cdrop", "Attendance rate (days present / days in membership)", "percent"),
    "CA":    ("chronic_absenteeism",  "cdrop", "Chronic absenteeism rate (not published for 2019)", "percent"),
    "912DR": ("dropout_9_12",         "cdrop", "Annual dropout rate, grades 9-12", "percent"),
    "708DR": ("dropout_7_8",          "cdrop", "Annual dropout rate, grades 7-8 (only campuses serving 7-8)", "percent"),
}
TARGETS = ["sat_total", "sat_erw", "sat_math", "act_composite", "act_english", "act_math",
           "sat_act_above_criterion", "sat_act_participation", "dropout_9_12", "attendance_rate",
           "chronic_absenteeism", "retention_rate"]
DEMOG_NAMES = [v[0] for v in DEMOGRAPHICS.values()]
VALID_RANGE = {"sat_total": (400, 1600), "sat_erw": (200, 800), "sat_math": (200, 800),
               "act_composite": (1, 36), "act_english": (1, 36), "act_math": (1, 36), "act_science": (1, 36)}

GROUPS = {"A0": "all", "B0": "african_american", "H0": "hispanic", "W0": "white", "30": "asian",
          "I0": "american_indian", "40": "pacific_islander", "20": "two_or_more", "E0": "econ_disadv",
          "L0": "english_learner", "S0": "special_ed", "R0": "at_risk", "M0": "male", "F0": "female"}

NOT_REPORTED = {".", ""}
SUPPRESSION_CODES = {-1.0: 1, -2.0: 2, -3.0: 3}   # raw code -> <var>_supp (-3 = cross-masked, retention files)

GRADE_TOKENS = {"EE": -3, "PK": -2, "KG": 0}

qa: list[dict] = []


def log(check: str, value, scope: str = "all", detail: str = "") -> None:
    qa.append({"check": check, "scope": scope, "value": value, "detail": detail})
    print(f"[{scope}] {check}: {value}" + (f"  ({detail})" if detail else ""))


def norm_campus_id(s: pd.Series) -> pd.Series:
    """Raw IDs come as 1902001 (2019 .dat, unquoted), '001902001 (Excel quote) or 001902001."""
    s = s.astype(str).str.strip().str.strip('"').str.lstrip("'").str.strip()
    bad = ~s.str.fullmatch(r"\d{1,9}")
    if bad.any():
        raise ValueError(f"non-numeric campus IDs: {s[bad].unique()[:5]}")
    return s.str.zfill(9)


def strip_quote(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lstrip("'").str.strip()


def grades_served(rng: str) -> set[int]:
    """'09-12' -> {9..12}; handles lists like 'PK 09-12', '09-10 12', 'EE KG-05'; AE (adult ed) ignored."""
    out: set[int] = set()
    for tok in rng.replace("EE KG", "EE").split():
        parts = tok.split("-")
        try:
            vals = [GRADE_TOKENS[p] if p in GRADE_TOKENS else int(p) for p in parts if p != "AE"]
        except ValueError:          # 'UG' (ungraded) and other non-grade codes
            continue
        if not vals:
            continue
        hi = 12 if parts[-1] == "AE" else max(vals)
        out.update(range(min(vals), hi + 1))
    return out


# ----------------------------------------------------------------------------------------------- loading
def read_campus_file(rel: str, skip: int) -> pd.DataFrame:
    df = pd.read_csv(RAW / rel, dtype=str, keep_default_na=False, skiprows=skip)
    df["campus_id"] = norm_campus_id(df["CAMPUS"])
    dupe_full = df.duplicated().sum()
    dupe_id = df["campus_id"].duplicated().sum()
    log("raw rows", len(df), rel)
    log("exact duplicate rows", int(dupe_full), rel)
    log("duplicate campus_id (non-identical rows)", int(dupe_id - dupe_full), rel)
    if dupe_id - dupe_full:
        raise ValueError(f"{rel}: conflicting rows share a campus_id")
    return df.drop_duplicates()


def parse_value(raw: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """-> (numeric value with suppression/not-reported as NaN, suppression code 0/1/2, n unparseable)."""
    s = raw.astype(str).str.strip()
    s = s.where(~s.str.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?"), s.str.replace(",", ""))   # "1,040" -> 1040
    num = pd.to_numeric(s.where(~s.isin(NOT_REPORTED)), errors="coerce")
    unparseable = int((num.isna() & ~s.isin(NOT_REPORTED)).sum())
    supp = num.map(SUPPRESSION_CODES).fillna(0).astype("int8")
    val = num.where(supp == 0)
    if (val < 0).any():
        raise ValueError(f"unexpected negative values: {val[val < 0].unique()[:5]}")
    return val, supp, unparseable


def load_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccad_rel, cdrop_rel, skip = YEAR_FILES[year]
    yy = str(year)[2:]
    raw = {"ccad": read_campus_file(ccad_rel, skip), "cdrop": read_campus_file(cdrop_rel, skip)}

    # CCAD and CDROP_ATT must describe the same campuses (checked: identical ID sets every year)
    a, b = set(raw["ccad"].campus_id), set(raw["cdrop"].campus_id)
    log("ccad/cdrop campus sets identical", a == b, str(year), f"ccad-only {len(a - b)}, cdrop-only {len(b - a)}")
    merged = raw["ccad"][["campus_id"]].merge(raw["cdrop"][["campus_id"]], on="campus_id",
                                              how="outer", validate="one_to_one", indicator=True)

    out = merged[["campus_id"]].copy()
    out["year"] = year
    sub_rows = []
    for code, (name, kind, _, _) in INDICATORS.items():
        df = raw[kind].set_index("campus_id").reindex(out["campus_id"])
        for g, gname in GROUPS.items():
            col = f"C{g}{code}{yy}R"
            if col not in df.columns:
                if g == "A0" and not (code == "CA" and year == 2019):
                    raise KeyError(f"{year}: expected column {col} missing")
                continue
            val, supp, bad = parse_value(df[col])
            if bad:
                log("unparseable cells -> missing", bad, str(year), col)
            if g == "A0":
                out[name] = val.values
                out[f"{name}_supp"] = supp.values
            else:
                sub_rows.append(pd.DataFrame({"campus_id": out["campus_id"].values, "year": year,
                                              "group": gname, "indicator": name, "value": val.values,
                                              "suppression": supp.values, "raw_column": col,
                                              "source_file": ccad_rel if kind == "ccad" else cdrop_rel}))
        if f"CA0{code}{yy}R" not in raw[kind].columns:      # chronic absenteeism in 2019
            out[name] = np.nan
            out[f"{name}_supp"] = np.int8(0)
    out["source_ccad"] = ccad_rel
    out["source_cdrop"] = cdrop_rel

    # names reported in the yearly file, used only to check ID stability against the Directory
    for kind in ("ccad", "cdrop"):
        if "CAMPNAME" in raw[kind].columns:
            out["_name_in_file"] = raw[kind].set_index("campus_id")["CAMPNAME"].reindex(out.campus_id).values
            break
    return out, pd.concat(sub_rows, ignore_index=True)


# -------------------------------------------------------------------------------------- directory + geo
def load_directory() -> pd.DataFrame:
    d = pd.read_csv(RAW / DIRECTORY, dtype=str, keep_default_na=False)
    d = d.apply(lambda c: strip_quote(c))
    d["campus_id"] = norm_campus_id(d["School Number"])
    log("directory rows", len(d), DIRECTORY)
    log("directory duplicate campus_id", int(d.campus_id.duplicated().sum()), DIRECTORY)
    d["grades"] = d["Grade Range"].map(grades_served)
    d["serves_9_12"] = d["grades"].map(lambda g: {9, 10, 11, 12} <= g)
    d["serves_12"] = d["grades"].map(lambda g: 12 in g)
    return d


def load_locations() -> pd.DataFrame:
    loc = pd.read_csv(RAW / LOCATIONS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    loc = loc[loc.STATE == "TX"].copy()
    loc["nces_id"] = loc.NCESSCH.str.strip().str.zfill(12)
    log("NCES locations rows (TX)", len(loc), LOCATIONS)
    log("NCES duplicate NCESSCH (TX)", int(loc.nces_id.duplicated().sum()), LOCATIONS)
    return loc


LOCALE_LABELS = {"11": "City-Large", "12": "City-Midsize", "13": "City-Small", "21": "Suburb-Large",
                 "22": "Suburb-Midsize", "23": "Suburb-Small", "31": "Town-Fringe", "32": "Town-Distant",
                 "33": "Town-Remote", "41": "Rural-Fringe", "42": "Rural-Distant", "43": "Rural-Remote"}


def name_key(s: str) -> set[str]:
    stop = {"H", "S", "HS", "HIGH", "SCHOOL", "THE", "OF", "ACADEMY", "EARLY", "COLLEGE", "ECHS", "AND"}
    return {w for w in re.sub(r"[^A-Z0-9 ]", " ", s.upper()).split() if w not in stop}


# ------------------------------------------------------------------------------ demographics + retention
def load_demographics() -> pd.DataFrame:
    rows = []
    for y, f in TAPR_FILES.items():
        t = pd.read_csv(TAPR_DIR / f, dtype=str, keep_default_na=False)
        t["campus_id"] = norm_campus_id(t["CAMPUS"])
        rel = f"TAPR/{f}"
        log("raw rows", len(t), rel)
        log("duplicate campus_id", int(t.campus_id.duplicated().sum()), rel)
        assert not t.campus_id.duplicated().any()
        out = pd.DataFrame({"campus_id": t.campus_id, "year": y})
        for code, (name, _, _) in DEMOGRAPHICS.items():
            val, supp, bad = parse_value(t[code])
            if bad:
                log("unparseable cells -> missing", bad, rel, code)
            out[name] = val.values
            if supp.any():
                log("suppressed cells -> missing", int((supp > 0).sum()), rel, code)
        out["source_demographics"] = "data/raw/tea/" + rel
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def load_retention() -> pd.DataFrame:
    rows = []
    for y, f in RET_FILES.items():
        x = pd.ExcelFile(RET_DIR / f)
        sheet = [s for s in x.sheet_names if s.startswith("Campus_Retention")]
        assert len(sheet) == 1, f"{f}: expected one Campus_Retention sheet, got {x.sheet_names}"
        t = x.parse(sheet[0], dtype=str, keep_default_na=False)
        t = t[t.GRADE.str.strip() == "AG"].copy()        # all grades combined
        t["campus_id"] = norm_campus_id(t["CAMPUS"])
        rel = f"retention/{f}"
        log("all-grades rows", len(t), rel)
        log("duplicate campus_id (AG rows)", int(t.campus_id.duplicated().sum()), rel)
        assert not t.campus_id.duplicated().any()
        out = pd.DataFrame({"campus_id": t.campus_id.values, "year": y})
        for col, name in (("CAMP_ALLR", "retention_rate"), ("CAMP_ALLN", "retention_count"),
                          ("CAMP_ALLD", "retention_denominator")):
            val, supp, bad = parse_value(t[col])
            if bad:
                log("unparseable cells -> missing", bad, rel, col)
            out[name] = val.values
            if name == "retention_rate":
                out["retention_rate_supp"] = supp.values
        out["source_retention"] = "data/raw/tea/" + rel
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------- cohort
def define_cohort(d: pd.DataFrame, panel_all: pd.DataFrame) -> pd.DataFrame:
    """Texas public high schools serving grades 9-12. See DATA_PIPELINE.md, Step 1."""
    # A campus "has graduates" in a year if TEA published an SAT/ACT participation value for it
    # (a real %, 0, or a -1/-2 suppression code). '.' means no graduating class was reported.
    grads = panel_all[panel_all.sat_act_participation.notna() | (panel_all.sat_act_participation_supp > 0)]
    has_grads = set(grads.campus_id)
    all_ids = set(panel_all.campus_id)
    log("campuses in any 2019-2023 file (all levels)", len(all_ids), "cohort")
    log("campuses with a graduating class reported in >=1 year", len(has_grads), "cohort")

    in_dir = d.campus_id.isin(all_ids)
    log("directory campuses appearing in 2019-2023 files", int(in_dir.sum()), "cohort")
    c = d.copy()
    c["rule_regular"] = c["Instruction Type"] == "REGULAR INSTRUCTIONAL"
    c["rule_not_aea"] = c["AEA"] == "N"
    # Grade 12 (not the full 9-12 span) is required: split-campus districts run senior highs such as
    # Plano Senior (11-12) and Allen (10-12) that graduate every senior. serves_full_9_12 flags the strict set.
    c["rule_grade_12"] = c["serves_12"]
    c["rule_has_graduates"] = c.campus_id.isin(has_grads)
    steps = [("directory campuses", np.ones(len(c), bool)),
             ("+ instruction type REGULAR (drops ALTERNATIVE/DAEP/JJAEP)", c.rule_regular),
             ("+ not AEA (alternative accountability)", c.rule_regular & c.rule_not_aea),
             ("+ grade range includes grade 12", c.rule_regular & c.rule_not_aea & c.rule_grade_12),
             ("+ graduating class reported in >=1 year 2019-2023",
              c.rule_regular & c.rule_not_aea & c.rule_grade_12 & c.rule_has_graduates)]
    for label, mask in steps:
        log(f"cohort filter: {label}", int(mask.sum()), "cohort")
    c["in_cohort"] = steps[-1][1]

    # Report what the rules exclude among campuses that did report graduates
    excl = c[c.rule_has_graduates & ~c.in_cohort]
    for reason, m in [("alternative/DAEP/JJAEP instruction", ~excl.rule_regular),
                      ("AEA campus", excl.rule_regular & ~excl.rule_not_aea),
                      ("2025 grade range has no grade 12 (middle/elementary/9-10 campuses)", excl.rule_regular & excl.rule_not_aea & ~excl.rule_grade_12)]:
        log(f"excluded campuses with graduates: {reason}", int(m.sum()), "cohort")
    closed = has_grads - set(d.campus_id)
    log("excluded campuses with graduates: not in 2025 Directory (closed/renumbered)", len(closed), "cohort",
        "no metadata to verify type; survivorship limitation")
    pd.Series(sorted(closed), name="campus_id").to_csv(OUT / "excluded_not_in_directory.csv", index=False)
    log("cohort campuses not serving the full 9-12 span (e.g. 10-12, 11-12)", int((c.in_cohort & ~c.serves_9_12).sum()), "cohort")
    log("under-construction campuses in cohort", int((c.in_cohort & (c["School Status"] != "Active")).sum()), "cohort")
    return c


# --------------------------------------------------------------------------------------------- features
def temporal_features(m: pd.DataFrame) -> pd.DataFrame:
    """Every feature on row (campus, t) uses only years < t, so the row's own value is a valid target.

    Lags are taken from a full campus x year grid, so a missing year yields NaN (never the year before it).
    No imputation: a feature is NaN if any value it needs is missing.
    """
    grid = pd.MultiIndex.from_product([sorted(m.campus_id.unique()), YEARS], names=["campus_id", "year"])
    g = m.set_index(["campus_id", "year"])[TARGETS].reindex(grid)
    feats = {}
    for v in TARGETS:
        wide = g[v].unstack("year")                     # campus x year
        lag = {k: wide.shift(k, axis=1) for k in (1, 2, 3)}
        feats[f"{v}_lag1"] = lag[1]
        feats[f"{v}_lag2"] = lag[2]
        feats[f"{v}_lag3"] = lag[3]
        feats[f"{v}_change_lag1"] = lag[1] - lag[2]                    # last observed year-over-year change
        feats[f"{v}_ma2"] = (lag[1] + lag[2]) / 2                      # mean of t-1, t-2
        feats[f"{v}_ma3"] = (lag[1] + lag[2] + lag[3]) / 3             # mean of t-1..t-3
        # OLS slope over t-3..t-1 (3 equally spaced points); lag2 does not enter the formula but must exist
        feats[f"{v}_trend3"] = ((lag[1] - lag[3]) / 2).where(lag[2].notna())
        feats[f"{v}_n_prior_obs"] = wide.notna().astype(int).cumsum(axis=1).shift(1, axis=1).fillna(0).astype(int)
    # demographics: previous-year snapshot and its change (used as inputs; year-t values are same-year observations)
    gd = m.set_index(["campus_id", "year"])[DEMOG_NAMES].reindex(grid)
    for v in DEMOG_NAMES:
        wide = gd[v].unstack("year")
        feats[f"{v}_lag1"] = wide.shift(1, axis=1)
        feats[f"{v}_change_lag1"] = wide.shift(1, axis=1) - wide.shift(2, axis=1)
    out = pd.concat({k: w.stack(future_stack=True) for k, w in feats.items()}, axis=1)
    out.index.names = ["campus_id", "year"]
    out = out.reset_index()
    # keep only campus-years that exist in the master table
    out = m[["campus_id", "year"]].merge(out, on=["campus_id", "year"], how="left", validate="one_to_one")
    return out


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    years, subs = [], []
    for y in YEARS:
        p, s = load_year(y)
        log("campus rows", len(p), str(y))
        years.append(p)
        subs.append(s)
    panel_all = pd.concat(years, ignore_index=True)
    log("campus-year rows, all campus levels", len(panel_all), "panel")
    log("unique campuses, all levels", panel_all.campus_id.nunique(), "panel")

    d = load_directory()
    cohort = define_cohort(d, panel_all)
    cohort_ids = set(cohort.loc[cohort.in_cohort, "campus_id"])

    # ---------------- metadata (Directory, 2025 snapshot) + geography (NCES, 2024-25)
    md = cohort[cohort.in_cohort].copy()
    md["district_id"] = md["District Number"].str.zfill(6)
    log("district_id == first 6 digits of campus_id", bool((md.district_id == md.campus_id.str[:6]).all()), "metadata")
    md["nces_id"] = md["NCES School ID"].where(md["NCES School ID"].str.fullmatch(r"\d{12}"))
    log("cohort with a 12-digit NCES School ID in Directory", int(md.nces_id.notna().sum()), "metadata",
        f"of {len(md)}")
    loc = load_locations()
    geo = loc[["nces_id", "NAME", "CITY", "LAT", "LON", "LOCALE", "CBSA", "NMCBSA", "NMCNTY", "SCHOOLYEAR"]]
    md = md.merge(geo, on="nces_id", how="left", validate="many_to_one", indicator="_geo")
    n_dup_nces = md.nces_id.dropna().duplicated().sum()
    log("cohort campuses sharing an NCES ID", int(n_dup_nces), "metadata")

    # A match is "reliable" if the NCES ID is unique in the cohort, and the NCES record agrees with the
    # Directory on city or on school-name words. Unreliable matches keep no coordinates.
    matched = md._geo == "both"
    city_ok = md["School City"].str.upper().str.strip() == md["CITY"].str.upper().str.strip()
    name_ok = [bool(name_key(a) & name_key(b)) if isinstance(b, str) else False
               for a, b in zip(md["School Name"], md["NAME"])]
    unique_nces = ~md.nces_id.duplicated(keep=False) | md.nces_id.isna()
    md["geo_match"] = np.select(
        [~matched, matched & unique_nces & (city_ok | pd.Series(name_ok, index=md.index)), matched],
        ["no_nces_record", "reliable", "rejected_name_city_mismatch"], "no_nces_record")
    for k, v in md.geo_match.value_counts().items():
        log(f"geo match: {k}", int(v), "metadata")
    bad = md.geo_match != "reliable"
    md.loc[bad, ["LAT", "LON", "LOCALE", "CBSA", "NMCBSA"]] = np.nan
    md[md.geo_match == "rejected_name_city_mismatch"][["campus_id", "School Name", "School City", "NAME", "CITY"]] \
        .to_csv(OUT / "geo_rejected_matches.csv", index=False)

    meta = pd.DataFrame({
        "campus_id": md.campus_id,
        "campus_name": md["School Name"],
        "district_id": md.district_id,
        "district_name": md["District Name"],
        "district_type": md["District Type"],
        "county_id": md["County Number"].str.zfill(3),
        "county_name": md["County Name"],
        "esc_region": md["ESC Region PEIMS"].str.zfill(2),
        "instruction_type": md["Instruction Type"],
        "charter_type": md["Charter Type"].replace("", "TRADITIONAL (NOT CHARTER)"),
        "is_charter": (md["Charter Type"] != "").astype(int),
        "is_magnet": (md["Magnet Status"] == "Y").astype(int),
        "is_aea": (md["AEA"] == "Y").astype(int),
        "is_virtual_or_hybrid": (md["Virtual/Hybrid Campus"] != "No").astype(int),
        "is_residential_facility": (md["Residential Facility"] == "Y").astype(int),
        "grade_range": md["Grade Range"],
        "serves_full_9_12": md.serves_9_12.astype(int),
        "lowest_grade": md.grades.map(min),
        "school_status_2025": md["School Status"],
        "enrollment_oct2025": pd.to_numeric(md["School Enrollment as of Oct 2025"], errors="coerce")
                                .where(lambda x: x >= 0),
        "nces_school_id": md.nces_id,
        "geo_match": md.geo_match,
        "latitude": pd.to_numeric(md.LAT, errors="coerce"),
        "longitude": pd.to_numeric(md.LON, errors="coerce"),
        "nces_locale_code": md.LOCALE,
        "nces_locale": md.LOCALE.map(LOCALE_LABELS),
        "urbanicity": md.LOCALE.str[0].map({"1": "City", "2": "Suburb", "3": "Town", "4": "Rural"}),
        "cbsa_code": md.CBSA,
        "cbsa_name": md.NMCBSA,
        "source_metadata": DIRECTORY + " (Oct 2025 snapshot)",
        "source_geo": np.where(md.geo_match == "reliable", LOCATIONS + " (2024-25)", ""),
    }).sort_values("campus_id")
    lat_ok = meta.latitude.between(25.5, 36.6) & meta.longitude.between(-106.7, -93.4)
    log("reliable coordinates inside Texas bounding box", f"{int(lat_ok.sum())}/{int(meta.latitude.notna().sum())}", "metadata")

    # ---------------- master table: cohort campuses x the years they appear in the TEA files
    m = panel_all[panel_all.campus_id.isin(cohort_ids)].copy()
    # ID-reuse check: where a yearly file carries a campus name, compare it with the Directory name
    nm = m.dropna(subset=["_name_in_file"]).merge(meta[["campus_id", "campus_name"]], on="campus_id")
    name_match = [bool(name_key(a) & name_key(b)) for a, b in zip(nm._name_in_file, nm.campus_name)]
    log("campus-years where file name shares a word with Directory name", f"{sum(name_match)}/{len(nm)}",
        "master", "low share would suggest reused campus IDs")
    nm.loc[~pd.Series(name_match, index=nm.index), ["campus_id", "year", "_name_in_file", "campus_name"]] \
        .to_csv(OUT / "name_mismatches.csv", index=False)
    m = m.drop(columns="_name_in_file")

    m = m.merge(meta[["campus_id", "district_id", "esc_region"]], on="campus_id", how="left",
                validate="many_to_one")
    dem = load_demographics()
    ret = load_retention()
    m = m.merge(dem, on=["campus_id", "year"], how="left", validate="one_to_one")
    m = m.merge(ret, on=["campus_id", "year"], how="left", validate="one_to_one")
    m["retention_rate_supp"] = m.retention_rate_supp.fillna(0).astype("int8")
    for y in YEARS:
        my = m[m.year == y]
        log("cohort rows matched to TAPR demographics", f"{int(my.source_demographics.notna().sum())}/{len(my)}", str(y))
        log("cohort rows matched to retention file", f"{int(my.source_retention.notna().sum())}/{len(my)}", str(y),
            "2019 not downloaded" if y == 2019 else "")
    race_sum = m[RACE_COLS].sum(axis=1, min_count=len(RACE_COLS))
    log("race shares sum to 100 +/- 1", f"{int(race_sum.between(99, 101).sum())}/{int(race_sum.notna().sum())}", "master")
    for v in DEMOG_NAMES[1:] + ["retention_rate"]:
        log(f"out-of-range values: {v} [0,100]", int((~m[v].between(0, 100) & m[v].notna()).sum()), "master")
    log("retention numerator > denominator", int((m.retention_count > m.retention_denominator).sum()), "master")
    m["covid_affected_year"] = m.year.isin([2020, 2021]).astype(int)
    m["has_graduates_reported"] = (m.sat_act_participation.notna() | (m.sat_act_participation_supp > 0)).astype(int)

    # range checks: out-of-range values are reported, never silently changed
    for v, (lo, hi) in VALID_RANGE.items():
        n = int((~m[v].between(lo, hi) & m[v].notna()).sum())
        log(f"out-of-range values: {v} [{lo},{hi}]", n, "master")
    for v in [i[0] for i in INDICATORS.values() if i[3] == "percent"]:
        log(f"out-of-range values: {v} [0,100]", int((~m[v].between(0, 100) & m[v].notna()).sum()), "master")
    sat_sum = (m.sat_total - m.sat_erw - m.sat_math).abs()
    log("SAT total != ERW + Math (|diff|>1)", int((sat_sum > 1).sum()), "master")

    # uniqueness
    dup = int(m.duplicated(["campus_id", "year"]).sum())
    log("duplicate campus-year rows in master", dup, "master")
    assert dup == 0

    supp_cols = [c for c in m.columns if c.endswith("_supp")]
    order = (["campus_id", "year", "district_id", "esc_region", "covid_affected_year", "has_graduates_reported"]
             + [i[0] for i in INDICATORS.values()] + ["retention_rate", "retention_count", "retention_denominator"]
             + DEMOG_NAMES + supp_cols + ["source_ccad", "source_cdrop", "source_demographics", "source_retention"])
    m = m[order].sort_values(["campus_id", "year"]).reset_index(drop=True)

    # ---------------- per-year coverage and repetition checks
    for y in YEARS:
        my = m[m.year == y]
        log("schools in master", len(my), str(y))
        log("schools with graduates reported", int(my.has_graduates_reported.sum()), str(y))
        for v in TARGETS:
            log(f"non-missing {v}", int(my[v].notna().sum()), str(y))
    per_campus = m.groupby("campus_id").year.nunique()
    log("schools present in all 5 years", int((per_campus == 5).sum()), "master")
    log("schools by number of years present", per_campus.value_counts().sort_index().to_dict(), "master")
    # copy check (the old combined file had SAT copied between years): share of schools whose value equals
    # the previous year's, ignoring boundary values 0 and 100 that legitimately repeat
    for v in [i[0] for i in INDICATORS.values()]:
        w = m.pivot(index="campus_id", columns="year", values=v)
        shares = []
        for y in YEARS[1:]:
            both = w[[y - 1, y]].dropna()
            both = both[~both[y].isin([0, 100])]
            shares.append(f"{y}:{(both[y] == both[y-1]).mean():.1%}" if len(both) else f"{y}:n/a")
        log(f"identical to previous year: {v}", " ".join(shares), "master", "excl. 0 and 100; old file ~99% for SAT")

    # missingness & suppression
    for v in [i[0] for i in INDICATORS.values()]:
        log(f"missing % {v}", round(100 * m[v].isna().mean(), 2), "master")
        log(f"suppressed -1 count {v}", int((m[f"{v}_supp"] == 1).sum()), "master")
        log(f"suppressed -2 count {v}", int((m[f"{v}_supp"] == 2).sum()), "master")
    for v in DEMOG_NAMES + ["retention_rate"]:
        log(f"missing % {v}", round(100 * m[v].isna().mean(), 2), "master")
    log("suppressed -1/-3 count retention_rate", int((m.retention_rate_supp > 0).sum()), "master")
    ind_cols = [i[0] for i in INDICATORS.values()]
    log("overall missing % (indicator cells)", round(100 * m[ind_cols].isna().to_numpy().mean(), 2), "master")
    missing = m[ind_cols].isna().astype(float).drop(columns="dropout_7_8")
    missing.loc[m.year == 2019, "chronic_absenteeism"] = np.nan    # not published in 2019: not applicable
    log("overall missing % excl. chronic_absenteeism 2019 and dropout_7_8",
        round(100 * np.nanmean(missing.to_numpy()), 2), "master")
    has_g = m.has_graduates_reported == 1
    log("missing % of SAT/ACT indicators among rows with graduates reported",
        round(100 * m.loc[has_g, ind_cols[:9]].isna().to_numpy().mean(), 2), "master")

    # ---------------- temporal features
    tf = temporal_features(m)
    assert not tf.duplicated(["campus_id", "year"]).any()
    log("temporal feature rows", len(tf), "temporal")
    log("temporal feature columns", tf.shape[1] - 2, "temporal")
    # leakage self-test: lag1 at year t must equal the raw value at t-1
    chk = m[["campus_id", "year", "act_composite"]].assign(year=lambda x: x.year + 1) \
        .merge(tf[["campus_id", "year", "act_composite_lag1"]], on=["campus_id", "year"])
    ok = np.allclose(chk.act_composite.fillna(-9), chk.act_composite_lag1.fillna(-9))
    log("leakage test: act_composite_lag1(t) == act_composite(t-1)", ok, "temporal")
    assert ok

    # ---------------- subgroup long table (cohort only)
    sub = pd.concat(subs, ignore_index=True)
    sub = sub[sub.campus_id.isin(cohort_ids)]
    assert not sub.duplicated(["campus_id", "year", "group", "indicator"]).any()
    log("subgroup rows", len(sub), "subgroup")

    # ---------------- write
    m.to_csv(OUT / "master_school_year.csv", index=False)
    meta.to_csv(OUT / "school_metadata.csv", index=False)
    tf.to_csv(OUT / "temporal_features.csv", index=False)
    sub.to_csv(OUT / "subgroup_school_year.csv", index=False)
    write_dictionary(m, meta, tf)
    pd.DataFrame(qa).to_csv(OUT / "data_quality_report.csv", index=False)
    print(f"\nwrote outputs to {OUT}")


def write_dictionary(m: pd.DataFrame, meta: pd.DataFrame, tf: pd.DataFrame) -> None:
    rows = [("master_school_year", "campus_id", "9-digit TEA campus number (text, leading zeros kept)", "CAMPUS / 9 Digit Campus Number", "id"),
            ("master_school_year", "year", "Data year = graduating class / school-year end (2019 = 2018-19)", "folder name; verified by column-code suffix", "year"),
            ("master_school_year", "district_id", "6-digit district number (Directory)", DIRECTORY, "id"),
            ("master_school_year", "esc_region", "ESC region 01-20 (Directory, ESC Region PEIMS)", DIRECTORY, "id"),
            ("master_school_year", "covid_affected_year", "1 for 2020 and 2021 (pandemic-disrupted years)", "derived", "flag"),
            ("master_school_year", "has_graduates_reported", "1 if TEA published SAT/ACT participation (value or suppression code) that year", "derived", "flag")]
    for code, (name, kind, desc, unit) in INDICATORS.items():
        src = "CCAD (college admissions)" if kind == "ccad" else "CDROP_ATT (attendance/dropout)"
        rows.append(("master_school_year", name, f"{desc}; All Students; target" if name in TARGETS else f"{desc}; All Students",
                     f"{src}: CA0{code}{{yy}}R", unit))
        rows.append(("master_school_year", f"{name}_supp", f"Suppression of {name}: 0 none, 1 raw -1 (small-group mask), 2 raw -2", f"CA0{code}{{yy}}R", "code"))
    rows += [("master_school_year", "retention_rate", "Grade-level retention rate, all grades (% of students retained in grade the following fall); target; 2019 not available", "Grade-Level Retention campus file, GRADE=AG, CAMP_ALLR", "percent"),
             ("master_school_year", "retention_count", "Students retained (numerator)", "CAMP_ALLN", "students"),
             ("master_school_year", "retention_denominator", "Students enrolled who returned or graduated (denominator)", "CAMP_ALLD", "students"),
             ("master_school_year", "retention_rate_supp", "Suppression of retention_rate: 0 none, 1 raw -1, 3 raw -3 (cross-masked)", "CAMP_ALLR", "code")]
    for code, (name, desc, unit) in DEMOGRAPHICS.items():
        rows.append(("master_school_year", name, f"{desc}; fall membership snapshot of the school year (same-year observation)", f"TAPR campus Profile: {code}", unit))
    rows += [("master_school_year", "source_demographics", "Raw TAPR profile file for this row", "", "path"),
             ("master_school_year", "source_retention", "Raw retention file for this row (empty for 2019)", "", "path"),
             ("master_school_year", "source_ccad", "Raw college-admissions file for this row", "", "path"),
             ("master_school_year", "source_cdrop", "Raw attendance/dropout file for this row", "", "path")]
    for c in meta.columns:
        rows.append(("school_metadata", c, "", "Directory.csv (Oct 2025)" if c not in
                     ("latitude", "longitude", "nces_locale_code", "nces_locale", "urbanicity", "cbsa_code", "cbsa_name", "geo_match")
                     else "Public_School_Locations_-_Current.csv via NCES School ID", ""))
    for c in tf.columns[2:]:
        rows.append(("temporal_features", c, "uses only years before the row's year", "derived from master", ""))
    pd.DataFrame(rows, columns=["file", "column", "description", "source", "unit"]).to_csv(OUT / "data_dictionary.csv", index=False)


if __name__ == "__main__":
    main()
