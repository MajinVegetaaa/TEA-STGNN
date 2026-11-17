"""Verified additive sources for the existing campus/year master builder.

No cohort selection, imputation, rolling features, or modeling happens here.
Raw tokens and a status per value distinguish masks, ambiguous blanks and absent
campuses. In particular, AP/IB blanks may mean undefined OR suppressed.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import html

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
STAFF = {
    "teacher_experience": ("CPSTEXPA", "Teacher average experience", "years"),
    "teacher_tenure": ("CPSTTENA", "Teacher average tenure with district", "years"),
    "teacher_salary": ("CPSTTOSA", "Teacher average base salary", "USD"),
    "student_teacher_ratio": ("CPSTKIDR", "Students per teacher FTE", "ratio"),
    "teacher_fte": ("CPSTTOFC", "Teacher full-time-equivalent count", "FTE"),
}
EXAMS = {
    "participation": ("Part_Rate", "Grades 11-12 participation"),
    "examinees_above_criterion": ("Exnees_Above_Crit_Rate", "Examinees meeting criterion on at least one exam"),
    "exams_above_criterion": ("Exams_Above_Crit_Rate", "Exams meeting criterion"),
}
GRAD = {
    "graduation_4yr": ("CAMP_ALLR_GRAD", "Four-year federal graduation rate", "percent"),
    "graduation_cohort": ("CAMP_ALLD", "Four-year adjusted cohort denominator", "students"),
    "graduation_count": ("CAMP_ALLN_GRAD", "Four-year graduates numerator", "students"),
}
# AA1 is Algebra I, not AA0 (Algebra I AND accelerated testers, added in 2021).
SUBJECTS = {"algebra1": "AA1", "english1": "AR1", "english2": "AR2", "biology": "ABI", "us_history": "AUS"}
FEATURE_GROUPS = {
    "staff": list(STAFF),
    "ap_ib": [f"{kind}_{name}" for kind in ("ap", "ib") for name in EXAMS],
    "staar": [f"staar_{name}_meets" for name in SUBJECTS],
    "graduation": ["graduation_4yr"],
}
FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]


def campus_ids(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.strip('"').str.lstrip("'")
    if not s.str.fullmatch(r"\d{1,9}").all():
        raise ValueError("Invalid campus identifier")
    return s.str.zfill(9)


def read_table(entry: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """Keep raw strings, including blank cells; .xls downloads are SAS HTML."""
    path = RAW / entry["file"]
    if path.suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=entry["sheet"], dtype=str,
                             keep_default_na=False, usecols=columns)
    if path.suffix == ".xls":
        tables = html.parse(str(path)).xpath("//table")
        for table in tables:
            rows = table.xpath(".//tr")
            if not rows:
                continue
            head = [c.text_content().strip() for c in rows[0].xpath("./th|./td")]
            if "CAMPUS" not in head:
                continue
            wanted = head if columns is None else columns
            positions = [head.index(c) for c in wanted]
            records = []
            for row in rows[1:]:
                cells = row.xpath("./th|./td")
                if len(cells) != len(head):
                    raise ValueError(f"Malformed SAS table row: {path}")
                records.append([cells[i].text_content().strip() for i in positions])
            return pd.DataFrame(records, columns=wanted)
        raise ValueError(f"No campus table: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       header=entry.get("header_rows", 1) - 1, usecols=columns)


def numeric(raw: pd.Series, *, ambiguous_blank: bool = False) -> tuple[pd.Series, pd.Series]:
    s = raw.astype(str).str.strip()
    status = pd.Series("observed", index=s.index, dtype=object)
    status.loc[s.isin(["", ".", "•", "n/a", "N/A"])] = "not_reported"
    if ambiguous_blank:
        status.loc[s == ""] = "undefined_or_suppressed"
    masked = s.isin(["-", "*", "**", "***"]) | s.str.match(r"^[<>]")
    status.loc[masked] = "suppressed"
    num = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    for code in (-1, -2, -3):
        status.loc[num == code] = f"suppressed_{abs(code)}"
    bad = (status == "observed") & (num.isna() | (num < 0) | ~np.isfinite(num))
    if bad.any():
        raise ValueError(f"Unrecognized numeric tokens: {s[bad].unique()[:8].tolist()}")
    return num.where(status == "observed"), status


def add_value(out: pd.DataFrame, name: str, raw: pd.Series, *, percent=False, ambiguous_blank=False) -> None:
    value, status = numeric(raw.reset_index(drop=True), ambiguous_blank=ambiguous_blank)
    if percent and not value.dropna().between(0, 100).all():
        raise ValueError(f"{name}: percentage outside [0,100]")
    out[name] = value
    out[name + "_raw"] = raw.reset_index(drop=True)
    out[name + "_status"] = status


def selected_table(source: str, year: int, entry: dict) -> pd.DataFrame:
    columns = None
    if source == "staff_tapr":
        columns = ["CAMPUS"] + [v[0] for v in STAFF.values()]
    elif source in ("ap", "ib"):
        columns = ["Campus", "Group"] + [v[0] for v in EXAMS.values()]
    elif source == "graduation_4yr":
        columns = ["CAMPUS", "CALC_FOR_STATE_ACCT"] + [v[0] for v in GRAD.values()]
    t = read_table(entry, columns)
    if source in ("ap", "ib"):
        t = t.loc[t.Group.str.strip() == "All Students"].copy()
        id_column = "Campus"
    elif source == "graduation_4yr":
        t = t.loc[t.CALC_FOR_STATE_ACCT.str.strip() == "No"].copy()
        id_column = "CAMPUS"
    else:
        id_column = "CAMPUS"
    t["campus_id"] = campus_ids(t[id_column])
    if t.empty or t.campus_id.duplicated().any():
        raise ValueError(f"{source}/{year}: expected nonempty, unique campus rows after selection")
    return t.reset_index(drop=True)


def verify_year(source: str, year: int, entry: dict, t: pd.DataFrame, src: dict) -> str:
    if source in ("ap", "ib", "graduation_4yr"):
        title = str(pd.read_excel(RAW / entry["file"], sheet_name=0, header=None, nrows=1).iloc[0, 0])
        expected = str(year) if source == "graduation_4yr" else f"{year-1}-{year%100:02d}"
        if expected not in title:
            raise ValueError(f"{source}/{year}: workbook title has wrong year: {title}")
        return title
    if source == "staff_tapr":
        # Same proof used for existing demographics: an EXACT, unique statewide
        # campus roster match to independently dated TAPR reports, never filename.
        matches = []
        for other, e in src["demographics"].items():
            reference = read_table(e, ["CAMPUS"])
            if set(campus_ids(reference.CAMPUS)) == set(t.campus_id):
                matches.append(other)
        if matches != [year]:
            raise ValueError(f"Staff {year}: ambiguous/wrong campus roster, matches {matches}; keep staged")
        return f"Exact unique roster match to demographics {year} ({len(t)} campuses)"
    if source == "staar_eoc":
        required = [f"CDA00{code}12{year%100:02d}" for code in SUBJECTS.values()]
        if not all(p + "R" in t or (p + "N" in t and p[:-3] + f"0{year%100:02d}D" in t) for p in required):
            raise ValueError(f"STAAR {year}: missing All Students EOC Meets fields (primary groups required)")
        return f"All five All Students EOC Meets codes embed {year}; see source_definitions"
    raise ValueError(source)


def load_source(source: str, src: dict, log=print) -> pd.DataFrame:
    frames = []
    for year, entry in src[source].items():
        t = selected_table(source, year, entry)
        evidence = verify_year(source, year, entry, t, src)
        log(f"{source}/{year}: {evidence}")
        out = pd.DataFrame({"campus_id": t.campus_id, "year": year})
        if source == "staff_tapr":
            for name, (col, _, _) in STAFF.items():
                add_value(out, name, t[col])
        elif source in ("ap", "ib"):
            for name, (col, _) in EXAMS.items():
                add_value(out, f"{source}_{name}", t[col], percent=True, ambiguous_blank=True)
        elif source == "graduation_4yr":
            for name, (col, _, unit) in GRAD.items():
                add_value(out, name, t[col], percent=unit == "percent")
            if (out.graduation_count > out.graduation_cohort).any():
                raise ValueError(f"Graduates exceed cohort: {year}")
        else:
            for subject, code in SUBJECTS.items():
                name = f"staar_{subject}_meets"
                base = f"CDA00{code}1"
                rate, num, den = (base + f"2{year%100:02d}R", base + f"2{year%100:02d}N", base + f"0{year%100:02d}D")
                for col, suffix in ((num, "_count"), (den, "_denominator")):
                    add_value(out, name + suffix, t[col] if col in t else pd.Series("", index=t.index))
                if rate in t:
                    # Never reconstruct a suppressed/blank published rate from counts.
                    add_value(out, name, t[rate], percent=True)
                    out[name + "_method"] = "published_rate"
                else:
                    n, d = out[name + "_count"], out[name + "_denominator"]
                    if (n > d).any():
                        raise ValueError(f"STAAR {year}/{subject}: numerator exceeds denominator")
                    out[name] = (100 * n / d).where(n.notna() & d.gt(0))
                    out[name + "_raw"] = ""  # derived, never a synthetic raw token
                    out[name + "_status"] = np.select(
                        [out[name + "_count_status"].str.startswith("suppressed") |
                         out[name + "_denominator_status"].str.startswith("suppressed"), d.eq(0), out[name].notna()],
                        ["suppressed_component", "zero_denominator", "observed"], default="not_reported")
                    out[name + "_method"] = "100 * unsuppressed numerator / positive denominator"
        out[f"source_{source}"] = entry["file"]
        frames.append(out)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["campus_id", "year"]).any():
        raise ValueError(f"Nonunique source: {source}")
    return result


def integrate(master: pd.DataFrame, src: dict, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = []
    for source in ("staff_tapr", "ap", "ib", "graduation_4yr", "staar_eoc"):
        if source not in src:
            continue
        addition = load_source(source, src, log)
        overlap = set(addition) & set(master) - {"campus_id", "year"}
        if overlap:
            raise ValueError(f"Attempt to overwrite master fields: {overlap}")
        original_keys = master[["campus_id", "year"]].copy()
        master = master.merge(addition, how="left", on=["campus_id", "year"], validate="one_to_one")
        pd.testing.assert_frame_equal(original_keys, master[["campus_id", "year"]])
        for col in [c for c in addition if c.endswith("_status")]:
            master[col] = master[col].fillna("source_absent")
            name = col.removesuffix("_status")
            for year, group in master.groupby("year"):
                counts = group[col].value_counts().to_dict()
                coverage.append({"source": source, "year": year, "field": name, "rows": len(group),
                                 "source_matched": int(group[f"source_{source}"].notna().sum()),
                                 "observed": int(group[name].notna().sum()),
                                 "coverage_pct": 100 * group[name].notna().mean(),
                                 "minimum": group[name].min(), "maximum": group[name].max(),
                                 "statuses": str(counts)})
    return master, pd.DataFrame(coverage)


def dictionary_rows(master: pd.DataFrame, existing: set[str]) -> list[tuple]:
    rows = []
    for name in master.columns:
        if name in existing:
            continue
        if name in STAFF:
            code, desc, unit = STAFF[name]
        elif name in GRAD:
            code, desc, unit = GRAD[name]
            desc += "; All Students; CALC_FOR_STATE_ACCT=No (without state statutory exclusions)"
        elif name.startswith(("ap_", "ib_")) and name.split("_", 1)[1] in EXAMS:
            kind, metric = name.split("_", 1)
            code, desc = EXAMS[metric]
            desc += f"; All Students; {kind.upper()} criterion >= {3 if kind == 'ap' else 4}"
            unit = "percent"
        else:
            code, desc, unit = "source files; configs/data_sources.yaml", "", ""
            if name.startswith("staar_") and name.endswith("_meets"):
                subject = name.removeprefix("staar_").removesuffix("_meets")
                code = f"CDA00{SUBJECTS[subject]}12{{yy}}R; otherwise N / CDA00{SUBJECTS[subject]}10{{yy}}D * 100"
                desc, unit = "All Students EOC Meets Grade Level; 2020 absent; 2023 redesign", "percent"
        if name.endswith("_raw"):
            desc = "Original source token; blank for a derived value or absent source; consult status and provenance"
        elif name.endswith("_status"):
            desc = "observed, not_reported, source_absent, undefined_or_suppressed, suppressed[_code/component], zero_denominator"
        elif name.startswith("source_"):
            desc, unit = "Unmodified source filename relative to data/raw", "path"
        elif name.endswith("_method"):
            desc = "Published rate or guarded numerator/denominator derivation"
        elif name.endswith(("_count", "_denominator")) and name.startswith("staar_"):
            desc, unit = "Matching unsuppressed STAAR EOC numerator or denominator, where published", "tests"
        rows.append(("master_school_year", name, desc, code, unit))
    return rows
