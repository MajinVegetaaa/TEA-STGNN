"""Verified additive sources for the existing campus/year master builder.

No cohort selection, imputation, rolling features, or modeling happens here.
Raw tokens and a status per value distinguish masks, ambiguous blanks and absent
campuses. In particular, AP/IB blanks may mean undefined OR suppressed.
"""
from __future__ import annotations

import re
import zipfile
from collections import Counter
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

# --- PEIMS actual financial data. Definitions are TEA's own, reproduced from its "PEIMS Actual Financial Data by
# Campus" reports (src/data/transcribe_finance_reports.py; checked line by line in validate_additions.py):
#   operating = objects 6100-6499 (TEA: "Objects 6100-6400"); instruction = functions 11 and 95;
#   core operating = the campus-attributable functions TEA marks with a single asterisk (11/95, 12, 13, 21, 23,
#   31, 32, 33, 34). Functions 35-99 are often not attributable to one campus (TEA footnote), so they are excluded.
#   General fund = fund 199 (district files) or funds 199 + 420 (charter files); both verified against TEA's reports.
# Every row of an archive is that fiscal year's spending: FUNDYEAR / FISCALYR is the grant (fund) year, not a filter.
FINANCE_OBJECTS = (6100, 6499)
FINANCE_FUNCTIONS = {
    "finance_instruction": ("11", "95"),
    "finance_core_operating": ("11", "95", "12", "13", "21", "23", "31", "32", "33", "34"),
}
FINANCE_VARIANTS = (("", "all funds"), ("_gf", "general fund"))
FINANCE_GENERAL_FUND = {"district": ("199",), "charter": ("199", "420")}
FINANCE_COLUMNS = {   # archive kind -> its column names (charter files use the nonprofit-accounting names)
    "district": {"fund": "FUND", "function": "FUNCTION", "object": "OBJECT", "year": "FUNDYEAR"},
    "charter": {"fund": "CS_NONPROF_ASSET", "function": "CS_NONPROF_FUNC", "object": "CS_NONPROF_OBJ",
                "year": "FISCALYR"},
}
FINANCE_DOLLARS = [f"{base}{suffix}_expenditure" for base in FINANCE_FUNCTIONS for suffix, _ in FINANCE_VARIANTS]
FINANCE_PER_STUDENT = [f"{base}{suffix}_per_student" for base in FINANCE_FUNCTIONS for suffix, _ in FINANCE_VARIANTS]
FINANCE_FIELDS = FINANCE_DOLLARS + FINANCE_PER_STUDENT
# Deliberately NOT in FEATURE_GROUPS: finance is in the master table but is not a model input until a rerun is approved
# (same precedent as TSI). To wire it in, add "finance": FINANCE_PER_STUDENT there and a prefix in the baseline runners.


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


def finance_archive(path: Path, kind: str, year: int) -> tuple[pd.DataFrame, str]:
    """Sum campus-coded operating expenditure rows of one archive; return (sums, evidence).

    Rows are aggregated BEFORE any status logic because ACTAMT legitimately contains negative amounts
    (adjustments). Only FIN_UNIT != 000 rows are campus-coded (000 is district-wide); the campus id is
    DISTRICT + FIN_UNIT, and is only meaningful where it equals a real campus number.
    """
    c = FINANCE_COLUMNS[kind]
    wanted = ["DISTRICT", "FIN_UNIT", "ACTAMT", "DTUPDATE", c["fund"], c["function"], c["object"], c["year"]]
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt archive: {path.name}")
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected exactly one member, found {names}")
        pieces, years, updates, rows = [], Counter(), set(), 0
        lo, hi = FINANCE_OBJECTS
        with archive.open(names[0]) as stream:
            for chunk in pd.read_csv(stream, dtype=str, usecols=wanted, chunksize=500_000):
                for column, pattern in (("DISTRICT", r"\d{6}"), ("FIN_UNIT", r"\d{3}")):
                    if not chunk[column].str.fullmatch(pattern).all():
                        raise ValueError(f"{path.name}: malformed {column}")
                rows += len(chunk)
                years.update(chunk[c["year"]])
                updates.update(chunk["DTUPDATE"])
                amount = pd.to_numeric(chunk["ACTAMT"], errors="raise")
                code = pd.to_numeric(chunk[c["object"]], errors="raise")
                keep = code.between(lo, hi) & chunk["FIN_UNIT"].ne("000")
                part = pd.DataFrame({
                    "campus_id": (chunk["DISTRICT"] + chunk["FIN_UNIT"])[keep],
                    "function": chunk[c["function"]][keep].str.strip(),
                    "general_fund": chunk[c["fund"]][keep].str.strip().isin(FINANCE_GENERAL_FUND[kind]),
                    "amount": amount[keep]})
                pieces.append(part.groupby(["campus_id", "function", "general_fund"], as_index=False).amount.sum())
    sums = pd.concat(pieces, ignore_index=True).groupby(["campus_id", "function", "general_fund"], as_index=False).amount.sum()
    if len(updates) != 1:
        raise ValueError(f"{path.name}: expected one DTUPDATE, found {sorted(updates)}")
    mode, count = years.most_common(1)[0]
    if mode != str(year) or count / rows < 0.95:
        raise ValueError(f"{path.name}: {c['year']} mode {mode} ({count / rows:.1%}) does not support year {year}")
    published = pd.to_datetime(next(iter(updates)), format="%d%b%Y")
    if published.year != year + 1 or published.month > 4:
        raise ValueError(f"{path.name}: DTUPDATE {published.date()} is not early {year + 1}")
    return sums, (f"{path.name}: {rows:,} rows; {c['year']} {mode} = {count / rows:.1%}; "
                  f"single DTUPDATE {published.date()}; CRC ok")


def load_finance(src: dict, master: pd.DataFrame, log=print) -> pd.DataFrame:
    enrollment = master.set_index(["campus_id", "year"]).enrollment
    frames = []
    for year, entry in src["finance_peims"].items():
        sums, kinds = {}, {}
        for kind, key in (("district", "file"), ("charter", "charter_file")):
            sums[kind], evidence = finance_archive(RAW / entry[key], kind, year)
            log(f"finance/{year}: {evidence}")
            kinds[kind] = set(sums[kind].campus_id)
        overlap = kinds["district"] & kinds["charter"]
        if overlap:
            raise ValueError(f"finance/{year}: {len(overlap)} campus ids in both district and charter files, e.g. {sorted(overlap)[:3]}")
        files = {"district": entry["file"], "charter": entry["charter_file"]}
        out = pd.DataFrame({"campus_id": sorted(kinds["district"] | kinds["charter"]), "year": year})
        out["source_finance_type"] = np.where(out.campus_id.isin(kinds["charter"]), "charter", "district")
        out["source_finance"] = out.source_finance_type.map(files)
        both = pd.concat(sums.values(), ignore_index=True)
        for base, functions in FINANCE_FUNCTIONS.items():
            chosen = both[both["function"].isin(functions)]
            for suffix, _ in FINANCE_VARIANTS:
                subset = chosen if suffix == "" else chosen[chosen.general_fund]
                total = subset.groupby("campus_id").amount.sum()
                name = f"{base}{suffix}_expenditure"
                value = out.campus_id.map(total).fillna(0.0)
                out[name] = value.where(value > 0)
                # Present in the archive but nothing (or a net refund) in this measure: never a silent zero.
                out[name + "_status"] = np.where(value > 0, "observed", "nonpositive_total")
                members = enrollment.xs(year, level="year").reindex(out.campus_id).to_numpy()
                per = out[name] / np.where(members > 0, members, np.nan)
                per_name = f"{base}{suffix}_per_student"
                out[per_name] = per
                out[per_name + "_status"] = np.select(
                    [out[name + "_status"].ne("observed"), np.isnan(members), members <= 0],
                    [out[name + "_status"], "missing_enrollment", "zero_denominator"], default="observed")
        frames.append(out)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["campus_id", "year"]).any():
        raise ValueError("Nonunique source: finance")
    return result


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
    for source in ("staff_tapr", "ap", "ib", "graduation_4yr", "staar_eoc", "finance"):
        if source not in ("finance",) and source not in src:
            continue
        if source == "finance":
            if "finance_peims" not in src:
                continue
            addition = load_finance(src, master, log)
        else:
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
        elif name in FINANCE_FIELDS:
            base = next(b for b in FINANCE_FUNCTIONS if name.startswith(b))
            general = name.startswith(base + "_gf")
            kind = "per enrolled student (master `enrollment`, the TAPR fall membership TEA prints as Total Enrolled Membership)" \
                if name.endswith("_per_student") else "total"
            scope = "instruction (functions 11, 95)" if base == "finance_instruction" else \
                "campus-attributable operating (functions " + ", ".join(FINANCE_FUNCTIONS[base]) + ")"
            fund = "general fund (district archive fund 199; charter archive funds 199 and 420)" if general else "all funds"
            desc = f"Campus-coded {scope} expenditure, objects 6100-6499, {fund}, {kind}; every archive row, FUNDYEAR/FISCALYR not filtered"
            code = "actget<YYYY>.zip (district) or charact<YY>.zip (charter): DISTRICT+FIN_UNIT; ACTAMT"
            unit = "USD per student" if name.endswith("_per_student") else "USD"
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
        elif name == "source_finance_type":
            desc, unit = "Which finance archive holds the campus: district (actget) or charter (charact); never both", "category"
        elif name.startswith("source_"):
            desc, unit = "Unmodified source filename relative to data/raw", "path"
        elif name.endswith("_method"):
            desc = "Published rate or guarded numerator/denominator derivation"
        elif name.endswith(("_count", "_denominator")) and name.startswith("staar_"):
            desc, unit = "Matching unsuppressed STAAR EOC numerator or denominator, where published", "tests"
        rows.append(("master_school_year", name, desc, code, unit))
    return rows
