"""Independent source-to-master checks and a fail-closed master-data gate.

Does not import the loaders or their field maps. Called by validate_processed.
Gate records hashes, so a stale PASS cannot authorize a new graph/feature build.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lxml import html
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"
RAW = ROOT / "data/raw"
FROZEN = ROOT / "results/frozen/pre_expansion"
GATE = P / "reports/master_data_gate.json"
SOURCES = ("staff_tapr", "ap", "ib", "graduation_4yr", "staar_eoc")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_files() -> list[Path]:
    src = yaml.safe_load((ROOT / "configs/data_sources.yaml").read_text())
    paths = [P / "master_school_year.csv", P / "temporal_features.csv", ROOT / "configs/data_sources.yaml",
             ROOT / "configs/risk_index.yaml", ROOT / "configs/split.yaml",
             P / "reports/raw_audit.csv", P / "reports/processed_validation.csv",
             P / "reports/addition_coverage.csv", P / "reports/addition_sources.csv"]
    paths += [ROOT / "src/data" / f for f in ("build_master.py", "new_sources.py", "validate_additions.py", "audit_raw.py")]
    paths += [RAW / e["file"] for source in (*SOURCES, "sat_act", "attendance_dropout", "demographics", "retention", "tsi")
              for e in src.get(source, {}).values()]
    paths += [RAW / src[name] for name in ("directory", "locations")]
    for entry in src.get("finance_peims", {}).values():
        paths += [RAW / entry["file"], RAW / entry["charter_file"]]
    paths.append(RAW / "source_definitions/finance_campus_reports/oracle.csv")
    paths += list((RAW / "source_definitions").glob("*.html"))
    # The pre-expansion files are the independent reference for the unchanged-column checks above.
    # Include them (and the manifest) in the gate fingerprint so a previously saved PASS cannot
    # remain valid after the reference archive is removed or changed.
    frozen_required = [
        FROZEN / "manifest.json",
        FROZEN / "data/processed/master_school_year.csv",
        FROZEN / "data/processed/temporal_features.csv",
        FROZEN / "data/processed/school_metadata.csv",
        FROZEN / "data/processed/risk_index_components.csv",
        FROZEN / "data/processed/subgroup_school_year.csv",
    ]
    paths.extend(frozen_required)
    frozen_manifest = FROZEN / "manifest.json"
    if frozen_manifest.exists():
        manifest = json.loads(frozen_manifest.read_text())
        paths.extend(FROZEN / rel for rel in manifest["files"])
    return sorted(set(paths))


def save_gate(results: list[dict]) -> None:
    passed = all(r["passed"] for r in results)
    payload = {"passed": passed, "checks": len(results), "failures": [r for r in results if not r["passed"]],
               "files": {str(p.relative_to(ROOT)): sha(p) for p in gate_files() if p.exists()}}
    GATE.write_text(json.dumps(payload, indent=2) + "\n")


def require_master_gate() -> None:
    if not GATE.exists():
        raise RuntimeError("Run audit_raw.py, build_master.py, and validate_processed.py before expanding inputs")
    gate = json.loads(GATE.read_text())
    if not gate["passed"]:
        raise RuntimeError("Master-data gate failed; see data/processed/reports/master_data_gate.json")
    for path in gate_files():
        if not path.exists() or gate["files"].get(str(path.relative_to(ROOT))) != sha(path):
            raise RuntimeError(f"Stale master-data gate: {path}; rerun validation")


def raw_records(entry: dict, requested: list[str]):
    """Read selected cells directly using independent workbook/CSV/HTML paths."""
    path = RAW / entry["file"]
    if path.suffix == ".xlsx":
        book = load_workbook(path, read_only=True, data_only=True)
        rows = book[entry["sheet"]].iter_rows(values_only=True)
        head = [str(v).strip() for v in next(rows)]
        indices = [head.index(c) for c in requested]
        for row in rows:
            yield {name: "" if row[i] is None else str(row[i]).strip() for name, i in zip(requested, indices)}
        book.close()
    elif path.suffix == ".xls":
        for table in html.parse(str(path)).xpath("//table"):
            rows = table.xpath(".//tr")
            head = [c.text_content().strip() for c in rows[0].xpath("./th|./td")] if rows else []
            if "CAMPUS" not in head:
                continue
            indices = [head.index(c) for c in requested]
            for row in rows[1:]:
                cells = row.xpath("./th|./td")
                yield {name: cells[i].text_content().strip() for name, i in zip(requested, indices)}
            return
    else:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = csv.reader(stream)
            for _ in range(entry.get("header_rows", 1) - 1):
                next(rows)
            head = next(rows)
            for row in rows:
                if row:
                    yield {name: row[head.index(name)] if name in head else "" for name in requested}


def expected(token: str, ambiguous=False) -> tuple[float, str]:
    s = token.strip()
    if s in ("", ".", "•", "n/a", "N/A"):
        return np.nan, "undefined_or_suppressed" if ambiguous and s == "" else "not_reported"
    if s in ("-", "*", "**", "***") or s.startswith(("<", ">")):
        return np.nan, "suppressed"
    value = float(s.replace(",", ""))
    if value in (-1, -2, -3):
        return np.nan, f"suppressed_{abs(int(value))}"
    if value < 0 or not np.isfinite(value):
        raise ValueError(f"Invalid source value: {s}")
    return value, "observed"


def validate(master: pd.DataFrame, check) -> None:
    src = yaml.safe_load((ROOT / "configs/data_sources.yaml").read_text())
    frozen_path = FROZEN / "data/processed/master_school_year.csv"
    check("pre-expansion snapshot exists", frozen_path.exists())
    if not frozen_path.exists():
        return
    old = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
    actual = pd.read_csv(P / "master_school_year.csv", dtype=str, keep_default_na=False)
    check("every pre-existing master column and row unchanged",
          set(old).issubset(actual) and old.equals(actual[old.columns]), f"{len(old)} rows, {len(old.columns)} old columns")
    old_temporal = pd.read_csv(FROZEN / "data/processed/temporal_features.csv", dtype=str, keep_default_na=False)
    temporal = pd.read_csv(P / "temporal_features.csv", dtype=str, keep_default_na=False)
    check("every pre-existing temporal feature unchanged",
          set(old_temporal).issubset(temporal) and old_temporal.equals(temporal[old_temporal.columns]))
    manifest = json.loads((FROZEN / "manifest.json").read_text())["files"]
    check("frozen snapshot checksums valid", all(sha(FROZEN / rel) == item["sha256"] for rel, item in manifest.items()))
    for rel in ("data/processed/school_metadata.csv", "data/processed/risk_index_components.csv",
                "data/processed/subgroup_school_year.csv"):
        check(f"unchanged: {rel}", sha(ROOT / rel) == manifest[rel]["sha256"])
    audit = pd.read_csv(P / "reports/raw_audit.csv")
    check("raw audit passes with all 34 additive source-years checked",
          not audit.result.eq("FAIL").any() and
          sum(audit.source.isin(SOURCES) & audit.check.str.endswith("source year, population and unique campus keys")) == 34)
    maps = {
        "staff_tapr": {"teacher_experience": "CPSTEXPA", "teacher_tenure": "CPSTTENA", "teacher_salary": "CPSTTOSA",
                       "student_teacher_ratio": "CPSTKIDR", "teacher_fte": "CPSTTOFC"},
        "graduation_4yr": {"graduation_4yr": "CAMP_ALLR_GRAD", "graduation_count": "CAMP_ALLN_GRAD", "graduation_cohort": "CAMP_ALLD"},
    }
    for source in ("ap", "ib"):
        maps[source] = {source + "_participation": "Part_Rate", source + "_examinees_above_criterion": "Exnees_Above_Crit_Rate",
                        source + "_exams_above_criterion": "Exams_Above_Crit_Rate"}
    source_cells = []
    for source in SOURCES:
        for year, entry in src[source].items():
            my = master[master.year == year].set_index("campus_id")
            sample_ids = set(sorted(my.index)[:5])
            amap = maps.get(source)
            if source == "staar_eoc":
                amap = {f"staar_{name}_meets": f"CDA00{code}12{year%100:02d}R" for name, code in
                        [("algebra1", "AA1"), ("english1", "AR1"), ("english2", "AR2"), ("biology", "ABI"), ("us_history", "AUS")]}
            idcol = "Campus" if source in ("ap", "ib") else "CAMPUS"
            filters = ["Group"] if source in ("ap", "ib") else (["CALC_FOR_STATE_ACCT"] if source == "graduation_4yr" else [])
            requested = [idcol] + filters + list(amap.values())
            # Independently inspect STAAR counts too; absence is preserved as missing.
            if source == "staar_eoc":
                for name, col in list(amap.items()):
                    amap[name + "_count"] = col[:-1] + "N"
                    amap[name + "_denominator"] = col[:-4] + "0" + col[-3:-1] + "D"
                requested = [idcol] + list(amap.values())
                if entry["file"].endswith(".xls"):
                    # 2019 rates-only HTML has no count columns.
                    requested = [idcol] + [c for c in amap.values() if c.endswith("R")]
            seen, errors, cells = set(), 0, 0
            for record in raw_records(entry, requested):
                if filters and record[filters[0]] != ("All Students" if filters[0] == "Group" else "No"):
                    continue
                campus = record[idcol].strip().lstrip("'").zfill(9)
                if campus in seen:
                    errors += 1
                seen.add(campus)
                if campus not in my.index:
                    continue
                for name, code in amap.items():
                    token = record.get(code, "")
                    exp, status = expected(token, source in ("ap", "ib"))
                    got = my.at[campus, name]
                    equal = (np.isnan(exp) and pd.isna(got)) or np.isclose(exp, got, rtol=0, atol=1e-10)
                    errors += not equal or my.at[campus, name + "_status"] != status
                    cells += 1
                    # Small reproducible evidence receipt, including masks and blanks.
                    if campus in sample_ids:
                        source_cells.append({"source": source, "year": year, "campus_id": campus,
                                             "field": name, "raw_column": code, "raw": token, "value": got, "status": status,
                                             "file": entry["file"]})
            absent = my.loc[~my.index.isin(seen)]
            for name in amap:
                errors += int(absent[name].notna().sum())
                errors += int(absent[name + "_status"].ne("source_absent").sum())
                values = my[name].dropna()
                limit = 100 if source in ("ap", "ib") or name == "graduation_4yr" or name.endswith("_meets") else np.inf
                errors += int((~values.between(0, limit)).sum())
            check(f"{source}/{year}: independent values, masks, absence and ranges", errors == 0,
                  f"{cells} source cells; {len(absent)} unmatched campuses; {errors} mismatches")
    pd.DataFrame(source_cells).to_csv(P / "reports/addition_source_cells.csv", index=False)
    staar = [c for c in master if c.startswith("staar_") and c.endswith("_meets")]
    check("all five STAAR subjects missing in 2020", len(staar) == 5 and master.loc[master.year == 2020, staar].isna().all().all())
    check("graduation numerator does not exceed cohort", not (master.graduation_count > master.graduation_cohort).any())
    validate_finance(master, src, check)
    coverage = pd.read_csv(P / "reports/addition_coverage.csv")
    check("coverage report matches observed master cells",
          all(r.observed == int(master.loc[master.year == r.year, r.field].notna().sum()) for r in coverage.itertuples()))


# ---- PEIMS finance. Written without the loader: plain csv streaming and its own copy of the field maps. ----
FIN_COLUMNS = {"district": ("DISTRICT", "FIN_UNIT", "FUND", "FUNCTION", "OBJECT", "ACTAMT"),
               "charter": ("DISTRICT", "FIN_UNIT", "CS_NONPROF_ASSET", "CS_NONPROF_FUNC", "CS_NONPROF_OBJ", "ACTAMT")}
FIN_GENERAL = {"district": {"199"}, "charter": {"199", "420"}}
FIN_MEASURES = {"finance_instruction": {"11", "95"},
                "finance_core_operating": {"11", "95", "12", "13", "21", "23", "31", "32", "33", "34"}}
ORACLE_FUNCTIONS = {"fn_11_95": {"11", "95"}, "fn_12": {"12"}, "fn_13": {"13"}, "fn_21": {"21"}, "fn_23": {"23"},
                    "fn_31": {"31"}, "fn_32": {"32"}, "fn_33": {"33"}, "fn_34": {"34"}, "fn_35": {"35"},
                    "fn_36": {"36"}, "fn_51": {"51"}, "fn_52": {"52"}, "fn_53": {"53"}}


def stream_finance(path: Path, kind: str, wanted: set[str]):
    """{(campus, function, general_fund): amount} and the set of campuses with any campus-coded operating row."""
    district, unit, fund, function, obj, amount = FIN_COLUMNS[kind]
    general = FIN_GENERAL[kind]
    sums, present = {}, set()
    with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        head = next(reader)
        i = [head.index(n) for n in (district, unit, fund, function, obj, amount)]
        for row in reader:
            u = row[i[1]]
            if u == "000" or not 6100 <= int(row[i[4]]) <= 6499:
                continue
            campus = row[i[0]] + u
            if campus not in wanted:
                continue
            present.add(campus)
            key = (campus, row[i[3]].strip(), row[i[2]].strip() in general)
            sums[key] = sums.get(key, 0.0) + float(row[i[5]])
    return sums, present


def validate_finance(master: pd.DataFrame, src: dict, check) -> None:
    fields = [f"{b}{s}_{k}" for b in FIN_MEASURES for s in ("", "_gf") for k in ("expenditure", "per_student")]
    check("finance columns present (8 values, 8 statuses, source type, source file)",
          all(f in master and f + "_status" in master for f in fields) and "source_finance_type" in master and "source_finance" in master)
    wanted = set(master.campus_id.unique())
    check("no model campus uses a unit code of 700 or above (a central-unit code could be mistaken for a campus)",
          not any(int(c[6:]) >= 700 for c in wanted), f"{sum(int(c[6:]) >= 700 for c in wanted)} such campuses")
    oracle = pd.read_csv(RAW / "source_definitions/finance_campus_reports/oracle.csv", dtype={"campus_id": str})
    enrollment = master.set_index(["campus_id", "year"]).enrollment
    oracle_cells = oracle_bad = missing_enrollment = 0
    for year, entry in src["finance_peims"].items():
        sums, present, kinds = {}, set(), {}
        for kind, key in (("district", "file"), ("charter", "charter_file")):
            s, p = stream_finance(RAW / entry[key], kind, wanted)
            sums.update(s); present |= p; kinds[kind] = p
        my = master[master.year == year].set_index("campus_id")
        errors = []
        if kinds["district"] & kinds["charter"]:
            errors.append("campus in both archives")
        for campus, row in my.iterrows():
            here = campus in present
            expected_type = "charter" if campus in kinds["charter"] else ("district" if here else None)
            if here and row.source_finance_type != expected_type:
                errors.append(f"{campus} type")
            if not here and (pd.notna(row.source_finance_type) or pd.notna(row.source_finance)):
                errors.append(f"{campus} absent but sourced")
            for base, functions in FIN_MEASURES.items():
                for suffix in ("", "_gf"):
                    total = sum(sums.get((campus, f, g), 0.0) for f in functions for g in ((True,) if suffix else (True, False)))
                    name = f"{base}{suffix}_expenditure"
                    want_status = "source_absent" if not here else ("observed" if total > 0 else "nonpositive_total")
                    got, status = row[name], row[name + "_status"]
                    if status != want_status or (want_status == "observed" and not np.isclose(got, total, rtol=0, atol=0.01)) or \
                            (want_status != "observed" and pd.notna(got)):
                        errors.append(f"{campus} {name}")
                    members = enrollment.get((campus, year), np.nan)
                    per, per_status = row[name.replace("_expenditure", "_per_student")], row[name.replace("_expenditure", "_per_student_status")]
                    if want_status != "observed":
                        want_per = want_status
                    elif pd.isna(members):
                        want_per = "missing_enrollment"
                    elif members <= 0:
                        want_per = "zero_denominator"
                    else:
                        want_per = "observed"
                    ok = per_status == want_per and ((want_per == "observed" and np.isclose(per, total / members, rtol=1e-12)) or
                                                     (want_per != "observed" and pd.isna(per)))
                    if not ok:
                        errors.append(f"{campus} {name} per student")
        check(f"finance/{year}: every model campus-year recomputed from the raw archives (dollars, statuses, per student)",
              not errors, f"{len(my)} campuses, {len(present)} present, {len(errors)} mismatches {errors[:2]}")
        # TEA's own campus reports: each printed function line and the enrollment denominator.
        for r in oracle[oracle.year == year].itertuples():
            gf = r.funds == "general_fund"
            for column, functions in ORACLE_FUNCTIONS.items():
                value = getattr(r, column)
                if pd.isna(value):
                    continue
                if gf and column == "fn_35":
                    # Food service is in neither measure. TEA's general-fund column for it also counts fund 101 for two
                    # small districts (e.g. 077901001 FY2018: 225,535 + 5,375 = 230,910); every measured function ties.
                    continue
                mine = sum(sums.get((r.campus_id, f, g), 0.0) for f in functions for g in ((True,) if gf else (True, False)))
                oracle_cells += 1
                oracle_bad += abs(mine - value) > 1
            if r.funds == "all_funds":
                known = enrollment.get((r.campus_id, year), np.nan)
                if pd.isna(known):
                    missing_enrollment += 1       # TAPR file lacks the campus-year; finance stays missing_enrollment
                else:
                    oracle_cells += 1
                    oracle_bad += known != r.membership
    check("finance matches TEA's own campus financial reports line by line (functions, general fund and all funds, enrollment)",
          oracle_cells > 800 and oracle_bad == 0, f"{oracle_cells} report cells for {oracle.groupby(['campus_id','year']).ngroups} campus-years; {oracle_bad} differ; "
          f"{missing_enrollment} report(s) where the master has no TAPR enrollment")
    coverage = master.groupby("year").finance_core_operating_per_student.apply(lambda s: s.notna().mean())
    check("finance covers at least 94% of model campuses every year", bool((coverage >= 0.94).all()),
          ", ".join(f"{y}: {v:.1%}" for y, v in coverage.items()))
    values = master[[f for f in fields if f.endswith("_per_student")]].stack()
    check("per-student values are positive and finite", bool(((values > 0) & np.isfinite(values)).all()))
    # All funds can fall below the general fund only when the other funds net NEGATIVE (refunds/reversals of grant
    # spending, which TEA's files carry as negative ACTAMT). The independent recomputation above already proves each value.
    observed = master.finance_core_operating_expenditure.notna() & master.finance_core_operating_gf_expenditure.notna()
    negative_other = int((master.finance_core_operating_gf_expenditure > master.finance_core_operating_expenditure + 0.01)[observed].sum())
    check("net non-general-fund spending is negative in under 0.25% of campus-years (refund reversals)",
          negative_other < 0.0025 * observed.sum(), f"{negative_other} of {int(observed.sum())} campus-years")


if __name__ == "__main__":
    require_master_gate()
    print("Master-data gate is current and passed")
