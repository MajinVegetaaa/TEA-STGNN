"""Independent source-to-master checks and a fail-closed master-data gate.

Does not import the loaders or their field maps. Called by validate_processed.
Gate records hashes, so a stale PASS cannot authorize a new graph/feature build.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
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
    paths += list((RAW / "source_definitions").glob("*.html"))
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
    check("finance excluded from master", not any("finance" in c for c in master))
    coverage = pd.read_csv(P / "reports/addition_coverage.csv")
    check("coverage report matches observed master cells",
          all(r.observed == int(master.loc[master.year == r.year, r.field].notna().sum()) for r in coverage.itertuples()))


if __name__ == "__main__":
    require_master_gate()
    print("Master-data gate is current and passed")
