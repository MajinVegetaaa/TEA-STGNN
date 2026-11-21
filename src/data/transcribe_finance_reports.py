"""Transcribe TEA's own "PEIMS Actual Financial Data by Campus" reports into a fixture.

The reports (PDF for 2019-20 to 2023-24, HTML for 2017-18 and 2018-19) are TEA's published
campus totals. They are the independent oracle for the finance integration: they are kept
unmodified under data/raw/source_definitions/finance_campus_reports/<year>/<campus>.{pdf,html}
and transcribed here, without importing the finance loader, to oracle.csv in the same folder.

Year Y is the school year (Y-1)-Y, the project's `year`. One row per report and funds column
(general fund / all funds); a value is blank where TEA prints none.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from lxml import html

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "data/raw/source_definitions/finance_campus_reports"
NUMBER = re.compile(r"\$?-?[\d,]+(?:\.\d+)?%?")
# label printed by TEA -> column. Function labels carry their codes; see report footers.
ROWS = {
    "Total Expenditures": "total_expenditure",
    "Operating-Payroll": "payroll",
    "Other Operating": "other_operating",
    "Total Operating Expenditures": "total_operating",
    "Instruction (11,95)": "fn_11_95",
    "Instructional Res/Media (12)": "fn_12",
    "Curriculum/Staff Develop (13)": "fn_13",
    "Instructional Leadership (21)": "fn_21",
    "School Leadership (23)": "fn_23",
    "Guidance/Counseling Svcs (31)": "fn_31",
    "Social Work Services (32)": "fn_32",
    "Health Services (33)": "fn_33",
    "Transportation(34)": "fn_34",
    "Food (35)": "fn_35",
    "Extracurricular (36)": "fn_36",
    "Plant Maint/Operation (51)": "fn_51",
    "Security/Monitoring (52)": "fn_52",
    "Data Processing Svcs (53)": "fn_53",
}


def page_text(path: Path) -> str:
    if path.suffix == ".pdf":
        from pypdf import PdfReader  # only needed to regenerate oracle.csv; not in requirements.txt
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    tree = html.parse(str(path))
    return "\n".join(" ".join(row.text_content().split()) for row in tree.xpath("//tr"))


def numbers(text: str) -> list[float]:
    values = []
    for token in NUMBER.findall(text):
        if token.endswith("%"):
            continue
        values.append(float(token.replace("$", "").replace(",", "")))
    return values


def parse(path: Path, year: int) -> list[dict]:
    text = page_text(path)
    if "Total Operating Expenditures" not in text:
        return []  # TEA printed "campus is not valid for the year", not a report
    campus = path.stem
    member = re.search(r"Total (?:Enrolled )?Membership:?\s*([\d,]+)", text)
    if not member:
        raise ValueError(f"{path}: no membership")
    out = {"general_fund": {}, "all_funds": {}}
    seen = set()
    for line in text.splitlines():
        line = line.replace("\xa0", " ").strip()
        for label, column in ROWS.items():
            if not line.startswith(label) or label in seen:
                continue
            tail = line[len(label):].replace("*", " ")
            # Per row TEA prints amount, share, per-student for general fund then all funds.
            # Amounts are the 1st and 4th non-percent numbers when percent signs are printed,
            # otherwise (HTML) the 1st and 4th of the six numbers.
            if "%" in tail:
                nums = [float(t.replace("$", "").replace(",", "")) for t in NUMBER.findall(tail) if not t.endswith("%")]
                # the % tokens were skipped, so [gf$, gf/student, af$, af/student]
                if len(nums) == 4:
                    amounts = (nums[0], nums[2])
                else:
                    continue
            else:
                nums = numbers(tail)
                if len(nums) != 6:
                    continue
                amounts = (nums[0], nums[3])
            out["general_fund"][column], out["all_funds"][column] = amounts
            seen.add(label)
    rows = []
    for funds in ("general_fund", "all_funds"):
        rows.append({"campus_id": campus, "year": year, "funds": funds,
                     "membership": float(member.group(1).replace(",", "")),
                     "source_file": str(path.relative_to(ROOT)), **out[funds]})
    return rows


def main() -> None:
    records = []
    for path in sorted(FOLDER.glob("20*/*")):
        if path.suffix in (".pdf", ".html"):
            records += parse(path, int(path.parent.name))
    table = pd.DataFrame(records)
    table.to_csv(FOLDER / "oracle.csv", index=False)
    print(f"Wrote {FOLDER / 'oracle.csv'}: {len(table)} rows, {table.groupby(['campus_id','year']).ngroups} reports")


if __name__ == "__main__":
    main()
