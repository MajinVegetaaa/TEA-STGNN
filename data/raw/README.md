# Raw data

Original downloads, **never edited**. One numbered folder per source, with the original file names.
Nothing here is in git. **Which file is used for which year is listed in `configs/data_sources.yaml`.** Set it up with `python3 src/data/setup_raw.py --zip Dataset.zip --tapr <folder> --retention <folder> --tsi <folder>`.

| Folder | Source | Years | Used for |
|---|---|---|---|
| `01_sat_act/<year>/` | TEA campus college admissions (CCAD; 2018 = CAMPPERF.dat; 2023/2024 files named "2024/2025 Campus College Admissions.csv") | 2017–2024 (2017 not used yet) | SAT, ACT, participation, college-ready % |
| `02_attendance_dropout/<year>/` | TEA campus attendance & dropout (CDROP_ATT; 2018 = CAMPGRAD.dat; 2023/2024 files named "2024/2025 Campus Attendance…") | 2017–2024 (2017 not used yet) | attendance, chronic absenteeism, dropout |
| `03_demographics_tapr/` | TAPR campus student information, one file per report year (2024: "2024 Campus Student Information.csv") | 2017–2024 (2017 not used yet) | enrollment, % low-income, EL, special ed, at-risk, race, mobility |
| `04_retention/` | TEA Grade-Level Retention, campus files | 2016-17 → 2023-24 (2016-17 not used yet) | retention rate |
| `07_tsi/<report year>` | TAPR "CCMR, College Ready, and TSI Criteria" campus downloads (CGRAD1), named by report year: `2020.dat` = 2019-20 report. Class Y is read from report Y+1 (2018 from `01_sat_act/2018/CAMPPERF.dat`, the same TAPR file) | classes 2018–2024 (reports 2020–2025) | TSI reading, math, both subjects (`tsi_both` is a target) |
| `05_school_directory/` | TEA AskTED directory (Oct 2025 snapshot) | — | which schools are high schools, district, ESC region, NCES ID |
| `06_school_locations/` | NCES Public School Locations (2024–25) | — | latitude, longitude, urban/rural |
| `unused/state_benchmarks/` | statewide totals (SCAD, SDROP_ATT) | 2019–2023 | not used (state, not campus, data) |
| `unused/previous_audit/` | earlier audit notes, scripts and PDFs that came in Dataset.zip | — | reference only |

**Year rule.** A folder or file year `Y` means school year (Y−1)–Y. For example, 2023 = 2022-23, and that is
what `year` means in every processed table. Two naming traps, both verified in `DATA_PIPELINE.md`:
- the 2023 SAT/ACT and attendance files are named "2024 …"
- retention files are named by school year (`ret-2223` = 2022-23 = year 2023)

**Pipeline:** `raw/` → `src/data/build_master.py` → `processed/` → `src/graphs/build_graph_data.py` → `graph/`.

## Staged additions — 2026-11-16

Listed under `not_yet_used` in `configs/data_sources.yaml`. These downloads are not yet model inputs.

| Folder | Contents | Coverage / remaining work |
|---|---|---|
| `08_ap/` | Seven original TEA AP campus workbooks | School years 2017–18 through 2023–24. AP only; IB absent. Select the data sheet and `All Students` group before joining; preserve suppressed values as missing. |
| `09_staff_tapr/` | `CSTAF.csv`, 2024 staff CSV, supplied dictionary PDF | 2024 confirmed in descriptive header; CSTAF presumed 2023 from download conversation, not internally dated. 2018–2022 missing. CSTAF has one header row and apostrophe-prefixed IDs; 2024 has two header rows. |
| `10_finance_peims/` | Seven original actuals ZIPs | file years 2018–2024. Detailed account rows; aggregate before joining. FIN_UNIT needs documented campus mapping; central units must not become school IDs. Charter-specific files were not supplied. Verify fiscal/school-year and release timing before using lagged features. |

Original filenames and file bytes preserved. Audit: `src/data/audit_new_downloads.py`;
results: `results/source_audit/downloads_2026-11-16.json`.

Audit findings: AP matches 1,103–1,149 model campuses per year (about 67–70%);
no duplicate campus/group keys or numeric rates outside 0–100. Absence is not zero.
Staff 2024 matches 1,666/1,666 model campuses; presumed 2023 matches 1,664/1,668.
Both staff files have unique campus IDs. All finance archives pass CRC checks and parse fully.
Crucially, FUNDYEAR is not a single observation year: the 2018 archive contains 2010–2019,
and the 2024 archive contains 2016–2025. Do not assign school years from FUNDYEAR or
filter to the filename year without validating TEA accounting definitions.

### Second batch update (supersedes missing staff/IB notes above)

All seven staff files now exist as `staff_2018.xls` through `staff_2024.csv`.
2018–2020 are SAS HTML tables with an `.xls` extension (read with an HTML table parser);
2021–2023 are coded CSVs; 2024 has descriptive and code header rows.
Years 2018–2023 are indicated by user filenames; coded columns alone do not establish report year.
`11_ib/` contains seven original IB campus workbooks, 2017–18 through 2023–24.
All 28 AP/staff/finance/IB data files are registered under `not_yet_used`.
The audit script now includes IB and all staff formats; rerunning replaces the audit JSON.

### Graduation and STAAR batch — 2026-11-16

- `12_graduation/`: seven four-year graduation workbooks for classes 2018–2024.
  Read the data worksheet, not the overview/dictionary. Select a consistent
  `CALC_FOR_STATE_ACCT` variant before joining campuses; retain the supplied dictionaries.
- `13_staar_eoc/`: 2019 and 2021–2024 staged; 2020 intentionally absent.
  **2018 is NOT available:** `staar_eoc_2018.xls` is byte-identical to `staar_eoc_2019.xls`,
  and its rate columns contain only year 2019. Registered as a rejected download, kept unchanged.
  Both XLS files are HTML tables. The 2024 CSV has two header rows.
- Reproducible checks: `src/data/audit_staar_graduation.py`; detailed output:
  `results/source_audit/staar_graduation_2026-11-16.json`.

Additional check: `13_staar_eoc/CSTAAR_ALL2.xls` has only 2017 rate fields
and all-grades subject codes. It does not fill the 2018 EOC gap.

`CSTAAR_ALL2 (1).xls` is another 2018 All Grades download. Its coded
columns have all-grades subject stems (A001/AC01/AM01/AR01/AS01/AW01),
not the EOC stems (AA11/ABI1/AR11/AR21/AUS1). The 2018 EOC gap remains.


### STAAR folder resolution

The verified 2018 advanced campus EOC file `CAMPSTAAR5.dat` was found and the
STAAR folder is now arranged by data year. See `13_staar_eoc/README.md` for
2018, 2019, 2021–2024 and the separate `reference_only/` files. The earlier
statements that 2018 EOC was missing are superseded. 2020 remains unavailable.

## Integration status — 2026-11-18

Staff, AP, IB, four-year graduation and STAAR now have active registrations in configs/data_sources.yaml and loaders in src/data/new_sources.py, called by the existing master builder. Finance remains staged. Source years, populations and unique keys pass the extended raw audit.

The prior CAMPSTAAR5 verification established the year but missed the required All Students population. It is preserved and excluded. The correct primary-group CAMPSTAAR2 was retrieved from TEA and contains published rates. See source_definitions/README.md for dictionaries, download parameters and checksum. No original download was modified.
