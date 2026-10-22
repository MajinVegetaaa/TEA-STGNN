# Raw data

Original downloads, **never edited**. One numbered folder per source, with the original file names.
Nothing here is in git. **Which file is used for which year is listed in `configs/data_sources.yaml`.** Set it up with `python3 src/data/setup_raw.py --zip Dataset.zip --tapr <folder> --retention <folder>`.

| Folder | Source | Years | Used for |
|---|---|---|---|
| `01_sat_act/<year>/` | TEA campus college admissions (CCAD; 2018 = CAMPPERF.dat; 2023/2024 files named "2024/2025 Campus College Admissions.csv") | 2017–2024 (2017 not used yet) | SAT, ACT, participation, college-ready % |
| `02_attendance_dropout/<year>/` | TEA campus attendance & dropout (CDROP_ATT; 2018 = CAMPGRAD.dat; 2023/2024 files named "2024/2025 Campus Attendance…") | 2017–2024 (2017 not used yet) | attendance, chronic absenteeism, dropout |
| `03_demographics_tapr/` | TAPR campus student information, one file per report year (2024: "2024 Campus Student Information.csv") | 2017–2024 (2017 not used yet) | enrollment, % low-income, EL, special ed, at-risk, race, mobility |
| `04_retention/` | TEA Grade-Level Retention, campus files | 2016-17 → 2023-24 (2016-17 not used yet) | retention rate |
| `05_school_directory/` | TEA AskTED directory (Oct 2025 snapshot) | — | which schools are high schools, district, ESC region, NCES ID |
| `06_school_locations/` | NCES Public School Locations (2024–25) | — | latitude, longitude, urban/rural |
| `unused/state_benchmarks/` | statewide totals (SCAD, SDROP_ATT) | 2019–2023 | not used (state, not campus, data) |
| `unused/previous_audit/` | earlier audit notes, scripts and PDFs that came in Dataset.zip | — | reference only |

**Year rule.** A folder or file year `Y` means school year (Y−1)–Y. For example, 2023 = 2022-23, and that is
what `year` means in every processed table. Two naming traps, both verified in `DATA_PIPELINE.md`:
- the 2023 SAT/ACT and attendance files are named "2024 …"
- retention files are named by school year (`ret-2223` = 2022-23 = year 2023)

**Pipeline:** `raw/` → `src/data/build_master.py` → `processed/` → `src/graphs/build_graph_data.py` → `graph/`.
