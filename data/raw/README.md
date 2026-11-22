# data/raw — original TEA downloads (not in git, never edited)

One numbered folder per source; `configs/data_sources.yaml` says which file is used for which year. Set up with
`python3 src/data/setup_raw.py --zip data/archives/Dataset.zip --tapr <folder> --retention <folder> --tsi <folder>`;
`make data` audits and processes it. Sources, year rules and traps: [`docs/DATA.md`](../../docs/DATA.md).

| Folder | Source |
|---|---|
| `01_sat_act/`, `02_attendance_dropout/` | TEA college admissions; attendance and dropout (per year) |
| `03_demographics_tapr/`, `04_retention/`, `07_tsi/` | TAPR student information; grade-level retention; TAPR CCMR/TSI |
| `05_school_directory/`, `06_school_locations/` | TEA AskTED directory (Oct 2025); NCES school locations (2024-25) |
| `08_ap/`, `11_ib/`, `09_staff_tapr/`, `12_graduation/`, `13_staar_eoc/` | AP, IB, staff, 4-year graduation, STAAR end-of-course |
| `10_finance_peims/` | PEIMS actual expenditure, district (`actget`) and charter (`charact`) archives |
| `source_definitions/` | TEA dictionary pages, download evidence, TEA's own campus finance reports + `oracle.csv` |
| `unused/` | state totals and the earlier audit's files (reference only) |

Never join anything in `13_staar_eoc/reference_only/` (all-grades tables, a mislabelled 2018 file identical to 2019).
