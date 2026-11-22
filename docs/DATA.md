# Data — sources, cleaning, features and checks

How the raw TEA downloads become the tables every model uses. Decisions and their reasons are dated in
[`DECISIONS.md`](../DECISIONS.md); the plan is in [`ROADMAP.md`](../ROADMAP.md).

```bash
python3 src/data/setup_raw.py --zip data/archives/Dataset.zip --tapr <folder> --retention <folder> --tsi <folder>  # once
make data       # audit_raw.py (200 checks) -> build_master.py -> validate_processed.py (127 checks)
make features   # D1 history features for graph v2 (data/processed/d1_temporal_features.csv)
```

Raw files are never edited and are not in git. Which file is used for which year is listed in one place,
`configs/data_sources.yaml`.

## Years

**2018–2024**, seven school years. `year` Y means school year (Y−1)–Y, e.g. 2024 = 2023-24. Split
(`configs/split.yaml`): train 2018–2022 (2018 only as history: it has no earlier year), validate 2023, test 2024.
2017 was tried and dropped (changed definitions, no accuracy gain); its files stay on disk, unused.

Naming traps (all verified from column codes): SAT/ACT and attendance files for year Y are named "Y+1 …"; retention
files are named by school year (`ret-2223` = year 2023); TSI for class Y is in TAPR report Y+1.

## Sources

| Folder (`data/raw/`) | Source | Gives | Notes |
|---|---|---|---|
| `01_sat_act/<year>/` | TEA campus college admissions (CCAD; 2018 = `CAMPPERF.dat`) | SAT total/ERW/math, ACT, participation, college-ready % | ACT composite is code `CAA` (`CAC` is science) |
| `02_attendance_dropout/<year>/` | TEA attendance & dropout (CDROP_ATT; 2018 = `CAMPGRAD.dat`) | attendance, chronic absenteeism (from 2020), dropout 9–12 / 7–8 | |
| `03_demographics_tapr/` | TAPR campus student information | enrollment, % low-income, EL, special ed, at-risk, race shares, mobility | file Y = year Y (verified by campus-list match) |
| `04_retention/` | TEA grade-level retention | retention rate (all grades) | file (Y−1)(Y) = year Y; extra mask `-3` |
| `07_tsi/` | TAPR CCMR/TSI criteria (CGRAD1) | TSI reading, math, **both** | published rates used (2023-24 report has no denominator) |
| `05_school_directory/` | TEA AskTED directory (Oct 2025) | cohort rules, district, ESC region, charter, NCES ID | static snapshot applied to all years |
| `06_school_locations/` | NCES public school locations (2024-25) | latitude, longitude, locale | 1,670/1,670 matched |
| `09_staff_tapr/` | TAPR staff | teacher experience, tenure, salary, student-teacher ratio, teacher FTE | 2018–2020 are SAS HTML tables named `.xls`; year proven by exact campus-roster match |
| `08_ap/`, `11_ib/` | TEA AP and IB campus workbooks | participation, % examinees and % exams meeting criterion (AP ≥3, IB ≥4) | All Students; about 67–70% of campuses have AP; blank = undefined or suppressed |
| `12_graduation/` | TEA four-year graduation | graduation rate, count, cohort | federal calculation (`CALC_FOR_STATE_ACCT=No`) |
| `13_staar_eoc/<year>/` | TAPR STAAR end-of-course | % Meets Grade Level: Algebra I, English I/II, Biology, US History | **no 2020** (cancelled); 2018 = `CAMPSTAAR2.dat` (`CAMPSTAAR5.dat` has no All Students fields); spring-2023 redesign is a comparability caveat |
| `10_finance_peims/` | PEIMS actual expenditure, district (`actget`) and charter (`charact`) archives | instruction and core operating spending, all funds / general fund, USD and per student | see Finance below |

Evidence of every year and definition: `results/source_audit/`, `data/raw/source_definitions/` (TEA dictionary
pages, the STAAR 2018 download parameters and checksum, TEA's own campus finance reports and their transcription
`oracle.csv`). `data/raw/13_staar_eoc/reference_only/` holds files that must never be joined (all-grades tables, a
mislabelled 2018 file identical to 2019).

**Finance rules** (verified against 68 of TEA's own campus reports, 1,727 cells, 0 differ):
file Y = school year (Y−1)–Y; use **every row** — `FUNDYEAR`/`FISCALYR` is the grant year, not a filter; campus id =
`DISTRICT` + `FIN_UNIT` on campus-coded rows (`000` is district-wide); a campus is in the district or the charter
archive, never both; operating = objects 6100–6499; instruction = functions 11, 95; core operating = 11/95, 12, 13, 21,
23, 31–34; general fund = fund 199 (charters 199 + 420); per student = USD / enrollment. 32 school-years (9 tiny
programs) exceed $40k per student; kept as published, entered into models on a log scale.

## Cohort: 1,670 high schools

From the directory: regular instruction (drops alternative, DAEP, JJAEP) → grade range includes 12 → a graduating
class reported in at least one year 2018–2024. Grade 12 rather than full 9–12 keeps senior highs such as Plano
Senior (11–12). Charters (224) and virtual campuses (8, flagged) are kept. Excluded although they reported graduates:
333 alternative campuses, 79 without grade 12 in 2025, 80 not in the 2025 directory (closed or renumbered).

## Master table (`data/processed/master_school_year.csv`)

One row per school and year: **11,546 rows × 174 columns**. Rules:
- IDs are 9-digit text (always read with `dtype=str`); `district_id` = first 6 digits.
- TEA masks become missing with a flag, never a number: `-1`/`-2`/`-3` → `_supp` codes; added sources keep the raw
  token (`_raw`) and a status (`observed`, `suppressed`, `not_reported`, `source_absent`, …).
- Nothing is imputed; a school missing from a source is `source_absent`, never zero.
- Range checks, SAT total = ERW + Math, and a copy check (values identical to the previous year) all pass.

Other outputs: `school_metadata.csv` (1,670 schools), `risk_index_components.csv`, `subgroup_school_year.csv`,
`data_dictionary.csv` (every column with its TEA code and unit), `reports/` (all checks and the hashed
master-data gate) and `audit/` (excluded and renamed campuses).

**Academic Risk Index** (`src/data/risk_index.py`, 0–100, higher = riskier): a corrected version of last year's
index — dropout and retention count as risk, missing stays missing (left empty below 50% weight coverage), fixed
weights from `configs/risk_index.yaml` (pending professor review) instead of circular tuning, decline measured
against prior years only. Spearman +0.53 with dropout, −0.84 with ACT.

## History features (model inputs)

Every feature on row (school, t) uses **only years before t**: lags 1–3, last change, 2- and 3-year means, 3-year
trend, number of prior observations; a missing year stays missing.

| File | Content | Status |
|---|---|---|
| `temporal_features.csv` | history of the original 14 indicators + last-year demographics (140 columns) | original (Phase 3), frozen |
| `expanded_temporal_features.csv` | + added sources, filtered to what was published by 1 August of t−1 (378 columns) | record of the expanded experiment |
| `d1_temporal_features.csv` | + full history of every added source incl. finance (308 columns) | **current** (decision D1: everything up to t−1 is known) |

Publication dates collected for a possible real-time analysis: `configs/feature_availability.yaml` and
`configs/finance_availability.yaml` (bounds per source and year; finance needs a lag of 2 under a 1-August cutoff).
A leakage test scrambles a year's values and confirms that year's features do not change.

## Checks

| Check | Command | Result |
|---|---|---|
| Raw files are the right dataset and year | `audit_raw.py` | 200/200 |
| Independent rebuild of every added value, TEA finance reports, frozen columns unchanged | `validate_processed.py` | 127/127 |
| Unit and leakage tests | `make test` | 44 (2 skipped) |

`results/frozen/pre_expansion/` (local, not in git) holds checksummed copies of the pre-expansion data, graph and
results; validators compare against it. Never overwrite it.

## Known limitations

1. **Survivorship:** the cohort uses the 2025 directory, so 80 campuses that closed are excluded; directory
   metadata (district, grade range, coordinates) is a 2025 snapshot applied to all years.
2. **COVID:** 2020–2021 are distorted (flag `covid_affected_year`); STAAR 2020 does not exist.
3. **Retrospective data:** downloads are today's revised files (TEA re-posted finance in May 2026); results are a
   research benchmark, not a real-time forecast.
4. **Definition drift:** college-ready % falls 31.6% → 25.0% (2019–2023); TSI jumps in 2023–2024 (column relabelled
   "ELA"); STAAR redesigned in 2023.
5. **Short history:** seven years, so a 2024 forecast has at most four prior training years.
