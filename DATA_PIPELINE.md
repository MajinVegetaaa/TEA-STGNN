# Data pipeline — TEA STGNN master dataset (2017–2024)
> **Updated 2026-10-22: the data now covers 2017–2024** (8 years). Years and files are listed in `configs/data_sources.yaml`. The 2017 college-ready % is blanked (definition change, see below).


One script builds everything from the raw TEA files:

```bash
python3 src/data/setup_raw.py --zip Dataset.zip --tapr <TAPR folder> --retention <retention folder>   # once; fills data/raw/
python3 src/data/build_master.py         # ~90 s; reads data/raw/, writes data/processed/
```

Raw files are never modified. They live in numbered source folders under `data/raw/` (layout and year rules in `data/raw/README.md`).
Years used are exactly those listed in `configs/data_sources.yaml` (2017–2024); no other years are created.

## Inputs

| Source | Files | Level | Used for |
|---|---|---|---|
| College admissions (CCAD) | `01_sat_act/<year>/`: `CCAD.dat` (2019), `CCAD.csv` (2020–22), `2024 Campus College Admissions.csv` (2023) | campus × year | SAT, ACT, participation, criterion |
| Attendance / dropout (CDROP_ATT) | `02_attendance_dropout/<year>/`: `CDROP_ATT.dat` (2019), `CDROP_ATT.csv` (2020–22), `2024 Campus Attendance Absenteeism Dropout.csv` (2023) | campus × year | attendance, chronic absenteeism, dropout |
| TEA Directory | `05_school_directory/Directory.csv` (Oct 2025 snapshot) | campus | cohort rules, district, ESC region, charter, NCES ID |
| NCES Public School Locations | `06_school_locations/Public_School_Locations_-_Current.csv` (2024–25) | school | latitude, longitude, locale |
| TAPR campus Profile (student membership) | `03_demographics_tapr/2019.dat`, `2020.dat`, `2021–2023.csv` | campus × year | enrollment, % econ. disadvantaged, EL, special ed, at-risk, race shares, mobility |
| Grade-Level Retention | `04_retention/campus-download-ret-1920 … 2223.xlsx` (2324 not used) | campus × grade × year | retention rate (all grades) |

Not used: state files (`SCAD`, `SDROP_ATT`; state benchmarks, not campus data), `Directory (1).csv` (same as
`Directory.csv` plus site-address columns), the `.geojson` (same points as the CSV), and the earlier audit's
`data/processed/*.csv` (rebuilt here from raw).

## Year definition

`year` = the folder year = the 2-digit suffix in every column code (`CA0CSA19R` → 2019) = school-year end /
graduating class (2019 = 2018–19). The 2023 files are named "2024 …" because they come from the 2023–24 TAPR
release, but their codes end in `23`. Checked against the old combined file: ACT composite matches the
same year for 78–87% of campuses and any other year for ≤4%. The 2023 participation and criterion columns
have an unresolved `&pfy` macro in their English header, but they still line up best with 2023 (23% vs ≤8% for other
years; agreement is low overall because the old file used SAT-only participation).

### Year alignment for the TAPR and retention files

**TAPR profile, file `Y` → year `Y`.** File `Y` is the TAPR report for school year (Y−1)–Y, and its
membership columns describe that school year. This was verified two ways:
- each TAPR file `Y` has exactly the same campus list as CCAD year Y−1, because both come from TAPR report `Y`
- TEA's 2023 field reference labels the columns "Campus 2023 Student Membership"

The profile column codes carry no year, so the file name is the only year key.

**Retention, file "(Y−1)(Y)" → year `Y`.** The rate is the share of students enrolled in (Y−1)–Y who were
retained in grade the following fall, so it is an outcome of year `Y`. The 2018-19 file (year 2019) was not
downloaded, and 2023-24 (year 2024) falls outside the panel. Rows used: `GRADE = AG` (all grades), columns
`CAMP_ALLR`, `CAMP_ALLN` and `CAMP_ALLD`. The retention files add suppression code `-3` (cross-masked), stored
as `_supp = 3`. Counts are written with thousands separators ("1,040") and are parsed as numbers. One `<0.1`
rate per file is kept as missing.

### Academic Risk Index (corrected version of last year's Eq. 1)

`src/data/risk_index.py` produces `risk_index`, scored 0–100 with higher meaning riskier. It follows last year's
code (`phase2code_ST_LLM_WGHTS_NODROP.py`) where that code is sound, and fixes five problems found in it:

| Last year's code | Problem | Corrected version |
|---|---|---|
| `risk = 1 − percentile` for every component | Higher dropout and retention gave *lower* risk | `percentile` for dropout and retention; `1 − percentile` for scores, participation and college-ready |
| Missing values filled with 0 | A school with no ACT data got "no ACT risk" | Weights re-normalised over the available components; the index is left empty below 50% weight coverage |
| 300 random weight vectors, the best by correlation with dropout and retention | Circular: those are components of the index itself | Fixed weights: the old code's defaults, in `configs/risk_index.yaml` (pending review) |
| 3-year rolling mean including the current year | Decline was measured partly against itself | Drop below the mean of up to 3 **prior** years (scores as within-year percentiles), worst across 5 scores |
| "Location" = percentile of a campus-type code | Categories have no order | Dropped |

**Components and weights:**

| Component | Weight |
|---|---|
| SAT ERW, SAT Math, ACT English, ACT Math | 0.11 each |
| College-ready % (stands in for TSI, which is not in our files) | 0.11 |
| Participation | 0.10 |
| Dropout 9–12 | 0.11 |
| Retention | 0.11 |
| Decline | 0.07 |

All components are within-year percentile ranks over the cohort.

**Results:**
- **Coverage:** 12,088 school-years get an index (88–95% per year; 2017 has no decline term and no college-ready component).
- **Direction:** Spearman +0.53 with dropout, +0.51 with retention, −0.84 with ACT composite.
- **Old vs corrected, on the same data with the old directions and missing = 0:**
  - correlation with dropout: 0.01 under the old formula, 0.53 corrected
  - the 50 highest-dropout schools in 2023 sit at the 53rd risk percentile under the old formula and the
    92nd corrected
- **Checks:** `validate_processed.py` hand-recomputes 40 rows and confirms that scrambling 2023 values leaves
  2019–2022 indices unchanged.
- **Component columns** are in `data/processed/risk_index_components.csv`.

The index summarises year *t*'s outcomes, so it is a target for year *t*; its `_lag1` is a valid input.

## Step 1 — Target cohort

| Rule (applied to the Directory) | Campuses left |
|---|---|
| All Directory campuses | 9,682 |
| Instruction Type = REGULAR INSTRUCTIONAL (drops Alternative, DAEP, JJAEP) | 8,550 |
| AEA = N | 8,550 (no regular campus is AEA, so this rule changes nothing; kept explicit) |
| Grade range includes grade 12 | 1,857 |
| A graduating class reported in ≥1 year 2017–2024 (SAT/ACT participation published as a value, 0, or a −1/−2 code) | **1,670** |

Why grade 12 and not "covers 9–12": split-campus districts run senior highs that graduate all their seniors
(Plano Senior and Plano West 11–12, Allen/Bell/Trinity 10–12, Carroll Senior 11–12). A strict 9–12 rule drops
them. `school_metadata.serves_full_9_12` marks the strict subset (9 campuses are not full 9–12).

Grade ranges are parsed as sets of grades, so non-contiguous values such as `PK 09-12` and `09-10 12` are
handled. Charter schools are public and are kept (`is_charter`, 214 campuses). Virtual/hybrid campuses are
kept and flagged (8).

Campuses that reported graduates but are excluded:
- 333 alternative, DAEP or JJAEP campuses.
- 79 campuses whose 2025 grade range has no grade 12. These are mostly middle schools with a one-off
  graduate record, and Plano/Carroll 9–10 campuses.
- 80 campuses not in the 2025 Directory (closed or renumbered); listed in
  `data/processed/audit/excluded_not_in_directory.csv`.

Comparison with the old combined Excel file (1,966 campuses): 1,621 of its campuses are in our cohort. Of the
345 that aren't, 209 are alternative or DAEP campuses, 42 are regular campuses without grade 12, and 94 are no
longer in the Directory.

## Step 2 — Master school × year table

1. **IDs.** Campus IDs arrive as `"001902001"`, `'001902001` (Excel quote) or `001902001`. Each is stripped of
   quotes and zero-padded to 9 digits and must be all digits. `district_id` comes from the Directory and is
   checked to equal the first 6 digits of `campus_id` (true for all campuses). The NCES ID is 12 digits.
2. **Per year**, CCAD and CDROP_ATT are merged on `campus_id` with `validate="one_to_one"`. Their campus sets
   are identical in every year (0 orphans).
3. **Values.** Raw cells are read as text. Values are mapped as follows:

   | Raw value | Value stored | `_supp` flag |
   |---|---|---|
   | `.` or blank (not reported) | NaN | 0 |
   | `-1` (small-group mask) | NaN | 1 |
   | `-2` | NaN | 2 |
   | anything else | the number | 0 |

   No other non-numeric tokens occur. No value is imputed.
4. **Variables.** Kept: only All-Students indicators published as a rate or average in every year (13
   columns). 2023-only count, denominator and "% graduates above criterion" columns are not in the master.
   Other student groups go in `subgroup_school_year.csv` (long format).
5. **Duplicates.** Every raw file is checked for exact duplicate rows and for conflicting rows that share an
   ID: 0 of both. The master is asserted unique on (`campus_id`, `year`).
6. **Rows.** A cohort campus gets a row for every year it appears in the TEA files. Missing years are not
   filled in.
7. **Directory metadata** is joined `many_to_one`. It is a 2025 snapshot and is treated as time-invariant
   (see limitations).
8. **Geography.** Directory `NCES School ID` ↔ NCES `NCESSCH` (TX only; IDs unique on both sides). A match is
   kept as `reliable` only if the NCES ID is unique in the cohort and the NCES record agrees with the
   Directory on city or on school-name words. Result: 1,670/1,670 reliable.
   - The NCES name agrees for 99.5% of campuses and the city for 94.5%.
   - No match fails both checks.
   - Every coordinate lies inside the Texas bounding box.
9. **Traceability.** Each row has `source_ccad` and `source_cdrop`, and `data_dictionary.csv` gives the raw
   column code for each variable.
10. **Checks.**
    - Range checks (SAT 400–1600, ACT 1–36, rates 0–100): 0 violations.
    - SAT total = ERW + Math in every row.
    - Copy check on all 13 indicators: the share of schools identical to the previous year (ignoring 0 and
      100) is 0–4% for scores, rates and attendance (the old file showed ~99% for SAT), so the raw values are
      genuine. Dropout 9–12 is 7–11%, consistent with small rates like 0.3% repeating.

## Step 3 — Temporal features (`temporal_features.csv`)

For each of 11 target variables. **Every feature on row (campus, t) uses only years before t**, so the row's
own value can be its target:

| Feature | Definition | First year available |
|---|---|---|
| `_lag1`, `_lag2`, `_lag3` | value in t−1, t−2, t−3 | 2020 / 2021 / 2022 |
| `_change_lag1` | lag1 − lag2 (last observed year-over-year change) | 2021 |
| `_ma2`, `_ma3` | mean of the last 2 / 3 prior years | 2021 / 2022 |
| `_trend3` | OLS slope over t−3..t−1 = (lag1 − lag3)/2; requires lag1, lag2 and lag3 | 2022 |
| `_n_prior_obs` | number of non-missing prior years | 2019 (= 0) |

Lags are read from a full campus × year grid, so a gap gives NaN rather than an older year. A feature is NaN
if any input is missing. A self-test asserts `act_composite_lag1(t) == act_composite(t−1)`.

## Outputs (`data/processed/`)

| File | Content |
|---|---|
| `master_school_year.csv` | 13,140 rows × 55 columns: IDs, year, flags, 13 indicators, retention (rate, count, denominator), risk index and its coverage, 13 demographic columns, 14 `_supp` flags, 4 source-file columns |
| `risk_index_components.csv` | 13,140 rows: the 9 risk components (0–1, higher = riskier) and weight coverage |
| `school_metadata.csv` | 1,670 campuses: names, district, county, ESC region, charter/magnet/virtual flags, grade range, enrollment (Oct 2025), NCES ID, lat/lon, locale, CBSA |
| `temporal_features.csv` | 13,140 rows × 130 features (8 per target × 13 targets incl. retention and risk index; lag1 and change for each of 13 demographic columns) |
| `subgroup_school_year.csv` | long table: campus, year, group, indicator, value, suppression, raw column, source file |
| `data_dictionary.csv` | every column with description, raw source code and unit |
| `reports/data_quality_report.csv` | every check and count printed by the script |
| `audit/` (`excluded_not_in_directory.csv`, `name_mismatches.csv`, `geo_rejected_matches.csv`) | audit lists |

## Validation summary

| Item | Value |
|---|---|
| Campuses in raw files (all levels) | 9,540 |
| Final target high schools | 1,670 |
| School-year rows | 13,140 |
| Years | 2017–2024 |
| Schools per year | 2019: 1,623 · 2020: 1,633 · 2021: 1,643 · 2022: 1,652 · 2023: 1,651 |
| Present in all 8 years | 1,591 |
| Duplicate campus-years | 0 |
| Geographic match | 100% (1,670/1,670) |
| Districts / ESC regions | 1,046 / 20 |
| Urbanicity | Rural 708 · City 501 · Suburb 290 · Town 154 |

**Missing data and suppression in the master table:**

| Variable | % missing | −1 suppressed |
|---|---|---|
| SAT total/ERW/math | 21.0 | 846 |
| ACT composite/English/math/science | 23.7 | 1,075 |
| SAT/ACT participation | 4.5 | 79 (+33 coded −2) |
| Above criterion | 6.7 | 226 |
| Attendance | 0.8 | 17 |
| Chronic absenteeism | 20.3 (all of 2019 is unpublished) | 5 |
| Dropout 9–12 | 1.1 | 13 |
| Dropout 7–8 | 71.9 (applies only to campuses with grades 7–8) | 64 |

Overall, 20.2% of indicator cells are missing, or 14.5% of the cells that can exist (excluding 2019 chronic
absenteeism, which TEA did not publish, and dropout 7–8).

## Known limitations and unresolved issues

1. **Survivorship.** The cohort is defined using the 2025 Directory, so 80 campuses that reported graduates
   in 2017–2024 but have since closed or been renumbered are excluded. The 2025 grade range and instruction
   type are applied to all years.
2. **Static metadata is from 2025.** `enrollment_oct2025` is measured after the study period. Do not use it as
   a model feature without acknowledging the look-ahead; a yearly enrollment source (TAPR/PEIMS) would fix
   this.
3. **Demographics** come from TAPR. They are matched for 99.3–100% of school-years; the 11 unmatched rows per
   year are campuses missing from that year's TAPR file. `mobility_rate` is TEA's prior-year mobility.
4. **Retention** covers 98–100% of school-years in every year 2017–2024 (the 2016-17 file is TEA's
   `campus-download-1617.xlsx`; the page that links it points to a retired path). 2021 retention is high (mean 4.4% vs 2.2–3.5%), consistent with pandemic-year retention.
5. **`-2` code meaning is undocumented.** It appears only on SAT/ACT participation (33 cells) and is treated
   as suppressed.
6. **"Above criterion" definition changed between 2017 and 2018:** statewide mean 19.3% → 34.8%; the median school
   moves +15.2 points, against −1.5 the year after. The 2017 values are blanked and flagged
   (`sat_act_above_criterion_supp` = 4, `known_breaks` in `configs/data_sources.yaml`), so no 2018 lag uses them. The
   mean also falls from 31.6% (2019) to 25.0% (2023); confirm against TEA's TAPR glossary before using it as a target.
   **SAT 2017:** total ≠ ERW + Math by 2–12 points in 7% of 2017 rows (0.2% in 2018, none after); kept, with a
   looser validator check for 2017.
7. **COVID.** 2021 has fewer ACT results (1,160 vs ~1,365 in 2019–20). `covid_affected_year` flags 2020–21.
8. **Renamed campuses.** 43 campus-years have a file name that differs from the Directory name. On inspection
   these are renames or charter takeovers (e.g. Legacy H S → Midland Lee H S), not reused IDs (see
   `audit/name_mismatches.csv`).
9. **Short history.** 8 years: `trend3` and `ma3` exist from 2020, so a model predicting 2024
   from history has at most 7 prior years.
10. **Earlier audit errors** (corrected here): ACT `CAC` is ACT *Science*, not composite (composite is
    `CAA`). The "~1,966 high schools" figure came from the old file, which includes alternative campuses.
