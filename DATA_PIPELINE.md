# Data pipeline — TEA STGNN master dataset (2019–2023)

One script builds everything from the raw TEA files:

```bash
unzip -q Dataset.zip -d data/raw/tea     # once; creates data/raw/tea/pro/
python3 src/data/build_master.py         # ~10 s; reads data/raw/tea/pro/, writes data/processed/
```

Raw files are never modified. `Dataset.zip` is extracted to `data/raw/tea/` (the `pro/` folder, unchanged).
Only 2019–2023 data exist; no other years are created.

## Inputs

| Source | Files | Level | Used for |
|---|---|---|---|
| College admissions (CCAD) | `2019/CCAD.dat`, `2020–2022/CCAD.csv`, `2023/2024 Campus College Admissions.csv` | campus × year | SAT, ACT, participation, criterion |
| Attendance / dropout (CDROP_ATT) | `2019/CDROP_ATT.dat`, `2020–2022/CDROP_ATT.csv`, `2023/2024 Campus Attendance Absenteeism Dropout.csv` | campus × year | attendance, chronic absenteeism, dropout |
| TEA Directory | `refernce/Directory.csv` (Oct 2025 snapshot) | campus | cohort rules, district, ESC region, charter, NCES ID |
| NCES Public School Locations | `refernce/Public_School_Locations_-_Current.csv` (2024–25) | school | latitude, longitude, locale |
| TAPR campus Profile (student membership) | `data/raw/tea/TAPR/2019.dat`, `2020.dat`, `2021–2023.csv` | campus × year | enrollment, % econ. disadvantaged, EL, special ed, at-risk, race shares, mobility |
| Grade-Level Retention | `data/raw/tea/retention/campus-download-ret-1920 … 2223.xlsx` (2324 not used) | campus × grade × year | retention rate (all grades) |

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

**Academic Risk Index: not computed.** Last year's Eq. 1 gives only qualitative weights (low, medium, high),
no numbers. Its "location" term is undefined, and it needs a TSI rate that our files don't carry separately.
Inventing weights would not reproduce it, so the exact weights are a question for the professor.

## Step 1 — Target cohort

| Rule (applied to the Directory) | Campuses left |
|---|---|
| All Directory campuses | 9,682 |
| Instruction Type = REGULAR INSTRUCTIONAL (drops Alternative, DAEP, JJAEP) | 8,550 |
| AEA = N | 8,550 (no regular campus is AEA, so this rule changes nothing; kept explicit) |
| Grade range includes grade 12 | 1,857 |
| A graduating class reported in ≥1 year 2019–2023 (SAT/ACT participation published as a value, 0, or a −1/−2 code) | **1,653** |

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
  `data/processed/excluded_not_in_directory.csv`.

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
4. **Variables.** Kept: only All-Students indicators published as a rate or average in all five years (13
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
   Directory on city or on school-name words. Result: 1,653/1,653 reliable.
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
| `master_school_year.csv` | 8,202 rows × 53 columns: IDs, year, flags, 13 indicators, retention (rate, count, denominator), 13 demographic columns, 14 `_supp` flags, 4 source-file columns |
| `school_metadata.csv` | 1,653 campuses: names, district, county, ESC region, charter/magnet/virtual flags, grade range, enrollment (Oct 2025), NCES ID, lat/lon, locale, CBSA |
| `temporal_features.csv` | 8,202 rows × 122 features (8 per target × 12 targets incl. retention; lag1 and change for each of 13 demographic columns) |
| `subgroup_school_year.csv` | long table: campus, year, group, indicator, value, suppression, raw column, source file |
| `data_dictionary.csv` | every column with description, raw source code and unit |
| `data_quality_report.csv` | every check and count printed by the script |
| `excluded_not_in_directory.csv`, `name_mismatches.csv`, `geo_rejected_matches.csv` | audit lists |

## Validation summary

| Item | Value |
|---|---|
| Campuses in raw files (all levels) | 9,540 |
| Final target high schools | 1,653 |
| School-year rows | 8,202 |
| Years | 2019–2023 |
| Schools per year | 2019: 1,623 · 2020: 1,633 · 2021: 1,643 · 2022: 1,652 · 2023: 1,651 |
| Present in all 5 years | 1,621 |
| Duplicate campus-years | 0 |
| Geographic match | 100% (1,653/1,653) |
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
   in 2019–2023 but have since closed or been renumbered are excluded. The 2025 grade range and instruction
   type are applied to all years.
2. **Static metadata is from 2025.** `enrollment_oct2025` is measured after the study period. Do not use it as
   a model feature without acknowledging the look-ahead; a yearly enrollment source (TAPR/PEIMS) would fix
   this.
3. **Demographics** come from TAPR. They are matched for 99.3–100% of school-years; the 11 unmatched rows per
   year are campuses missing from that year's TAPR file. `mobility_rate` is TEA's prior-year mobility.
4. **Retention is missing for 2019** (2018-19 file not downloaded). It covers 2020–2023 for 99–100% of
   school-years. 2021 retention is high (mean 4.4% vs 2.2–3.5%), consistent with pandemic-year retention.
5. **`-2` code meaning is undocumented.** It appears only on SAT/ACT participation (33 cells) and is treated
   as suppressed.
6. **"Above criterion" definition** may have changed across years: the mean falls from 31.6% (2019) to 25.0%
   (2023). Confirm against TEA's TAPR glossary before using it as a target.
7. **COVID.** 2021 has fewer ACT results (1,160 vs ~1,365 in 2019–20). `covid_affected_year` flags 2020–21.
8. **Renamed campuses.** 43 campus-years have a file name that differs from the Directory name. On inspection
   these are renames or charter takeovers (e.g. Legacy H S → Midland Lee H S), not reused IDs (see
   `name_mismatches.csv`).
9. **Short history.** Only 5 years: `trend3` and `ma3` exist only for 2022–2023, so a model predicting 2023
   from history has at most 4 prior years.
10. **Earlier audit errors** (corrected here): ACT `CAC` is ACT *Science*, not composite (composite is
    `CAA`). The "~1,966 high schools" figure came from the old file, which includes alternative campuses.
