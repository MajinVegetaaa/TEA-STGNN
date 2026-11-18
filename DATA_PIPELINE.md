# Data pipeline — TEA STGNN master dataset (2018–2024)
> **Updated 2026-10-22: the data now covers 2018–2024** (7 years; 2017 files are on disk for later). Years and files are listed in `configs/data_sources.yaml`.


One script builds everything from the raw TEA files:

```bash
python3 src/data/setup_raw.py --zip Dataset.zip --tapr <TAPR folder> --retention <retention folder>   # once; fills data/raw/
python3 src/data/build_master.py         # ~90 s; reads data/raw/, writes data/processed/
```

Raw files are never modified. They live in numbered source folders under `data/raw/` (layout and year rules in `data/raw/README.md`).
Years used are exactly those listed in `configs/data_sources.yaml` (2018–2024); no other years are created.

## Inputs

| Source | Files | Level | Used for |
|---|---|---|---|
| College admissions (CCAD) | `01_sat_act/<year>/`: `CCAD.dat` (2019), `CCAD.csv` (2020–22), `2024 Campus College Admissions.csv` (2023) | campus × year | SAT, ACT, participation, criterion |
| Attendance / dropout (CDROP_ATT) | `02_attendance_dropout/<year>/`: `CDROP_ATT.dat` (2019), `CDROP_ATT.csv` (2020–22), `2024 Campus Attendance Absenteeism Dropout.csv` (2023) | campus × year | attendance, chronic absenteeism, dropout |
| TEA Directory | `05_school_directory/Directory.csv` (Oct 2025 snapshot) | campus | cohort rules, district, ESC region, charter, NCES ID |
| NCES Public School Locations | `06_school_locations/Public_School_Locations_-_Current.csv` (2024–25) | school | latitude, longitude, locale |
| TAPR campus Profile (student membership) | `03_demographics_tapr/2019.dat`, `2020.dat`, `2021–2023.csv` | campus × year | enrollment, % econ. disadvantaged, EL, special ed, at-risk, race shares, mobility |
| Grade-Level Retention | `04_retention/campus-download-ret-1920 … 2223.xlsx` (2324 not used) | campus × grade × year | retention rate (all grades) |
| TAPR CCMR / TSI criteria (CGRAD1) | `07_tsi/2020.dat`, `2021–2025.csv` (named by report year); class 2018 from `01_sat_act/2018/CAMPPERF.dat` | campus × graduating class | % of graduates meeting TSI criteria: reading, math, both (`tsi_both` is a target) |

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

**TSI criteria, report `Y+1` → year `Y`** (added 2026-10-25, after Phase 3). Like SAT/ACT, a graduating class
is published in the next TAPR report: the 2019-20 report (`07_tsi/2020.dat`) holds class 2019. The column codes
carry the class (`CACRB19R` = class 2019, both subjects, rate), and `audit_raw.py` checks that the configured
class is the newest one in each file. Reports up to 2022-23 also repeat the previous class; every repeated value
equals the configured file (checked). Columns used: `CACRR`, `CACRM`, `CACRB` (reading, math, both; rates). The
reading column is labelled "ELA" from the 2023-24 report on, with the same code. The published rate is used, not
numerator ÷ denominator: the 2023-24 report publishes no TSI denominator, and for 282 campuses its rate implies a
larger denominator than the CCMR graduates count (`CA0GP23D`). Masks: `.` (old files) or blank (new) = not
reported, `-1` = small-group mask (stored as missing).

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
| College-ready % (stands in for TSI; kept after TSI was added on 2026-10-25 so the frozen index doesn't change) | 0.11 |
| Participation | 0.10 |
| Dropout 9–12 | 0.11 |
| Retention | 0.11 |
| Decline | 0.07 |

All components are within-year percentile ranks over the cohort.

**Results:**
- **Coverage:** 7,630 school-years get an index (91–95% per year). 2019 has no retention or decline term.
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
| A graduating class reported in ≥1 year 2018–2024 (SAT/ACT participation published as a value, 0, or a −1/−2 code) | **1,670** |

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

For each of 14 target variables (13 indicators incl. retention and risk index, plus `tsi_both`). **Every feature on row (campus, t) uses only years before t**, so the row's
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
| `master_school_year.csv` | 11,546 rows × 59 columns: IDs, year, flags, 13 indicators, retention (rate, count, denominator), risk index and its coverage, 13 demographic columns, 14 `_supp` flags, 4 source-file columns, then TSI reading / math / both and its source file (appended 2026-10-25) |
| `risk_index_components.csv` | 11,546 rows: the 9 risk components (0–1, higher = riskier) and weight coverage |
| `school_metadata.csv` | 1,670 campuses: names, district, county, ESC region, charter/magnet/virtual flags, grade range, enrollment (Oct 2025), NCES ID, lat/lon, locale, CBSA |
| `temporal_features.csv` | 11,546 rows × 138 features (8 per target × 14 targets incl. retention, risk index and `tsi_both`; lag1 and change for each of 13 demographic columns) |
| `subgroup_school_year.csv` | long table: campus, year, group, indicator, value, suppression, raw column, source file |
| `data_dictionary.csv` | every column with description, raw source code and unit |
| `reports/data_quality_report.csv` | every check and count printed by the script |
| `audit/` (`excluded_not_in_directory.csv`, `name_mismatches.csv`, `geo_rejected_matches.csv`) | audit lists |

## Validation summary

| Item | Value |
|---|---|
| Campuses in raw files (all levels) | 9,540 |
| Final target high schools | 1,670 |
| School-year rows | 11,546 |
| Years | 2018–2024 |
| Schools per year | 2019: 1,623 · 2020: 1,633 · 2021: 1,643 · 2022: 1,652 · 2023: 1,651 |
| Present in all 7 years | 1,616 |
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
   in 2018–2024 but have since closed or been renumbered are excluded. The 2025 grade range and instruction
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
8. **TSI shifts upward in the validation and test years.** The cohort's mean `tsi_both` is 43.6–46.5% for classes
   2018–2022, then 51.3% (2023) and 57.9% (2024); year-to-year correlation per campus falls from 0.79–0.84 to 0.71,
   and persistence error rises from ~10 to 14.4 points.
   The cause is not verified (the reading column is relabelled "ELA" from the 2023-24 report). Level-version
   models will underpredict 2023–2024; the relative / z-score target versions are meant for this.
8. **Renamed campuses.** 43 campus-years have a file name that differs from the Directory name. On inspection
   these are renames or charter takeovers (e.g. Legacy H S → Midland Lee H S), not reused IDs (see
   `audit/name_mismatches.csv`).
9. **Short history.** 7 years: `trend3` and `ma3` exist from 2021, so a model predicting 2024
   from history has at most 4 prior years.
10. **Earlier audit errors** (corrected here): ACT `CAC` is ACT *Science*, not composite (composite is
    `CAA`). The "~1,966 high schools" figure came from the old file, which includes alternative campuses.
# November 2026 additive master-data milestone

The approved integration's master-data gate passed on 2026-11-17: **179/179 raw
audit checks and 115/115 processed-data checks**. The master remains **11,546
campus-years, 1,670 campuses, 2018–2024**; all **59 pre-existing columns** match
the frozen master exactly. It now has 156 columns: 17 new substantive indicators,
12 supporting count/denominator fields, raw tokens, status fields and provenance.
The Risk Index formula and existing temporal features are unchanged.

`src/data/new_sources.py` is a loader module called by the existing
`build_master.py`, not a separate modeling pipeline. Every addition is a validated
campus/year left join. `audit_raw.py` verifies all 34 new source-years, and
`validate_processed.py` calls the independent `validate_additions.py` checks.

| Source | Included fields and selection |
|---|---|
| Staff | `teacher_experience`, `teacher_tenure`, `teacher_salary`, `student_teacher_ratio`, `teacher_fte`; exact unique statewide campus-roster match to independently dated demographics verifies every year |
| AP / IB | Separate `{ap,ib}_participation`, `{ap,ib}_examinees_above_criterion`, `{ap,ib}_exams_above_criterion`; `Group=All Students`, grades 11–12; AP criterion ≥3, IB ≥4 |
| Graduation | `graduation_4yr`, `graduation_count`, `graduation_cohort`; `CALC_FOR_STATE_ACCT=No`, All Students, federal calculation without state statutory exclusions; one row per campus/year |
| STAAR | `staar_{algebra1,english1,english2,biology,us_history}_meets`; All Students EOC Meets Grade Level; matching numerator/denominator columns retained when published |

The supplied 2018 `CAMPSTAAR5.dat` has additional groups only, with no All Students
fields. It is preserved and staged. The correct **`CAMPSTAAR2.dat` was downloaded
from TEA**, verified and registered instead. Its published 2018 rates are used;
no reconstruction was necessary. Download parameters, checksum and dictionaries
are recorded in `data/raw/source_definitions/README.md`. The dictionaries confirm
the same five EOC subject definitions across included years; `AA0` (which includes
accelerated testers) is excluded. 2020 is missing throughout, and the spring 2023
[STAAR redesign](https://tea.texas.gov/assessment-and-accountability/assessment-initiatives/staar-redesign)
remains a comparability caveat.

Every new numeric field retains `_raw` and `_status`. Explicit masks remain
missing. AP/IB blank percentages are `undefined_or_suppressed`, because the
workbooks do not distinguish those reasons. Unmatched campuses are
`source_absent`, never zero. Published rates take precedence even when blank or
suppressed. Derivation is allowed only when the entire rate column is absent,
with matching unsuppressed counts and a positive denominator. Finance stays staged.

Observed campus counts (source absence and masks excluded):

| Year | Teacher experience | AP participation | IB participation | Graduation | STAAR Algebra I |
|---|---:|---:|---:|---:|---:|
| 2018 | 1,588 | 1,130 | 57 | 1,427 | 1,305 |
| 2019 | 1,614 | 1,143 | 62 | 1,447 | 1,588 |
| 2020 | 1,628 | 1,103 | 70 | 1,475 | 0 |
| 2021 | 1,640 | 1,122 | 74 | 1,506 | 1,621 |
| 2022 | 1,646 | 1,126 | 73 | 1,541 | 1,634 |
| 2023 | 1,657 | 1,132 | 72 | 1,561 | 1,645 |
| 2024 | 1,659 | 1,117 | 71 | 1,579 | 1,647 |

Full coverage, all-cell source checks, selected source examples and the gate
receipt are in `data/processed/reports/addition_*.csv` and
`master_data_gate.json`. The gate hashes its inputs and outputs; a failed or stale
receipt cannot authorize expanded feature generation. Eleven regression tests
cover masks, duplicate joins, wrong groups, rate derivation, calendar gaps and
gate invalidation (`python3 -m unittest discover -s tests -v`).

Rebuild sequence (Python environment with `requirements.txt` installed):

```sh
# Run once before the first expanded rebuild; refuses to overwrite a snapshot.
python3 src/data/freeze_outputs.py
python3 src/data/audit_raw.py
python3 src/data/build_master.py
python3 src/data/validate_processed.py
python3 src/data/validate_additions.py  # verifies that the passing gate is current
```

The snapshot is `results/frozen/pre_expansion/`: 100 checksum-verified files
covering processed data, graph/PyG artifacts, feature catalogs, baseline results,
configs, code and documentation. It is local and gitignored. Preserve it when
handing off data; do not rerun the freeze command over it.

## Expanded histories and model inputs (2026-11-18)

The master gate now feeds `src/data/prepare_expanded_features.py`. It preserves
all original temporal values, adds raw calendar histories for audit, and writes
102 separately publication-filtered inputs to `expanded_temporal_features.csv`
(11,546 rows, 378 columns). Calendar gaps remain missing; rolling statistics need
all three observations. The forecast-cutoff assumption, release evidence and
revision limitations are recorded in `configs/feature_availability.md` and YAML.

The assumed cutoff is August 1 before the target school year. Added staff/STAAR
inputs start at lag 2 and AP/IB/graduation at lag 3, subject to source-year release
checks. Original inputs and persistence remain retrospective comparators. Do not
interpret the complete experiment as a deployment-time or archived-vintage test.
Graduation's own histories are filtered too; its unfiltered lag 1 is used only
for the shared evaluation eligibility rule and persistence benchmark.

Expanded artifacts use `data/graph_expanded/` and `results/expanded/`; the original
graph and baseline results remain unchanged. Training-only medians, missing flags
and KNN scaling apply to every expanded comparison, including existing inputs.
Consequently, comparisons within the six new runs isolate input-set differences;
comparisons to frozen results also include the preprocessing change and fixed
level target representation. There is no new Risk Index formula or edge definition.

Reproduction after the master gate:

```sh
python src/data/prepare_expanded_features.py
TEA_EXPANDED=1 python src/graphs/build_graph_data.py
TEA_EXPANDED=1 python src/graphs/validate_graph.py
TEA_EXPANDED=1 python src/graphs/to_pyg.py
TEA_EXPANDED=1 python -m unittest discover -s tests -v
python src/baselines/run_expansion.py
python src/eval/validate_expansion.py
python src/eval/compare_expansion.py
python src/eval/report_expansion.py
```

Use an environment with the project's requirements and PyTorch Geometric.
The local `.venv-expanded` provides PyG for conversion, tests and smoke training.
The experiment runner reuses the existing baseline implementation, model grids
and seeds. It locks models within each input set and the overall winning input
set per target using 2023 only, before predicting 2024. `selection_locked.csv`
and `feature_selection_locked.csv` preserve those choices. All comparisons use
2018–2022 training, 2023 validation, 2024 test and seeds 0–4; target mode is level.
Deterministic LR/KNN and HGB predictions are reused across seeds; HGB reuse is
restricted to fewer than 10,000 training rows (no automatic early stopping or
binning subsample), with a seed-equivalence check before caching. Stochastic
forests/XGBoost still run each seed. No model family or grid is reduced.

See `results/expanded/README.md` for the final validation and accuracy report.

If interrupted, `python src/baselines/run_expansion.py --resume` reuses complete
validation runs from the same unchanged experiment after checking every target,
model, seed, campus/year row and tuning-grid count. Incomplete input-set runs are
recomputed. Do not use resume after changing inputs or modeling settings. The
durable run log is `results/expanded/validation/experiment.log`.
