# Data lead handoff — TEA STGNN (2017–2024)
> **Updated 2026-10-22: the data now covers 2017–2024** (8 years). Years and files are listed in `configs/data_sources.yaml`. The 2017 college-ready % is blanked (definition change, see below).


This is the hand-off from the data stage (Person A) to the baselines/evaluation (B) and graph/model (C)
stages. Details are in `DATA_PIPELINE.md` (processed data) and `GRAPH_DATA_DOCUMENTATION.md` (graph files).
The data decisions are logged in `DECISIONS.md`.

## 1. Data sources and years
Inputs: `Dataset.zip` plus the TAPR profile and Grade-Level Retention downloads, placed unchanged into numbered source folders under `data/raw/` by `src/data/setup_raw.py` (see `data/raw/README.md`).

| Source | Years | Content |
|---|---|---|
| TEA campus college admissions (CCAD / "Campus College Admissions"; 2018 from CAMPPERF) | 2017–2024 | SAT, ACT, participation, criterion |
| TEA campus attendance & dropout (CDROP_ATT; 2018 from CAMPGRAD) | 2017–2024 | attendance, chronic absenteeism (2020+), dropout |
| TEA Directory (`Directory.csv`) | Oct 2025 snapshot | district, ESC region, charter, AEA, grades, NCES ID |
| NCES Public School Locations | 2024–25 | latitude/longitude, locale |
| TAPR campus student information | 2017–2024 | enrollment, % econ. disadvantaged, EL, special ed, at-risk, race shares, mobility |
| Grade-Level Retention | 2020–2023 (2019 not downloaded) | all-grades retention rate |

**Years covered: 2017–2024** (from `configs/data_sources.yaml`). `year` is the school-year end or graduating class (2024 = 2023–24). The
"2024 …" files in the `2023/` folder hold 2023 data; this was verified from the column codes and against the
old Excel file.

## 2. High-school selection rule
These rules are applied to the Directory:
- Instruction type REGULAR
- not AEA
- the grade range includes grade 12
- TEA reported a graduating class (SAT/ACT participation) in at least one year of 2017–2024

That gives **1,670 schools**. 1,591 of them are present in all 8 years, and 1,644 serve the full 9–12 span
(`serves_full_9_12 = 1`). The other 9 are senior highs such as Plano Senior (11–12) and Allen (10–12).
Excluded campuses that reported graduates:
- 333 alternative, DAEP or JJAEP campuses
- 79 campuses with no grade 12 in 2025
- 80 campuses missing from the 2025 Directory

## 3. Master dataset
`data/processed/master_school_year.csv` has **13,140 rows** (one per campus-year) and **55 columns**.
Per year: 1,623 · 1,633 · 1,643 · 1,652 · 1,651.

All targets below are All-Students values for year t. Non-missing counts by year (2019 → 2023):

| Target | Non-missing |
|---|---|
| `sat_total`, `sat_erw`, `sat_math` | 1,298 / 1,334 / 1,248 / 1,290 / 1,314 |
| `act_composite`, `act_english`, `act_math` | 1,366 / 1,363 / 1,160 / 1,189 / 1,180 |
| `sat_act_participation` | 1,516 → 1,618 |
| `sat_act_above_criterion` | 1,494 → 1,576 |
| `dropout_9_12` | 1,589 → 1,645 |
| `attendance_rate` | 1,609 → 1,646 |
| `chronic_absenteeism` | 2020–2023 only, ~1,620–1,648 |
| `retention_rate` | 2020–2023 only: 1,623 / 1,633 / 1,640 / 1,649 |
| `risk_index` (0–100, corrected Eq. 1) | 1,479 / 1,519 / 1,513 / 1,544 / 1,575 |

Student composition (enrollment, % econ. disadvantaged, EL, special ed, at-risk, race shares, mobility) is available for 99.3–100% of school-years. It is a fall snapshot of year t, so use its `_lag1` as a model input.

The Academic Risk Index is a corrected version of last year's code. It fixes the dropout/retention direction,
missing-as-zero, circular weight tuning and current-year decline, and drops "location". Weights are pending
review in `configs/risk_index.yaml`; details are in `DATA_PIPELINE.md`. Not available: STAAR and graduation rate.

## 4. Cleaning and suppression decisions
- **IDs:** campus IDs are 9-digit text with leading zeros kept (raw formats `1902001`, `'001902001` and
  `001902001` are all normalised).
- **Value mapping:**

  | Raw value | Stored as | `<var>_supp` |
  |---|---|---|
  | `.` or blank (not reported) | NaN | 0 |
  | `-1` (small-group mask) | NaN | 1 |
  | `-2` (undocumented; only on participation) | NaN | 2 |

  Nothing is imputed.
- **Duplicates:** 0 found in any raw file, and the master is unique on (campus_id, year).
- **Merges:** CCAD↔CDROP_ATT one-to-one (identical campus sets every year), Directory many-to-one, and NCES
  matched by NCES ID and checked on name or city (100% reliable).
- **Variables:** only indicators published in the same form in every year are kept. 2023-only counts and
  denominators are excluded.
- **ACT codes:** composite = TEA code `CAA`. `CAC` is ACT Science; the earlier audit dictionary had this wrong.
- **Raw data checks:**
  - year-over-year copy test on all 13 indicators: 0–4% identical, so no copying like the old Excel file
  - range checks: 0 violations
  - SAT total = ERW + Math in every row
  - 600 random cells re-read from the raw files: all match

## 5. Temporal features (`temporal_features.csv`, also included in `nodes.csv`)
For each of the 11 targets, row (campus, t) holds values computed **only from years before t**. A missing
input gives NaN; there is no carry-forward.

| Feature | Definition | First year available |
|---|---|---|
| `_lag1`, `_lag2`, `_lag3` | value at t−1, t−2, t−3 | 2020 / 2021 / 2022 |
| `_change_lag1` | lag1 − lag2 | 2021 |
| `_ma2`, `_ma3` | mean of lag1..lag2 / lag1..lag3 | 2021 / 2022 |
| `_trend3` | OLS slope over t−3..t−1, i.e. (lag1 − lag3)/2; needs lag1, lag2 and lag3 | 2022 |
| `_n_prior_obs` | count of non-missing prior years | 2019 |

## 6. Graph edge types

| Type | Status | Construction |
|---|---|---|
| District | built | clique of same-`district_id` campuses per year, weight 1 |
| Geographic | built | 5 nearest within 100 km (haversine), weight exp(−km/20); virtual campuses excluded |
| Academic | built 2020–2023 | 7 per-target sets: 10-NN on the z-scored year t−1 profile without the predicted target's family, needs ≥4 shared features, weight exp(−d²/2σ²) |
| Demographic | built 2020–2023 | 10-NN on the z-scored year t−1 TAPR composition (11 shares + log enrollment), needs ≥6 shared features, weight exp(−d²/2σ²) |
| Transition | **not built** | needs elementary/middle nodes and a feeder-pattern map |
| ESC region | built | clique of same-region campuses per year, weight 1 (~95k edges/year) |
| Temporal | built | same campus t−1 → t, consecutive years only |

**Target exclusion.** The academic edges come in 7 per-target sets (`edge_set` column; mapping in
`data/graph/academic_edge_sets.csv`). Each set leaves the predicted variable's family out of the similarity
profile, and no set uses any year-t value. When training a model for target T, load only the rows where
`edge_set` equals T's set.

## 7. Missing data and limitations
**Missing data:**
- 20.2% of indicator cells are missing; 14.5% of the cells that can exist.
- SAT 21%, ACT 24% (many are small-group masks: 846 SAT, 1,075 ACT).
- Participation 4.5%, criterion 6.7%, attendance 0.8%, dropout 1.1%.

**Limitations:**
1. **Short history:** 8 years. The 2017 nodes have no lags and no academic or demographic edges, and 3-year features exist
   only for 2020–2024.
2. **Survivorship:** the cohort is defined by the 2025 Directory, so 80 closed or renumbered campuses are
   excluded.
3. **Static metadata from 2025:** district, region, grade range and coordinates are applied to every year.
4. **Retention** covers 98–100% of school-years (2017: 98%). **Risk Index weights** are last year's defaults, not re-tuned, and are pending
   professor review.
5. **Definition change confirmed for 2017:** the "above criterion" statewide mean is 19.3% in 2017 and 34.8% in 2018 (median school +15.2 points, against −1.5 the year after). The 2017 values are blanked and flagged (`sat_act_above_criterion_supp` = 4; `known_breaks` in `configs/data_sources.yaml`). It also falls from 31.6% to 25.0% over 2019–2023; check TEA's glossary before using it as a target. SAT total ≠ ERW + Math by 2–12 points in 7% of 2017 rows (0.2% in 2018, none after); kept, with a looser check for 2017.
6. **Undocumented code:** the meaning of `-2` is not documented (33 cells).
7. **COVID:** 2021 has fewer ACT results. `covid_affected_year` flags 2020–2021.
8. **⚠️ Graph parameters flagged for professor review:** k, radius, τ and profile features are reasoned
   defaults in `configs/graph.yaml` (`status: pending_professor_review`, Q10).
9. **Split:** `configs/split.yaml` = train 2017–2022, validate 2023, test 2024. The 2017 rows have no
   history, so the effective training targets are 2018–2022.

## 8. Files and schemas

| Path | Rows × cols | Key | Content |
|---|---|---|---|
| `data/processed/master_school_year.csv` | 13,140 × 55 | campus_id, year | indicators, retention, risk index, demographics, `_supp` flags, district/region, source files |
| `data/processed/risk_index_components.csv` | 13,140 × 13 | campus_id, year | the 9 risk components and weight coverage |
| `data/processed/school_metadata.csv` | 1,670 × 31 | campus_id | Directory + NCES static metadata |
| `data/processed/temporal_features.csv` | 13,140 × 132 | campus_id, year | lag/MA/trend features |
| `data/processed/subgroup_school_year.csv` | long | campus_id, year, group, indicator | subgroup outcomes |
| `data/processed/data_dictionary.csv` | — | file, column | descriptions, raw codes, units |
| `data/processed/reports/data_quality_report.csv` | — | — | build-time checks and counts |
| `data/processed/reports/processed_validation.csv` | 64 | — | independent re-validation results |
| `data/graph/nodes.csv` | 13,140 × 200 | campus_id, year (`node_index` per year) | node table |
| `data/graph/node_feature_catalog.csv` | 200 | column | static / annual / target role |
| `data/graph/edges_<type>.csv` | see §6 | — | `src_campus_id, dst_campus_id, src_year, dst_year, year, edge_type, weight, directed, distance, method` |
| `data/graph/reports/graph_quality_report.csv` | 225 | — | graph checks |
| `data/graph/academic_edge_sets.csv` | 12 | target | which academic edge set to use per target |

## 9. Loading

```python
import pandas as pd

ID = {"campus_id": str, "district_id": str, "esc_region": str, "src_campus_id": str, "dst_campus_id": str}
nodes = pd.read_csv("data/graph/nodes.csv", dtype=ID)
edges = {t: pd.read_csv(f"data/graph/edges_{t}.csv", dtype=ID)
         for t in ["district", "geographic", "academic", "region", "temporal"]}

# one snapshot: map campus_id -> node_index for that year, add reverse direction for undirected edges
y = 2022
idx = nodes[nodes.year == y].set_index("campus_id").node_index
e = edges["geographic"].query("year == @y")
src, dst = idx[e.src_campus_id].to_numpy(), idx[e.dst_campus_id].to_numpy()
edge_index = [list(src) + list(dst), list(dst) + list(src)]      # undirected: both directions
```

**Rules:**
- Always pass `dtype=str` for every ID column. Read as numbers, IDs lose their leading zeros and joins fail
  silently.
- Use as inputs only the columns whose `role` in `node_feature_catalog.csv` starts with "input". The 27
  year-t observations are targets or target masks: the indicators (including `act_science`), the `_supp`
  flags and `has_graduates_reported`.
- Rebuild everything with:

```bash
python3 src/data/setup_raw.py --zip Dataset.zip --tapr <TAPR folder> --retention <retention folder>
python3 src/data/build_master.py && python3 src/data/validate_processed.py
python3 src/graphs/build_graph_data.py && python3 src/graphs/validate_graph.py
```

## 10. Final checklist

| Deliverable | Status | Evidence |
|---|---|---|
| Raw files preserved | ✅ pass | `Dataset.zip` and the extracted `pro/` are unchanged; the earlier audit files were not overwritten |
| master_school_year.csv | ✅ pass | 64/64 independent checks, incl. demographic and retention cells re-read from raw (`processed_validation.csv`) |
| school_metadata.csv | ✅ pass | IDs unique, 100% coordinates, inside Texas |
| temporal_features.csv | ✅ pass | lag equality and no-future tests |
| data_quality_report.csv, DATA_PIPELINE.md | ✅ pass | present and consistent with the re-validation |
| nodes.csv | ✅ pass | identical to the processed data; no duplicates |
| District, geographic, region, temporal edges | ✅ pass | exact counts, recomputed distances, no duplicates or invalid IDs |
| Academic edges | ✅ pass (2020–2023) | 6 per-target sets; 600/600 recomputed; target excluded per set; shuffle test passes. None for 2019 (no history) |
| Demographic edges | ✅ pass (2020–2023) | 400/400 recomputed; shuffle test passes; none for 2019 (no 2018 profile) |
| Transition edges | ❌ incomplete | out of scope with the current data (no elementary/middle nodes or feeder map) |
| graph_quality_report.csv, GRAPH_DATA_DOCUMENTATION.md | ✅ pass | 84 pass / 0 fail |
| Retention target | ✅ pass (2020–2023) | 2019 missing: 2018-19 file not downloaded |
| Academic Risk Index | ✅ pass (weights pending review) | corrected Eq. 1; 40/40 hand-recomputed; direction and no-future tests pass |
| split.yaml | ✅ done | train 2017–2022 / val 2023 / test 2024 |
| Graph parameters confirmed by professor | ⚠️ flagged | `configs/graph.yaml` status pending_professor_review |

**Verdict:** the 2017–2024 school × year dataset, the retention target and 6 of the 7 edge types pass
validation and can be used for baselines and graph models. Still incomplete: transition edges (need feeder
data the PPT doesn't list) and 2019 retention (needs the 2018-19 file). Risk Index weights await the professor's
review.
