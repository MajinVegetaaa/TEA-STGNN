# Graph data — TEA STGNN (2019–2023)

```bash
python3 src/data/build_master.py        # data/processed/  (see DATA_PIPELINE.md)
python3 src/data/validate_processed.py  # 54 independent checks on data/processed/
python3 src/graphs/build_graph_data.py  # data/graph/  (~5 s)
python3 src/graphs/validate_graph.py    # data/graph/graph_quality_report.csv (72 pass / 0 fail)
```

The graph is a sequence of **yearly snapshots G₂₀₁₉ … G₂₀₂₃**. A node is one campus in one year. Spatial
edges connect nodes of the same year, and temporal edges connect a campus to itself in the next year. No model
is trained here.

## Nodes — `data/graph/nodes.csv`

There are 8,202 nodes (one per row of `master_school_year.csv`) and 139 columns. `node_index` numbers the nodes
0…n−1 within each year, in `campus_id` order, so it can be used directly as the PyG node index of that
year's snapshot.

| Year | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|
| Nodes | 1,623 | 1,633 | 1,643 | 1,652 | 1,651 |

`node_feature_catalog.csv` classifies every column. In summary:

| Group | Columns | Varies | Use |
|---|---|---|---|
| Keys | `campus_id`, `year`, `node_index` | — | identifiers |
| Same-year indicators | `sat_total`, `sat_erw`, `sat_math`, `act_composite`, `act_english`, `act_math`, `act_science`, `sat_act_participation`, `sat_act_above_criterion`, `attendance_rate`, `chronic_absenteeism`, `dropout_9_12`, `dropout_7_8` | annually | **targets at year t**; as inputs only for predicting a later year |
| Suppression flags | `<indicator>_supp` (0 none, 1 = raw −1, 2 = raw −2) | annually (observed in year t) | **target mask only**: they reveal year-t reporting, so not an input for year t |
| Year flags | `covid_affected_year` (input); `has_graduates_reported` (year-t observation: mask only) | annually | see role |
| Temporal features | `<target>_lag1/2/3`, `_change_lag1`, `_ma2`, `_ma3`, `_trend3`, `_n_prior_obs` (88 columns) | annually, from years < t only | node features for predicting year t |
| Static metadata | `district_id`, `district_name`, `district_type`, `county_id`, `county_name`, `esc_region`, `charter_type`, `is_charter`, `is_magnet`, `is_virtual_or_hybrid`, `is_residential_facility`, `serves_full_9_12`, `lowest_grade`, `grade_range`, `latitude`, `longitude`, `nces_locale_code`, `urbanicity`, `cbsa_code`, `campus_name` | static (Directory Oct 2025, NCES 2024–25) | features / grouping |

`enrollment_oct2025` is deliberately **left out** of the nodes because it is measured after the study period.

**Leakage rule for the model team.** To predict year t, use only the columns whose catalog `role` starts
with "input": temporal features, static metadata and `covid_affected_year`. All 27 columns observed in
year t are labelled "same-year observation" and must not be inputs for year t:
- the 13 indicators, including `act_science`, which is part of `act_composite`
- the 13 `_supp` flags
- `has_graduates_reported`

`validate_graph.py` checks these labels.

## Edge files — common schema

| Column | Meaning |
|---|---|
| `src_campus_id`, `dst_campus_id` | 9-digit campus IDs (read as text) |
| `src_year`, `dst_year` | years of the two endpoint nodes |
| `year` | snapshot the edge belongs to (= `dst_year`) |
| `edge_type` | district, geographic, academic, region, temporal |
| `weight` | in (0, 1] |
| `directed` | 0 = undirected, stored **once** with `src < dst` (add the reverse when loading into PyG); 1 = directed |
| `distance` | km (geographic) or RMS z-distance (academic); empty otherwise |
| `method` | how the edge was built, including parameters |

## The seven edge types

| # | Type | File | Status | Edges/year | Mean degree |
|---|---|---|---|---|---|
| 1 | District membership | `edges_district.csv` | built | 3,737–3,929 | 4.7 |
| 2 | Geographic proximity | `edges_geographic.csv` | built | 5,087–5,167 | 6.3 |
| 3 | Academic similarity | `edges_academic.csv` | built for 2020–2023, 6 per-target sets | 9,800–10,900 per set | 12–13 |
| 4 | Demographic similarity | `edges_demographic.csv` | **not built: data unavailable** (header only) | 0 | — |
| 5 | Academic transition | `edges_transition.csv` | **not built: data unavailable** (header only) | 0 | — |
| 6 | ESC region | `edges_region.csv` | built | 92,844–96,833 | ~116 |
| 7 | Temporal | `edges_temporal.csv` | built | 1,623–1,651 (into 2020–2023) | — |

### 1. District membership
Every pair of cohort campuses in the same 6-digit district in that year (full clique), weight 1.
- `district_id` comes from the Directory and always equals `campus_id[:6]`.
- 901 of 1,045 districts have only one high school, so about 900 nodes per year have no district edge. This
  is expected for rural Texas.

### 2. Geographic proximity
- **Inputs:** NCES 2024–25 coordinates joined by NCES ID (100% reliable match). Distances are haversine.
- **Neighbours:** each campus links to its **k = 5 nearest** campuses in the same year. Links longer than
  **100 km** are dropped. The union of all neighbour lists is kept, so mean degree is 6.3.
- **Weight:** `exp(−km / 20)`. The median edge is 12.2 km (weight 0.54) and the longest is 98.5 km.
- **Exclusions:** the 8 virtual/hybrid campuses get no geographic edges, because their address is not where
  their students are. They are the only geographic isolates.
- **Co-located campuses:** 10 campuses share coordinates with another campus. Their distance is 0 km, so the
  weight is 1.
- **Why these values (⚠️ PENDING PROFESSOR REVIEW):** all graph parameters live in `configs/graph.yaml`
  (`status: pending_professor_review`, Questions PDF Q10). k = 5 and
  100 km were chosen because 99% of campuses have their 5th-nearest neighbour within 80 km. Tune on the
  validation year.

### 3. Academic similarity (prediction-time safe, one edge set per target)
For snapshot t, each campus's profile is built **only from year t−1 values** of up to 7 indicators:
- `sat_total`, `act_composite`, `sat_act_above_criterion`, `sat_act_participation`, `attendance_rate`,
  `dropout_9_12`
- `chronic_absenteeism`, from the 2021 snapshot on, since it was not published for 2019

**The predicted target is left out of its own similarity edges.** There are six edge sets (column
`edge_set`). Each drops the target family it is used for; `academic_edge_sets.csv` maps each target to its
set:

| Edge set | Dropped from profile | Use it for targets |
|---|---|---|
| `excl_sat` | `sat_total` (= ERW + Math) | `sat_total`, `sat_erw`, `sat_math` |
| `excl_act` | `act_composite` | `act_composite`, `act_english`, `act_math` |
| `excl_criterion` | `sat_act_above_criterion` | `sat_act_above_criterion` |
| `excl_participation` | `sat_act_participation` | `sat_act_participation` |
| `excl_attendance` | `attendance_rate`, `chronic_absenteeism` (both from absence days) | `attendance_rate`, `chronic_absenteeism` |
| `excl_dropout` | `dropout_9_12` | `dropout_9_12` |

Without this, the academic layer would amount to a nearest-neighbour lookup on the target's own history.
That is the KNN approach the brief found does not beat persistence.

Steps (for each set and year):
1. **Normalise.** z-score each feature with the mean and standard deviation of the year t−1 cross-section,
   so no year-t statistic is used.
2. **Distance.** Take the root-mean-square z-difference over the features both schools reported. The pair
   must share **at least 4** features, otherwise it has no distance. Missing values are never imputed.
3. **Neighbours.** Link each campus to its **10 most similar** campuses and keep the union. This gives
   about 9,800–10,900 edges per set per year, with mean degree 12–13.
4. **Weight.** `exp(−d² / 2σ²)`, where σ is the median edge distance for that set and year (0.20–0.26).

Results:
- **No 2019 academic edges**, because there is no 2018 data.
- Campuses with no usable t−1 profile have no academic edges in that set. These are new campuses, or
  campuses with fewer than 4 of the set's features reported.

**Leakage checks.** `validate_graph.py` shuffles every year-t indicator across campuses and rebuilds all 24
set-year edge lists; they come out identical. It also checks that each target's set excludes the target,
and recomputes 600 sampled distances from that set's own features.

Trajectory measures (Pearson correlation, DTW) were not used. With at most 4 prior years they would be
unstable. Revisit them if longer history is added.

### 4. Demographic similarity — not built yet (data listed in the PPT, not yet downloaded)
The architecture PPT asks for this edge type (slide 7, Edge Type 4), and lists the data on slide 2 ("Student
Data — student demographics") and slide 3 ("Student Demographics: ELL, Special Ed, At-Risk, Race"). It needs each campus's student **composition**: % economically disadvantaged, % English learners,
% special education, race/ethnicity shares and enrollment, for every year. `Dataset.zip` does not contain it.
- `subgroup_school_year.csv` gives subgroup *outcomes* (e.g. the SAT average of economically disadvantaged
  students), not subgroup *shares*.
- The Directory's `enrollment_oct2025` is a single post-period value.
- **Required source:** TEA TAPR campus student-information files (or PEIMS enrollment by student group),
  2019–2023.

### 5. Academic transition — not built (PPT slide 7, Edge Type 5)
This needs elementary and middle-school nodes, plus a feeder-pattern map (which middle school sends students
to which high school). The cohort is high schools only, and TEA does not publish feeder patterns.
- **Required source:** district attendance-zone or feeder-pattern files, or student-level transition data
  (not public).
- **Decision needed:** feeder patterns are not among the PPT's slide-2 datasets, so ask the professor
  whether transition edges stay in scope (Questions PDF, Q6).

### 6. ESC region
Every pair of cohort campuses in the same ESC region in that year (full clique), weight 1.
- Regions range from 34 to 231 campuses, which gives about 95k edges per year (50 MB CSV).
- For models where that density is costly, an equivalent lighter option is a region "hub" node or one
  region embedding. Pick one when designing the model.

### 7. Temporal
`(campus, t−1) → (campus, t)` when both rows exist, directed forward, weight 1. Edges never bridge a missing
year; no campus currently has a gap. There are no backward edges, so information only flows from past to
future.

## Quality report — `data/graph/graph_quality_report.csv`
72 checks pass and 0 fail; 124 more rows are informational counts. The checks cover:
- **Nodes:**
  - node keys equal the master keys, with no duplicates and all IDs 9 digits
  - node values identical to `master_school_year`, `temporal_features` and `school_metadata`
  - no missing coordinates
  - every year-t column is labelled as not an input
- **Every edge file:**
  - every endpoint is an existing node
  - no invalid IDs, missing weights, duplicates or self-loops
  - weights in (0, 1]
  - undirected pairs stored once
- **District and region edges:** endpoints really share the group, and edge counts equal the exact
  pair counts for each year.
- **Geographic edges:**
  - distances recomputed with an independent great-circle formula (max error 0.0001 km)
  - weights match the formula and stay within the 100 km cap
  - every 5-nearest link is present
  - no virtual campuses
- **Academic edges:**
  - 500 sampled distances recomputed in a plain loop: 500/500 match
  - weights match the formula
  - none in 2019
  - the shuffle-the-target leakage test passes
- **Temporal edges:** same campus, consecutive years, forward only, and the count equals the number of
  consecutive pairs.
- **Isolated nodes:**
  - by layer: about 900 per year in the district layer (one-HS districts), 5–8 in the geographic layer
    (virtual campuses), 59–140 in the academic layer (no prior-year profile), and all 2019 nodes in the
    academic layer
  - temporal layer: the 2019 nodes have no incoming edge (no 2018), and the 10 campuses that first appear
    after 2019 have no incoming edge in their first year
  - **0 isolated nodes** across all spatial edge types, every year

## Leakage summary
| Risk | Status |
|---|---|
| Same-year targets in features | Temporal features use years < t only (tested). Same-year indicators are labelled as targets. |
| Same-year targets in academic edges | Edges use t−1 only, and the predicted target's family is dropped per edge set. The shuffle test passes. |
| Normalisation statistics | Computed from the t−1 cross-section only. |
| Static metadata from 2025 | District, region, coordinates, charter and grade range are a 2025 snapshot applied to all years. They are structural, not outcomes, but a campus that changed district or type would be misdescribed in early years. `enrollment_oct2025` is excluded from nodes. |
| Cohort definition | It uses "reported graduates in ≥1 year of 2019–2023", which looks at the whole period. This decides which schools are in the study, not any feature value. |
