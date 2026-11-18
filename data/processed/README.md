# Processed data

Built from `data/raw/` by `python3 src/data/build_master.py`; checked by `python3 src/data/validate_processed.py`.
Never edit by hand; rebuild instead. Always read ID columns as text (`dtype=str`) to keep leading zeros.

| File | One row per | What it holds |
|---|---|---|
| `master_school_year.csv` | school × year (11,546) | **the main table**: all targets, demographics, risk index, suppression flags, source file of each row |
| `school_metadata.csv` | school (1,670) | name, district, county, ESC region, charter/magnet/virtual flags, grade range, coordinates, urban/rural |
| `temporal_features.csv` | school × year | history features built only from earlier years (lags, change, averages, trend) |
| `risk_index_components.csv` | school × year | the 9 parts of the Academic Risk Index (0–1, higher = riskier) and weight coverage |
| `subgroup_school_year.csv` | school × year × student group × indicator | results for subgroups (e.g. econ. disadvantaged SAT average) |
| `data_dictionary.csv` | column | meaning, raw source code and unit of every column above |
| `reports/data_quality_report.csv` | check | every count and check printed while building |
| `reports/processed_validation.csv` | check | the independent validator's pass/fail results |
| `reports/addition_sources.csv` | source × year | verified year evidence, original filename and SHA-256 |
| `reports/addition_coverage.csv` | field × year | matched campuses, observed values and missing/suppression statuses |
| `reports/addition_source_cells.csv` | selected source cell | reproducible examples from independent source-to-master checks |
| `reports/master_data_gate.json` | validation run | pass/fail and input/output hashes; stale passes cannot authorize expansion |
| `expanded_temporal_features.csv` | campus × year | original histories plus raw added histories and 102 publication-filtered inputs; see `configs/feature_availability.md` |
| `reports/expanded_feature_receipt.json` | feature build | master/availability hashes and eligible input-column list |
| `audit/excluded_not_in_directory.csv` | school | campuses with graduates that are missing from the 2025 directory (excluded) |
| `audit/name_mismatches.csv` | school × year | campus names that differ between a yearly file and the directory (renames) |
| `audit/geo_rejected_matches.csv` | school | coordinate matches rejected as unreliable (currently none) |
