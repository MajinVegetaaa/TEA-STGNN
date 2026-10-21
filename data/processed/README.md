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
| `audit/excluded_not_in_directory.csv` | school | campuses with graduates that are missing from the 2025 directory (excluded) |
| `audit/name_mismatches.csv` | school × year | campus names that differ between a yearly file and the directory (renames) |
| `audit/geo_rejected_matches.csv` | school | coordinate matches rejected as unreliable (currently none) |
