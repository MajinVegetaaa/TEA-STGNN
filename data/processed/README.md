# data/processed — built by `make data` / `make features` (not in git; never edit by hand)

Always read ID columns as text (`dtype=str`). Details: [`docs/DATA.md`](../../docs/DATA.md).

| File | One row per | Content |
|---|---|---|
| `master_school_year.csv` | school × year (11,546) | **the main table**: outcomes, demographics, added sources, finance, risk index, masks, provenance |
| `school_metadata.csv` | school (1,670) | names, district, region, charter/virtual flags, grade range, coordinates, locale |
| `d1_temporal_features.csv` | school × year | **current** history features (every source, years < t) — read by graph v2 |
| `temporal_features.csv`, `expanded_temporal_features.csv` | school × year | Phase 3 features; expanded-experiment features (records) |
| `risk_index_components.csv`, `subgroup_school_year.csv` | school × year | risk-index parts; student-subgroup results |
| `data_dictionary.csv` | column | meaning, TEA source code and unit of every column |
| `reports/` | check | audit and validation results, source coverage, the hashed master-data gate, feature receipts |
| `audit/` | school | excluded, renamed and rejected-match campuses |
