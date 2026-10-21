# Graph data

Built from `data/processed/` by `python3 src/graphs/build_graph_data.py`; checked by
`python3 src/graphs/validate_graph.py`. One graph snapshot per year, 2019–2023. Full rules are in
`GRAPH_DATA_DOCUMENTATION.md`.

| File | What it holds |
|---|---|
| `nodes.csv` | one node per school per year: targets, history features, demographics, static school info; `node_index` = 0…n−1 within each year |
| `node_feature_catalog.csv` | for every node column: static or yearly, and whether it may be used as a model input |
| `edges_district.csv` | schools in the same district |
| `edges_geographic.csv` | each school's 5 nearest schools within 100 km |
| `edges_academic.csv` | 10 most similar schools on last year's results; one set per target (`edge_set` column) |
| `academic_edge_sets.csv` | which `edge_set` to use for each target |
| `edges_demographic.csv` | 10 most similar schools on last year's student make-up |
| `edges_region.csv` | schools in the same ESC region |
| `edges_temporal.csv` | the same school, year t−1 → t |
| `edges_transition.csv` | empty (header only): needs feeder data; waiting on the professor |
| `reports/graph_quality_report.csv` | the graph validator's checks and counts |

All edge files share one layout: `src_campus_id, dst_campus_id, src_year, dst_year, year, edge_type, weight,
directed, distance, method`. Undirected edges are stored once (`src < dst`); add the reverse when loading.
