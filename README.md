# TEA STGNN

Forecasting Texas high-school outcomes with a spatio-temporal graph neural network.
See `TEA_STGNN_Project_Brief.pdf` and `Team_Work_Division.pdf` for the plan.

## Layout
- `data/raw/` — TEA downloads, untouched (not in git)
- `data/processed/` — `panel.parquet` + data dictionary (rebuilt by script)
- `src/data/` cleaning & features · `src/eval/` split & metrics · `src/baselines/` · `src/graphs/` · `src/models/`
- `configs/split.yaml` — train ≤2022, val 2023, test 2024
- `results/` — prediction CSVs (`model,seed,campus_id,year,target,y_true,y_pred`)

## Rebuild data and graph
```bash
unzip -q Dataset.zip -d data/raw/tea
python3 src/data/build_master.py && python3 src/data/validate_processed.py
python3 src/graphs/build_graph_data.py && python3 src/graphs/validate_graph.py
```
Docs: `DATA_PIPELINE.md`, `GRAPH_DATA_DOCUMENTATION.md`, `DATA_LEAD_HANDOFF.md`.
