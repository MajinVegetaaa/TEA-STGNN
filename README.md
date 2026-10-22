# TEA STGNN

Forecasting Texas high-school outcomes with a spatio-temporal graph neural network.
See `TEA_STGNN_Project_Brief.pdf` and `Team_Work_Division.pdf` for the plan.

## Layout
- `data/raw/` — downloads in numbered source folders, untouched (not in git; see `data/raw/README.md`)
- `data/processed/` — cleaned tables (`master_school_year.csv` etc.; see `data/processed/README.md`); `data/graph/` — yearly graph files
- `src/data/` cleaning & features · `src/eval/` split & metrics · `src/baselines/` · `src/graphs/` · `src/models/`
- `configs/data_sources.yaml` — every raw file and year used (2017–2024)
- `configs/split.yaml` — train 2017–2022 (targets from 2018), validate 2023, test 2024; `configs/graph.yaml`, `configs/risk_index.yaml` — settings pending professor review
- `results/level/`, `results/relative/`, `results/zscore/` — experiment runs (predictions format `model,seed,campus_id,year,target,y_true,y_pred`)

## Rebuild data and graph
```bash
python3 src/data/setup_raw.py --zip Dataset.zip --tapr <TAPR folder> --retention <retention folder>
python3 src/data/audit_raw.py            # proves every raw file is the right dataset and year
python3 src/data/build_master.py && python3 src/data/validate_processed.py
python3 src/graphs/build_graph_data.py && python3 src/graphs/validate_graph.py
```
Docs: `PROJECT_PLAN.md` (start here), `DATA_PIPELINE.md`, `GRAPH_DATA_DOCUMENTATION.md`, `DATA_LEAD_HANDOFF.md`, `DECISIONS.md`.
