# TEA STGNN

Forecasting next year's outcomes (SAT, ACT, TSI, college readiness, participation, attendance, dropout, retention,
graduation, risk index) for 1,670 Texas public high schools with a spatio-temporal graph neural network.

**Start here: [`ROADMAP.md`](ROADMAP.md)** — the plan, the decisions, the step we are on and how to run each step.

## Setup

Python 3.13 (3.12 also works).

```bash
pip install -r requirements.txt            # data pipeline, graph build, tabular baselines
pip install -r requirements-gpu.txt        # graph models (torch 2.12.1, PyG); see docs/PROJECT_LOG.md §4 for CUDA wheels
```

Raw TEA downloads are not in git. Put them in `data/raw/` with
`python3 src/data/setup_raw.py --zip data/archives/Dataset.zip --tapr <folder> --retention <folder> --tsi <folder>`
and see [`data/raw/README.md`](data/raw/README.md) for the other sources.

## Run

```bash
make help     # list everything
make check    # validate current files and run tests (no rebuild)
make data     # audit raw files -> master table -> validate
make graph    # rebuild + validate the expanded graph
```

## Docs

| File | What it covers |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | Plan, decisions D1–D6, steps S0–S8 and their status |
| [`DECISIONS.md`](DECISIONS.md) | Every decision, dated |
| [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) | How raw files become `data/processed/` (cohort, years, sources, checks) |
| [`docs/GRAPH_DATA_DOCUMENTATION.md`](docs/GRAPH_DATA_DOCUMENTATION.md) | Graph nodes, edges, weights and checks |
| [`docs/DATA_LEAD_HANDOFF.md`](docs/DATA_LEAD_HANDOFF.md) | Data hand-off and audit summary |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | History of Phases 1–5 (the old project plan) |
| [`results/phase5/README.md`](results/phase5/README.md) | Current graph-model baseline scores |

Reference material (PPT, project brief, papers) lives in `docs/references/` and is not in git.
