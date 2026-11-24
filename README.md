# TEA STGNN

Forecasting next year's outcomes (SAT, ACT, TSI, college readiness, participation, attendance, dropout, retention,
graduation, risk index) for 1,670 Texas public high schools, 2018–2024, with a spatio-temporal graph neural network
that learns from related schools (same district, nearby, academically or demographically similar, same region).

**Start here: [`ROADMAP.md`](ROADMAP.md)** — the plan, the decisions, which step we are on.

## Docs

| File | What it covers |
|---|---|
| [`ROADMAP.md`](ROADMAP.md) | plan, decisions D1–D6, steps S0–S8 and their status |
| [`DECISIONS.md`](DECISIONS.md) | every decision with its reason, dated |
| [`docs/DATA.md`](docs/DATA.md) | sources, years, cohort, cleaning, history features, checks, limitations |
| [`docs/GRAPH.md`](docs/GRAPH.md) | graph versions, nodes, the 7 edge types and their weights, leakage safeguards |
| [`docs/MODELS.md`](docs/MODELS.md) | evaluation protocol, the bar to beat, neural harness, results so far, environments |
| [`docs/KAGGLE_S5_ROUND2.md`](docs/KAGGLE_S5_ROUND2.md) | Round 2 Kaggle preflight, run order, resume and result backup commands |

## Setup and run

Python 3.13 (3.12 works). Raw TEA downloads are not in git (see [`data/raw/README.md`](data/raw/README.md)).

```bash
pip install -r requirements.txt          # data, graph, tabular models
pip install -r requirements-gpu.txt      # neural models (torch 2.12.1 + PyG); GPU notes in docs/MODELS.md
make help                                # all commands
make check                               # validate everything and run the tests (no rebuild)
make data && make features && make graph && make snapshot   # rebuild from the raw downloads
```

Layout: `src/data/` (cleaning, sources, features) · `src/graphs/` (graph, PyG snapshots) · `src/baselines/`
(tabular models) · `src/train/` (neural harness) · `src/models/` (graph models) · `src/eval/` (protocol, metrics,
reports) · `configs/` · `tests/` · `results/<step>/`.
