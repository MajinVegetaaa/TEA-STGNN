# Models — protocol, baselines, neural models and results so far

## Protocol (`configs/protocol.yaml`, `src/eval/protocol.py`)

- **10 targets:** ACT, SAT total, college-ready %, SAT/ACT participation, dropout 9–12, attendance, retention,
  4-year graduation, Academic Risk Index, TSI (both subjects).
- **Split:** train 2018–2022, choose everything on 2023, test 2024 **once**; seeds 0–4.
- **Forecast rule (D1, retrospective):** everything observed up to t−1 may be an input for year t.
- **Scored rows:** target and last year's value observed, so every model is scored on the same schools as
  persistence ("same as last year").
- **Score:** 2023 MAE ÷ persistence MAE, averaged over targets (per seed; mean ± SD). Below 1 beats persistence.
  Ties within one seed SD go to the simpler model.
- **Neural label (D6):** the change from last year; forecast = last year + predicted change.
- `make protocol` checks that this code reproduces the stored baseline numbers (19/19).

## The bar to beat (S1c): 0.930 on 2023

Tabular models (persistence, linear regression, KNN, random forest, gradient boosting, XGBoost; with/without other
indicators, demographics and graph-neighbour averages) × 3 target versions (level; relative to the statewide
mean; z-score) × 2 input sets (Phase 3 inputs; + every added source at lag 1) = about 90 candidates per target. Per
target the best 2023 candidate is the bar, **persistence included** (`results/s1c/numbers_to_beat.csv`, `bar.json`).

| Target | Bar (chosen on 2023) | 2024 vs persistence |
|---|---|---|
| ACT | random forest + context, all inputs | −10.0% |
| SAT total | gradient boosting + context | −2.0% (n.s.) |
| College-ready % | random forest + context, all inputs | −14.1% |
| Participation | persistence | — |
| Dropout | linear regression, relative | −6.5% |
| Attendance | persistence | — |
| Retention | linear regression, z-score | −3.1% (n.s.) |
| Graduation | gradient boosting + region, all inputs | −8.1% |
| Risk index | gradient boosting + district, all inputs | −15.7% |
| TSI | gradient boosting + graph, all inputs | −8.2% |

Earlier bars, kept as records: the original Phase 3 (`results/selection.csv`, 8 targets, 0.953) and the expanded
experiment (`results/expanded/`, publication-filtered inputs, level only), which Phase 5 was compared with.

## Neural models (`src/train/`)

One harness for every neural model: `data.py` (snapshot → years × schools × targets), `trainer.py` (training loop:
level or change label, masked single- or multi-task loss, epoch chosen on 2023), `families.py` (model registry),
`run.py` (one YAML in `configs/experiments/` → selection on 2023; `--val-seeds` scores 2023 for all 5 seeds;
`--test` scores 2024 and is reserved for S6 for our own models).

**Reproduction:** on graph v1 the harness reproduces the Phase 5 GCN + LSTM benchmark (2023 identical, 2024 within
5.5e-6) and the 2023 selections of GConvGRU and GraphWaveNet exactly (`check_reproduction.py`).

### Phase 5 baselines (graph v1, level label, 2024)

GCN + LSTM, T-GCN, GConvGRU, A3TGCN and a fixed-support GraphWaveNet, each with the combined graph and the best
single relation (width 32, Adam 0.003, 1–30 epochs chosen on 2023). **None beat the tabular baselines on any target;
against persistence 0–3 of 10.** Details: `results/phase5/README.md`.

### S3 go/no-go (2023, 5 seeds; bar 0.930)

| Model | A: v1, Phase 5 inputs, level | B: v2, D1 inputs, level | C: v2, D1 inputs, change |
|---|---|---|---|
| GConvGRU | 1.135 | 1.228 | 1.077 |
| GraphWaveNet | 1.128 | 1.064 | **1.033** |

The change label helps both and halves the gap to the bar. GraphWaveNet (C) beats persistence on ACT, SAT,
college-ready, risk index and TSI — on SAT (0.900) it beats the bar (0.911) — but loses on slow-moving rates
(attendance 1.42, dropout 1.11, retention 1.08, graduation 1.05, participation 1.02). Next: MTGNN (S4) and the
proposed model (S5); design options under research (see `DECISIONS.md`, 2026-11-22).

## Running

```bash
make s1c                                   # tabular bar (~2.5 h on the laptop)
make s2                                    # harness reproduces Phase 5 GCN + LSTM (~1 min)
.venv-expanded/bin/python src/train/run.py configs/experiments/<name>.yaml --val-seeds
python3 src/eval/report_s3.py              # S3 table
```

**Environments.** Data, graph and tabular models: `requirements.txt` (Python 3.13; 3.12 works). Neural models:
`requirements-gpu.txt` (torch 2.12.1 + PyG 2.8.1); the laptop's `.venv-expanded` has these and runs every neural
step so far on CPU. For a GPU (Colab or Lightning AI): install torch from the CUDA wheel index
(`pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130`, `cu126` for older drivers), then
`pip install -r requirements.txt -r requirements-gpu.txt -f https://data.pyg.org/whl/torch-2.12.1+cu130.html`;
on Colab only Google Drive persists, so upload the raw data and results there.

## History

Phase 1 data and Phase 2 graph (Oct 2026) → Phase 3 tabular baselines → Phase 4 GPU environment → expanded sources
(Nov 2026) → Phase 5 graph baselines → finance → roadmap steps S0–S3 (Nov 2026). Every step and decision is dated in
`DECISIONS.md`; the full pre-roadmap project log is in git history (`docs/PROJECT_LOG.md`, removed 2026-11-22).
