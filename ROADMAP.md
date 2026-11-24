# TEA STGNN — Roadmap (start here)

**Goal.** Forecast next year's outcomes for every Texas public high school and test whether learning from related
schools (a spatio-temporal graph neural network) beats predicting each school on its own.
**Baseline to beat:** MTGNN (Wu et al., KDD 2020) plus the Phase 3 tabular models. **Final model:** our proposed
heterogeneous STGNN (relation-aware graph attention + GRU + multi-task heads).

Every decision, dated: [`DECISIONS.md`](DECISIONS.md). Details: [`docs/DATA.md`](docs/DATA.md), [`docs/GRAPH.md`](docs/GRAPH.md), [`docs/MODELS.md`](docs/MODELS.md).

## How we work

- Work happens in **steps (S0, S1, …)**. Each step runs only on command ("run S3") and ends with a short report:
  what changed, what passed, what is next. Then it stops.
- A step starts only when the decisions it depends on are settled.
- Each step has a **done-when** condition. If it is not met, the report says so; we do not move on silently.
- The 2024 test year is used **once**, in S6, after the final model is locked.

## Decisions (settled 2026-11-21)

| ID | Decision | Chosen |
|---|---|---|
| D1 | What is known on forecast day | **Retrospective:** every source's values up to t−1 are inputs for year t, nothing from year t. A research benchmark, not a deployable forecast (publication dates not enforced). In `configs/protocol.yaml` |
| D2 | Targets | **The current 10**; accountability ratings later only if the professor asks |
| D3 | MTGNN | **Wu et al., KDD 2020** ("Connecting the Dots"), as the main graph baseline |
| D4 | Graph settings | **Freeze, do not tune on 2023.** Keep k=5 geographic (≤100 km), k=10 academic and demographic. Fix the kernel width σ from the training years. Per-school geographic scale (σᵢ = distance to 5th neighbour, floor 1 km). Raw distances as GAT edge features. Robustness check over k in S7. **Applied in S1b: `configs/graph_v2.yaml`, `data/graph_v2/`** |
| D5 | District and region | **Test as an ablation** (cliques vs hierarchy nodes vs features only) |
| D6 | What the model predicts | **Change from last year** (persistence shortcut), not the raw level |

Why D4: tuning on a single validation year overfits model selection (Cawley & Talbot, JMLR 2010); with a fixed 20 km
scale, rural and town schools get about half the geographic weight of city schools (median 5th-neighbour distance
28 km vs 6 km), and self-tuning local scales (Zelnik-Manor & Perona, 2004) remove that bias.

## Steps

| Step | What | Depends on | Done when | Status |
|---|---|---|---|---|
| **S0** | Clean the repo: folder layout, one command (`make`), docs into `docs/`, no science changes | — | validators give the same counts; data and results byte-identical | **done 2026-11-21** |
| **S1** | Lock the protocol: one config for targets, split, metrics, persistence shortcut and forecast-timing rule (`configs/protocol.yaml`, `src/eval/protocol.py`); fix the "2023 test year" wording | D1, D2, D6 | Phase 3 "numbers to beat" reproduce exactly | **done 2026-11-21** (19/19, max diff 4e-15; bar = 0.953) |
| **S1b** | Graph v2: σ fixed from training years, per-school geographic scale, raw distances stored as edge features; old graph archived | D4 | graph validator passes; v1 vs v2 differences documented | **done 2026-11-21** (86/86; same edges, re-weighted; geographic weight by locale now 0.47–0.50 vs 0.34–0.74; PyG snapshot is built at the start of S2 in the GPU environment: `make snapshot`) |
| **S1c** | Numbers to beat under D1: make every source's lag-1 history an input (staff, AP/IB, STAAR, graduation and finance; today only publication-filtered `available_*` versions are inputs), then rerun the tabular baselines (persistence, LR, KNN, RF, GB, XGBoost; level/relative/zscore) for all 10 targets on graph v2, select on 2023 | S1, S1b | one `results/s1c/numbers_to_beat.csv` that every later step compares against | **done 2026-11-21**: bar on 2023 = **0.930** (10 targets, persistence counts as a candidate); `results/s1c/numbers_to_beat.csv`, `bar.json` |
| **S2** | One training harness for every neural model: data loading, masks, single/multi-task loss, early stopping on 2023, 5 seeds, equal tuning budget | S1, S1b, S1c | Phase 5 GCN+LSTM reproduces within tolerance on the harness | **done 2026-11-22**: `src/train/` harness reproduces Phase 5 GCN+LSTM (9/9: 2023 MAE of all 59 candidates identical, 2024 per-seed MAE within 5.5e-6); change label, multi-task and graph v2 smoke-tested; v2 snapshot built on the laptop (`make snapshot`), no GPU needed |
| **S3** | Go/no-go diagnostic: GConvGRU and GraphWaveNet with the change-from-last-year target | S2 | we know whether the shortcut closes the gap to Phase 3 | **done 2026-11-22: no-go as is.** Best: GraphWaveNet, v2, change label = 1.033 ± 0.06 vs bar 0.930 (GConvGRU 1.077). Change label helps both; graph models beat persistence on test-score targets (SAT 0.900 beats the bar's 0.911) but fail on slow-moving rates (attendance, dropout, retention, graduation). Design for S4/S5 to be decided (see DECISIONS.md) |
| **S4** | MTGNN baseline (Wu et al. 2020) on graph v2, change label, shared budget B — see "S4/S5 design" below | S2, S3, D3 | implemented with tests; 2023 score over 5 seeds next to the bar | **done 2026-11-22**: MTGNN 2023 score **1.011 ± 0.048** vs bar 0.930 (GraphWaveNet 1.033). Beats persistence on 6/10 targets and the bar on 2 (SAT 0.877 vs 0.911, TSI 0.981 vs 0.992); still above persistence on attendance (1.48), dropout, retention, participation, graduation |
| **S5** | Proposed model, pure STGNN (no hybrid), versions v0–v3 with budget B — see "S4/S5 design" below | S2, S3 | every version scored on 2023 (5 seeds); best version vs MTGNN and the bar, or the result documented honestly | Round 1 implementation and guards preserved; Round 2 code/configs prepared, not trained. Separate Graph V2 bundle contains 2018–2023 only (fingerprint `09f87650e47552392326034004e15ff985c1bf16fa981dd8eb5dcb8440fcd4d9`). Local CPU preflight passes; Kaggle T4 preflight is still required before any run. The corrected-snapshot MTGNN comparator remains first. Historical frozen archive is unavailable; the known 2024 graph tie discrepancy is an S6 review item. Kaggle commands, backup and resume: [`docs/KAGGLE_S5_ROUND2.md`](docs/KAGGLE_S5_ROUND2.md). |
| **S6** | Lock the final model with the rule below, then **one** 2024 test run: 5 seeds, paired bootstrap vs MTGNN and best baseline | S4, S5 | test table produced once | not started |
| **S7** | Paper experiments: relation ablations, static vs yearly graph, single vs multi-task, spatial vs temporal, robustness over k, breakdowns, attention analysis, at-risk ranking | S6 | every table and figure generated by script | not started |
| **S8** | Paper: tables and figures from `results/` by one script; IEEE draft | S7 | draft reviewed | not started |

**Final-model rule (fixed now, before any 2024 number):** for each target, the 2023 MAE divided by persistence's
2023 MAE, averaged over the 10 targets and 5 seeds. Lowest wins. If two models are within one seed standard
deviation, the simpler one wins. MTGNN and the proposed model get the same tuning budget.

## S4/S5 design (decided 2026-11-22, before any S4/S5 run)

Pure spatio-temporal GNN; no hybrid with the tabular models. All runs: graph v2, D1 inputs, **change-from-last-year
label** (D6), selection on 2023 only, 5 seeds, scored with the final-model rule.

**Shared budget B** (identical for MTGNN and every S5 version; a small grid, not a search, to avoid overfitting the
single validation year): width ∈ {32, 64} × dropout ∈ {0, 0.2}; Adam, learning rate 0.003, weight decay 1e-4; up to
100 epochs, the best 2023 epoch kept (stop after 20 epochs without improvement). Each grid point is chosen at seed 0,
then scored over 5 seeds. Budgets are recorded in each run's `protocol.json`.

**S4 — MTGNN** (Wu et al., KDD 2020): graph-learning layer (school embeddings → learned sparse adjacency, top-k), used
together with the v2 relations as fixed supports; mix-hop propagation (2 hops); dilated inception temporal
convolutions with kernel sizes {1, 2, 3} (adapted to 7 yearly steps); residual and skip connections. One model per
target, as in Phase 5.

**S5 — proposed model**, versions compared on 2023 (each adds one thing, so every gain is attributable):

| Version | Adds | Question it answers |
|---|---|---|
| v0 | encoder + GRU over years, **no graph** | is the graph needed at all? |
| v1 | + relation-aware attention: one GATv2 per relation using the v2 edge features, then attention over relations | does attending to typed neighbours help? |
| v2 | + gated persistence shortcut: correction × α per target, α starting at 0 (ReZero-style), so the model starts at "last year + average change" and must earn any departure | does protecting slow-moving targets help? |
| v3 | + one model for all 10 targets (equally weighted standardised losses) | does sharing across outcomes help? |

The final S5 model is the best version under the final-model rule (ties → simpler). The hybrid (graph correction of
the tabular bar) is **not** pursued; it stays recorded as a possible later framing.

## One command

```bash
make help        # list targets
make check       # run every validator and test on the current files (no rebuild)
make data        # audit raw files, rebuild the master table, validate it
make features    # rebuild the publication-filtered (expanded) features
make graph       # rebuild and validate graph v2, compare with v1
make snapshot    # graph v2 -> PyG snapshot (uses .venv-expanded)
make s2          # harness reproduces Phase 5 GCN + LSTM
make s5          # proposed STGNN v0-v3, 2023 only (five validation seeds)
make test        # unit tests
make protocol    # check the locked protocol reproduces the stored baseline numbers
```

## Where things are

```
ROADMAP.md  DECISIONS.md  README.md  Makefile
docs/                 DATA.md (sources, cleaning, features, checks), GRAPH.md (graph), MODELS.md (protocol, baselines, results)
docs/references/      PPT, project brief, papers, team division (not in git)
configs/              data_sources, split, graph, risk_index, feature/finance availability
src/data/             audit, cleaning, master table, sources, validation
src/graphs/           graph build, validation, PyG snapshots
src/baselines/        Phase 3 tabular baselines
src/models/           Phase 5 STGNN baselines (new: MTGNN in S4, proposed model in S5)
src/train/            neural harness: data.py (panel), trainer.py (training loop), families.py (models), run.py (CLI)
configs/experiments/  one YAML per neural run (model, graph, label, tasks, budget: hidden / lr / max_epochs)
src/eval/             metrics, comparisons, reports
tests/                unit and regression tests
data/                 raw/, processed/, graph_v2/ (current), graph_expanded/ (v1, Phase 5, frozen), graph/ (original, frozen), archives/
results/              Phase 3 (level/relative/zscore, selection.csv), expanded/, phase5/, frozen/ (do not edit)
```

New work from S2 on goes into new, clearly named folders (`src/train/`, `src/models/mtgnn/`, `src/models/proposed/`,
`configs/experiments/`, `results/s<N>/`) rather than into the existing scripts.

## Open questions for the professor

- Graph settings (D4): confirm "freeze + robustness check" with the two corrections above.
- Transition edges: keep in scope? TEA publishes no feeder data.
- Risk Index weights: accept the defaults?
- Venue, page limit, deadlines, authorship.

## Research questions for us (not built until decided)

- Trajectory similarity (DTW or trend correlation) vs the current one-year snapshot for academic edges.
- Learned adjacency (MTGNN-style) vs hand-built relations, or both.
- Whether attention is reliable with only 5 training years (v0 vs v1 in S5 answers part of this).
- How to frame the paper if graph models only tie strong tabular models (analysis-led contribution).
