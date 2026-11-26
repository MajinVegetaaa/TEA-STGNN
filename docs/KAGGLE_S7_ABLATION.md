# Kaggle run: S7 post-test ablation (all STGNN versions, both setups, 2022–2024)

**What it is.** Every proposed STGNN version (V0, V1_original, V1_shared4, V2, V3) trained in both setups and
scored on 2022, 2023 and 2024, plus MTGNN on 2022 and 2023. The setups are:
* `r2`, the Round 2 design: the model predicts the change from last year's value;
* `planA`: the model predicts the change from the shrinkage forecast, with the yearly statewide swing removed.

It answers: does Plan A help or hurt each version, does the graph / the gate / multi-task help, and which version is
best on average. It is **post-test** (2024 was already seen in S6), so it does not change the S6 result; it goes in
the paper as an ablation. Plan and contrasts: DECISIONS.md, 2026-11-26.

**Protocol (same for every run).** For a forecast of year s: train on the years before s−1, choose hidden size,
dropout and epochs on s−1 (budget B grid, seed 0), retrain on the years before s with those settings for seeds 0–4,
forecast s. For 2024 the settings are chosen on 2023 with the 2018–2023 bundle and the model is retrained on
2018–2023 with the 2018–2024 bundle, exactly as in S6 (Plan A uses the locked S6 shrinkage).

## Inputs

The repository at the commit that added this file (or later `main`), with:
* `data/graph_v2/pyg/s5_validation/` (2018–2023 bundle) and
* `data/graph_v2/pyg/s6_test/` (2018–2024 bundle, the one used for S6).

Both bundles are checked against the locked 2018–2023 content hash (`41283f26…`) before anything is trained; the run
stops if they differ. The Kaggle 2018–2023 bundle can be used if it passes this check, otherwise use the one built on
the laptop.

## Run

```bash
cd /kaggle/working/TEA-STGNN
python src/s7/ablation_run.py status                 # 26 pending, 7 reused from Step 4 / S6
python src/s7/ablation_run.py all --device cuda      # runs everything pending, cheapest first
python src/s7/ablation_run.py status                 # should end with "0 of 26 runs pending"
```

`all` can be stopped and started again (also in a new Kaggle session): finished runs are skipped, and an
interrupted run continues from its saved settings and finished seeds. A single run:
`python src/s7/ablation_run.py one --setup r2 --model v2 --origin 2023 --device cuda`.

Do not change any setting, model or year, and do not re-run a finished run to "improve" it. If a run fails, send the
error instead of working around it.

## Time

Measured on the laptop CPU: V0 about 1 minute per year; V3 (one model for all targets) a few minutes; V1, V1_shared4
and V2 about 30–50 minutes per year (tuning dominates); MTGNN longer. On a T4 the whole set should take roughly
3–5 hours and fits in one Kaggle session; it can also be split across sessions.

## Hand-back

The forecasts are about 170 MB, too large for the repository. Zip the folder and share it (Drive), as with the
Round 2 checkpoints:

```bash
cd /kaggle/working/TEA-STGNN && zip -r s7_ablation_results.zip results/s7_ablation
```

The report is then built on the laptop (it needs the Phase A tree forecasts, which are only there):

```bash
.venv-expanded/bin/python src/s7/ablation_report.py   # results/s7_ablation/ablation_{summary,contrasts,per_target}.csv
```
