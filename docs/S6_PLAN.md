# S6: the single 2024 test

S6 answers one question once: how do the models do on 2024, a year no one has looked at? Every choice is made
before that number exists. After the run nothing is changed, re-tuned or re-run, whatever the result.

## The three rules

1. **Lock everything first.** Which models, each model's hidden size, dropout and number of epochs per target, the
   shrinkage coefficients, the honest-bar picks, the scoring rows, the comparisons. All of it is in
   `configs/s6_lock.json`, committed to `main` before any 2024 value is read.
2. **Retrain on 2018–2023 with the locked settings.** Training labels are 2019–2023 (2018 has no previous year).
   Using 2023 for training is now allowed because no more choices are made with it.
   Input preprocessing (medians, scaling, category lists, edge-distance statistics) stays fitted on 2018–2022, as in
   validation and in the S1c tabular refit. The 2018–2023 inputs are therefore exactly the ones the settings were
   chosen with; the lock stores a hash of them and every S6 part checks it before fitting anything.
3. **Predict 2024 once and report it.** The report script writes the test table; that table goes in the paper as is.

## What is locked, and how each choice was made

| Model | Role | Settings, and where they come from |
|---|---|---|
| Persistence (last year's value) | reference, score 1.00 | none |
| Shrinkage | simple own-history forecast | last year's value + β × (own 3-year average − last year's value); β per target × school-size tercile, fitted on label years 2019–2023. The 30 β values, the tercile cut points and the size used when enrollment is missing are all in the lock and applied as is, never re-fitted (the 2024 bundle has extra all-zero rows for schools that first appear in 2024, which would move a re-fit) |
| Tree | tabular model | Phase A gradient-boosted tree, fixed parameters, S1c inputs, trained on all years before 2024 |
| **Honest bar** | the fair simple benchmark | per target, tree or shrinkage, whichever scored better on 2021–2023 (5 targets each) |
| S1c tabular bar | old bar (0.930 on 2023) | picks made on 2023 among ~90 models; refit on 2018–2023 and its 2024 forecasts written in S1c, before S6. Reported, marked as chosen on 2023 |
| MTGNN | main graph baseline (Wu et al. 2020) | as designed in S4: change from last year, settings chosen on 2023 in S5 Round 2 |
| **V0** (STGNN without graph) | **final model** | Plan A: shrinkage starting point + relative-change target; hidden/dropout/epochs per target chosen on 2023 after training on 2019–2022 |
| **V1_original** (STGNN with graph) | proposed graph model | same as V0, with the planned relation attention |

**Why V0 is the final model.** The final-model rule (lowest score, ties within one seed SD go to the simpler model),
applied to the honest rolling results: Plan A V0 1.014 and Plan A V1 1.016 (mean of 2022 and 2023) are tied, so the
simpler V0 wins. V1 is still run and reported so that V1 − V0 (what the graph adds) is measured on 2024.

**Why Plan A.** It is the best STGNN configuration under honest selection (Step 4: V1 1.108 without it, 1.016 with it).

## What is reported (fixed now)

* Per target and overall: score = MAE / persistence MAE on the same schools for every model (2024 value and its
  2023 value observed, and present in every model's forecasts). Seeded models: mean and SD over seeds 0–4.
* Paired bootstrap (seed-averaged forecasts, same schools, 2000 resamples, seed 0), per target and overall, for:
  V0 vs MTGNN, V0 vs honest bar, V0 vs S1c bar, V1 vs MTGNN, V1 vs honest bar, V1 vs S1c bar, V1 vs V0.
* No other model, setting or comparison is added after 2024 is seen. Extra analyses belong to S7 and are labelled
  as done after the test.

## How to run it

Every command is run from the repository root on `main` at or after the commit that added `configs/s6_lock.json`.

**0. Rehearsal (done 2026-11-26, no 2024 data).** The same code replayed 2023 with the settings chosen on 2022:
V0, V1, shrinkage, tree and honest bar reproduce the Step 4 / Plan A numbers exactly (difference 0.0 on every target,
same schools); the S1c bar gives 0.930 with every target equal to the recorded value; MTGNN gives 1.021 (Round 2 on
the Kaggle GPU: 1.020). Tables in `results/s6_rehearsal/`. The only code that runs for the first time in the real
test is the 2018–2024 bundle loader, and it runs (and would fail) before anything is fitted.

**1. Build the 2018–2024 data (laptop, about 5 minutes).**

```bash
.venv-expanded/bin/python src/s6/s6_data.py build        # writes data/graph_v2/pyg/s6_test/ (refuses to overwrite)
```

**2. Neural models: STGNN V0, V1 and MTGNN (Kaggle GPU or laptop).**

*Kaggle:* upload `data/graph_v2/pyg/s6_test/` as a private dataset (about 95 MB), copy the repository at the lock
commit into `/kaggle/working/TEA-STGNN`, copy the bundle to `data/graph_v2/pyg/s6_test/` inside it, then:

```bash
python src/s6/run.py neural --model v0 --device cuda
python src/s6/run.py neural --model v1 --device cuda
python src/s6/run.py neural --model mtgnn --device cuda
```

Download `results/s6/neural/` (about 20 MB of CSVs) and commit it, or send it back to be committed. The 2024
predictions are the test record and are committed in full.
*Laptop:* the same commands with `--device cpu`.

Each command first checks the 2018–2023 data hash (the tree checks the 2018–2023 CSV source digests). **If the check fails, stop**: nothing has been fitted and no 2024
number has been produced, so the data can be fixed and the command run again. If a command crashes part-way, run it
again: finished targets and seeds are kept and nothing is re-chosen.

**3. Tree and S1c bar (laptop: they need the `data/graph_v2/*.csv` sources and the local S1c prediction files).**

```bash
.venv-expanded/bin/python src/s6/run.py tree
.venv-expanded/bin/python src/s6/run.py s1c
```

**4. The test table (once all four parts are complete).**

```bash
.venv-expanded/bin/python src/s6/report.py                # results/s6/s6_*.csv
```

## Time

| Part | Laptop (CPU), measured in the rehearsal | Kaggle T4 |
|---|---|---|
| Build 2018–2024 data | ~5 min | (built on the laptop) |
| V0 (10 targets × 5 seeds, no tuning) | ~30 s | under a minute |
| V1 | ~6 min | a few minutes |
| MTGNN | ~14 min | a few minutes |
| Tree, S1c extract, report | ~5 min | (laptop) |

The whole test takes under 30 minutes on the laptop, so Kaggle is optional. There is no tuning in S6, so it is much shorter than Round 2.

## Notes for the paper

* The S1c tabular 2024 forecasts were written in S1c (2026-11-21) as part of that step's protocol, before any of
  this; S6 is the first 2024 forecast for the STGNN, MTGNN, shrinkage and tree.
* The validation bundle used for all post-S5 work was rebuilt locally (fingerprint `85da39…`); its targets and
  persistence errors equal the Kaggle bundle's. The lock records the content hash of the exact 2018–2023 data.
* MTGNN's settings were chosen on the Kaggle validation bundle in Round 2; they are applied unchanged.
