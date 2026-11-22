# Expanded-data experiment — 2026-11-18

Completed the approved six comparisons for ten targets. Selection used 2023
only; models were then refit on 2018–2023 and scored once on 2024. Preprocessing
remained fitted on 2018–2022. Seeds are 0–4 and target representation is level.
The validation-selected input/model choices have lower test MAE than the new
existing-input comparator for 2/10 targets, higher for 2/10, and the
same for 6/10. This describes the held-out results, not a basis to
change the locked selections.

## Locked choices and test coverage

Errors are in each target's native units (ACT points, SAT points, percentage
points for rates, or Risk Index units). MAE/RMSE/R² are averaged over five seeds;
MAE SD is the population standard deviation across seeds. Coverage is the share
of the 1,666 campuses present in the 2024 panel (from the fixed
1,670-campus cohort) with both observed target and observed raw prior-year target.
The new graduation model uses publication-filtered histories even though
raw lag 1 determines the common mask and retrospective persistence.

| target | input_set | model | val_mae | n | eligible_pct | mae | mae_sd | rmse | r2 | persistence_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attendance_rate | all | persistence (last year's value) | 1.351 | 1657 | 99.460 | 0.728 | 0.000 | 1.104 | 0.842 | 0.728 |
| dropout_9_12 | all | persistence (last year's value) | 0.500 | 1654 | 99.280 | 0.449 | 0.000 | 0.891 | 0.462 | 0.449 |
| graduation_4yr | all | persistence (last year's value) | 2.199 | 1550 | 93.037 | 2.215 | 0.000 | 3.614 | 0.716 | 2.215 |
| retention_rate | all | persistence (last year's value) | 1.232 | 1660 | 99.640 | 1.071 | 0.000 | 1.800 | 0.872 | 1.071 |
| sat_act_participation | all | persistence (last year's value) | 12.051 | 1604 | 96.279 | 8.667 | 0.000 | 15.490 | 0.618 | 8.667 |
| act_composite | ap_ib | random_forest (+context +graph) | 1.181 | 1079 | 64.766 | 1.148 | 0.002 | 1.558 | 0.763 | 1.276 |
| sat_act_above_criterion | existing | random_forest (+context) | 6.068 | 1547 | 92.857 | 5.889 | 0.011 | 8.511 | 0.788 | 6.751 |
| risk_index | staar | gradient_boosting (+context +district only) | 6.456 | 1546 | 92.797 | 6.714 | 0.000 | 8.704 | 0.783 | 7.672 |
| sat_total | staar | gradient_boosting (+context) | 30.675 | 1260 | 75.630 | 30.284 | 0.000 | 42.063 | 0.839 | 30.842 |
| tsi_both | staar | random_forest (+context +graph) | 12.691 | 1604 | 96.279 | 14.333 | 0.080 | 19.205 | 0.461 | 14.415 |

## All six fixed comparisons

Each cell is test MAE for the model selected within that input set on 2023.
The best test cell is not used to select an input set.
Equal validation minima use the runner's stable table order. In particular,
persistence is identical across input sets and uses no added contextual fields;
an input-set label on a persistence winner does not imply those additions helped.
The graduation target has its own publication-filtered history in every input
set, including existing inputs. The graduation addition supplies those histories
as contextual inputs across targets, rather than removing the new target's own
history from its baseline.

| target | existing | staff | ap_ib | staar | graduation | all |
| --- | --- | --- | --- | --- | --- | --- |
| act_composite | 1.150 | 1.151 | 1.148 | 1.153 | 1.152 | 1.150 |
| attendance_rate | 0.728 | 0.728 | 0.728 | 0.728 | 0.728 | 0.728 |
| dropout_9_12 | 0.449 | 0.449 | 0.449 | 0.449 | 0.449 | 0.449 |
| graduation_4yr | 2.215 | 2.215 | 2.215 | 2.215 | 2.215 | 2.215 |
| retention_rate | 1.071 | 1.071 | 1.071 | 1.071 | 1.071 | 1.071 |
| risk_index | 6.710 | 6.683 | 6.688 | 6.714 | 6.737 | 6.602 |
| sat_act_above_criterion | 5.889 | 5.932 | 5.895 | 6.071 | 5.912 | 5.904 |
| sat_act_participation | 8.667 | 8.667 | 8.667 | 8.667 | 8.667 | 8.667 |
| sat_total | 29.974 | 29.816 | 29.573 | 30.284 | 30.100 | 30.364 |
| tsi_both | 14.415 | 14.415 | 13.552 | 14.333 | 14.415 | 14.415 |

## Frozen original-eight comparison

Exactly the same campus/year rows and truths are used for each original target.
These MAEs use the prediction averaged across seeds, so they can differ from the
mean of per-seed MAEs above. Differences are new minus frozen MAE; negative is
better. The 95% intervals use 2,000 paired campus bootstrap samples with seed 0.
They describe sampling variation conditional on the fitted models and cohort;
they are not multiple-comparison-adjusted or proof of causal benefit.

| target | input_set | n_identical_rows | old_mae | new_mae | mae_difference | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| act_composite | ap_ib | 1079 | 1.146 | 1.146 | 0.000 | -0.006 | 0.006 |
| attendance_rate | all | 1657 | 0.824 | 0.728 | -0.095 | -0.125 | -0.066 |
| dropout_9_12 | all | 1654 | 0.419 | 0.449 | 0.029 | 0.011 | 0.048 |
| retention_rate | all | 1660 | 1.038 | 1.071 | 0.033 | -0.012 | 0.072 |
| sat_act_above_criterion | existing | 1547 | 5.894 | 5.876 | -0.019 | -0.054 | 0.016 |
| sat_total | staar | 1260 | 30.227 | 30.284 | 0.057 | -0.689 | 0.795 |
| tsi_both | staar | 1604 | 14.036 | 14.310 | 0.274 | 0.109 | 0.441 |
| risk_index | staar | 1546 | 6.651 | 6.714 | 0.063 | -0.078 | 0.198 |

Frozen models could select level/relative/zscore representations and used the
previous preprocessing. The six new runs all use level and training-only median
imputation plus missing flags. The frozen selection table records fitted-model
winners separately from persistence; the new selection admits persistence as a
winner. Therefore within-expansion comparisons isolate
input additions more closely than the frozen comparison, which also changes
preprocessing and the representation-selection protocol.

## Validation and interpretation

- Master/source gates: 179 raw checks, 115 processed checks, unchanged 59 original
  master columns and 261,780 independently checked source cells.
- Graph: 86 checks; all 7 edge files identical to the
  originals. PyG: 63 checks, seven snapshots, 1,670 fixed campuses, ten targets,
  430 features. Unit suite: 16 tests.
- Experiment audit: every model/seed has the exact observed-target/observed-lag
  rows, finite predictions, matching truths and validation-only tuning outputs;
  all selected test model names and validation minima agree with the lock files.
  All 56 original graph/results files retain their checksums.
- Ten-epoch CPU GNN smoke tests passed for graduation and participation, with
  validation MAEs 2.7975 and 13.7579 respectively. These smoke runs do not claim a
  GNN accuracy improvement and do not inspect the test year.
- Added inputs use the assumed August 1 Y−1 cutoff with source-year publication
  checks. Existing inputs, graph structure and persistence remain retrospective;
  downloaded data revisions are not archived historical vintages. See
  [availability policy](../../docs/DATA.md#history-features-model-inputs). Static metadata
  and cohort-selection limitations from the original pipeline also remain.
- Early years have limited added history. Training-empty columns are dropped;
  STAAR 2020 remains missing and 2023 assessment redesign is not harmonized away.
  Finance stays staged, AP/IB stay separate, and Risk Index weights stay fixed.

## Files and reproduction

`comparison.csv` contains all 60 fixed test summaries; `final_selection_report.csv`
contains the ten validation-locked choices. `selection_locked.csv` and
`feature_selection_locked.csv` precede test scoring. `frozen_comparison.csv`
contains all 48 paired comparisons. `validation.json` records assertions and
artifact fingerprints. Each input-set folder contains validation predictions
and tuning grids; `selected_test_predictions.csv` contains only locked test runs.
Large predictions, raw/processed data, graph tensors and the frozen snapshot are
local and gitignored: preserve them separately when sharing the project.

Run the commands in [DATA.md](../../docs/DATA.md). The experiment
reuses existing model families/grids, caches identical feature tables/predictions,
and reuses deterministic HGB seed outputs only where early stopping and binning
subsampling cannot apply, after checking seed equivalence. No stochastic forest
or XGBoost seed is omitted.
