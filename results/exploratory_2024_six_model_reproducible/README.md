# Exploratory six-model 2024 results

Completed exploratory evaluation from the supplied Kaggle result bundle. The full source bundle includes prediction CSVs and fit-cache data; those are intentionally not committed here, following the existing instruction to keep predictions out of GitHub. This directory keeps the compact metrics and run provenance.

Models: V0 Temporal, V1 Original, V1 Shared-4 single-task (`v1s4`), V2 Original, MTGNN, and locked Tree. The Tree is deterministic (`random_state=0`) and was fitted once; its results occupy five comparison seed slots.

Primary aggregate: mean per-seed normalized MAE over the ten targets; lower is better. See `summary_by_model.csv`, `summary_by_target.csv`, and `per_seed_metrics.csv`. Paired bootstrap tables compare seed-averaged predictions and therefore answer a different question than the per-seed aggregate.
