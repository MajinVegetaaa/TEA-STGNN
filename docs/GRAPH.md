# Graph — nodes, edges, weights and checks

```bash
make graph      # TEA_GRAPH=v2: build_graph_data.py -> validate_graph.py (86/86) -> compare_v1_v2.py
make snapshot   # graph v2 -> PyG snapshot (data/graph_v2/pyg/snapshots.pt), uses .venv-expanded
```

## Three graphs; one is current

| Folder | Name | Used by | Status |
|---|---|---|---|
| `data/graph_v2/` | **v2** | S1c onwards (all new work) | **current** |
| `data/graph_expanded/` | v1 | Phase 5 baselines | frozen (its hash is checked by the Phase 5 results) |
| `data/graph/` | original | Phase 3 | frozen |

`TEA_GRAPH=v2|expanded` selects the graph for every script (unset = original; `TEA_EXPANDED=1` = expanded).
Never run `validate_graph.py` without `TEA_GRAPH`/`TEA_EXPANDED`: it would rewrite a frozen report (restore it from
`results/frozen/pre_expansion/` if that happens).

## Nodes

One node per school per year (11,546; 1,619–1,668 per year), in a fixed school order so node i is the same school
every year. `node_feature_catalog.csv` labels every column:
- **inputs** — history features (years < t only), `covid_affected_year`, static school information;
- **never inputs for year t** — anything observed in year t (outcomes, suppression flags, same-year demographics).

`validate_graph.py` checks these labels; `enrollment_oct2025` is excluded (measured after the study period).

## Edges (rebuilt every year)

| # | Relation | Rule | v2 weight | Edges in 2024 |
|---|---|---|---|---|
| 1 | District | every pair of schools in the same district | 1 | 4,107 (902 schools have no district neighbour) |
| 2 | Geographic | 5 nearest within 100 km (virtual campuses excluded) | exp(−km² / (sᵢ·sⱼ)), sᵢ = km to school i's 5th-nearest school, floor 1 km | 5,209 |
| 3 | Academic | 10 most similar on the t−1 profile (≥4 shared indicators); **7 sets**, each leaving out the predicted target's family | Gaussian, one σ per set from the training years | ~75,700 (all sets) |
| 4 | Demographic | 10 most similar on t−1 student make-up (12 features, ≥6 shared) | Gaussian, one σ from the training years | 11,837 |
| 5 | Transition | feeder middle school → high school | — | **not built**: TEA publishes no feeder data |
| 6 | ESC region | every pair of schools in the same region | 1 | ~98,900 |
| 7 | Temporal | same school, t−1 → t | 1 | implicit in the fixed node order |

Similarity distance = root-mean-square difference of z-scores (computed on year t−1 only). Settings:
`configs/graph.yaml` (frozen; pending professor review) plus `configs/graph_v2.yaml`; the training-year kernel
widths are in `data/graph_v2/graph_v2_parameters.json`.

**Why v2 (decision D4).** Same edges as v1, different weights: with v1's fixed 20 km scale, rural and town schools
got about half the geographic weight of city schools (0.34 vs 0.74); v2 gives every locale 0.47–0.50. v1 re-fitted
σ every year, so a weight meant different things in different years; v2's single σ makes years comparable.
Settings were frozen rather than tuned on the single validation year; a robustness check over k is planned.

## PyG snapshot (what neural models read)

One `HeteroData` per year: 1,670 schools × the feature count recorded in the snapshot (history, static, missing-value flags; medians and scaling
from training years only; categorical vocabularies also fit on training years, with unseen later categories all-zero;
finance on a signed log scale), targets with `y_mask` / `eval_mask`, one relation per
edge type (academic: one per set), `edge_weight`, and in v2 `edge_attr = [weight, standardised raw distance]`
for attention models. 63 self-checks, including that the scored rows equal the tabular baselines' rows.

## Leakage safeguards

| Risk | Safeguard |
|---|---|
| Year-t outcomes in features | history uses years < t only; tested |
| Year-t outcomes in similarity edges | profiles from t−1 only; the target's family is dropped from its edge set; shuffling year-t values leaves year-t edges unchanged (tested) |
| Normalisation | statistics from t−1 (edges) or training years (snapshot) only |
| 2025 metadata | structural only (district, region, coordinates); acknowledged as a limitation |
