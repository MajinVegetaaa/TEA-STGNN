"""Write the repository handoff from locked, validated expanded results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/expanded'

def table(df):
    def value(v):
        return f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(value(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    audit=json.loads((OUT/'validation.json').read_text())
    assert audit['passed']
    comparison=pd.read_csv(OUT/'comparison.csv')
    locked=pd.read_csv(OUT/'feature_selection_locked.csv')
    chosen=comparison.merge(locked[['target','input_set']],on=['target','input_set'],validate='one_to_one')
    nodes=pd.read_csv(ROOT/'data/graph_expanded/nodes.csv')
    persistence={}
    for target in chosen.target:
        n=nodes[(nodes.year==2024)&nodes[target].notna()&nodes[target+'_lag1'].notna()]
        persistence[target]=(n[target]-n[target+'_lag1']).abs().mean()
    chosen['persistence_mae']=chosen.target.map(persistence)
    base=comparison[comparison.input_set=='existing'].set_index('target').mae
    chosen['existing_mae']=chosen.target.map(base)
    chosen['delta_vs_existing']=chosen.mae-chosen.existing_mae
    chosen.to_csv(OUT/'final_selection_report.csv',index=False)
    paired=pd.read_csv(OUT/'frozen_comparison.csv').merge(locked[['target','input_set']],on=['target','input_set'],validate='one_to_one')
    wins=int((chosen.delta_vs_existing < -1e-10).sum())
    losses=int((chosen.delta_vs_existing > 1e-10).sum())
    text=f'''# Expanded-data experiment — 2026-11-18

Completed the approved six comparisons for ten targets. Selection used 2023
only; models were then refit on 2018–2023 and scored once on 2024. Preprocessing
remained fitted on 2018–2022. Seeds are 0–4 and target representation is level.
The validation-selected input/model choices have lower test MAE than the new
existing-input comparator for {wins}/10 targets, higher for {losses}/10, and the
same for {10-wins-losses}/10. This describes the held-out results, not a basis to
change the locked selections.

## Locked choices and test coverage

Errors are in each target's native units (ACT points, SAT points, percentage
points for rates, or Risk Index units). MAE/RMSE/R² are averaged over five seeds;
MAE SD is the population standard deviation across seeds. Coverage is the share
of the 1,670-campus cohort with both observed target and observed raw prior-year
target. The new graduation model uses publication-filtered histories even though
raw lag 1 determines the common mask and retrospective persistence.

{table(chosen[['target','input_set','model','val_mae','n','eligible_pct','mae','mae_sd','rmse','r2','persistence_mae']])}

## All six fixed comparisons

Each cell is test MAE for the model selected within that input set on 2023.
The best test cell is not used to select an input set.

{table(comparison.pivot(index='target',columns='input_set',values='mae')[['existing','staff','ap_ib','staar','graduation','all']].reset_index())}

## Frozen original-eight comparison

Exactly the same campus/year rows and truths are used for each original target.
These MAEs use the prediction averaged across seeds, so they can differ from the
mean of per-seed MAEs above. Differences are new minus frozen MAE; negative is
better. The 95% intervals use 2,000 paired campus bootstrap samples with seed 0.
They describe sampling variation conditional on the fitted models and cohort;
they are not multiple-comparison-adjusted or proof of causal benefit.

{table(paired[['target','input_set','n_identical_rows','old_mae','new_mae','mae_difference','ci_low','ci_high']])}

Frozen models could select level/relative/zscore representations and used the
previous preprocessing. The six new runs all use level and training-only median
imputation plus missing flags. Therefore within-expansion comparisons isolate
input additions more closely than the frozen comparison, which also changes
preprocessing and the representation-selection protocol.

## Validation and interpretation

- Master/source gates: 179 raw checks, 115 processed checks, unchanged 59 original
  master columns and 261,780 independently checked source cells.
- Graph: 86 checks; all {audit['unchanged_edge_files']} edge files identical to the
  originals. PyG: 63 checks, seven snapshots, 1,670 fixed campuses, ten targets,
  430 features. Unit suite: 16 tests.
- Experiment audit: every model/seed has the exact observed-target/observed-lag
  rows, finite predictions, matching truths and validation-only tuning outputs;
  all selected test model names and validation minima agree with the lock files.
  All {audit['preserved_files']} original graph/results files retain their checksums.
- Ten-epoch CPU GNN smoke tests passed for graduation and participation, with
  validation MAEs 2.7975 and 13.7579 respectively. These smoke runs do not claim a
  GNN accuracy improvement and do not inspect the test year.
- Added inputs use the assumed August 1 Y−1 cutoff with source-year publication
  checks. Existing inputs, graph structure and persistence remain retrospective;
  downloaded data revisions are not archived historical vintages. See
  [availability policy](../../configs/feature_availability.md). Static metadata
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

Run the commands in [DATA_PIPELINE.md](../../DATA_PIPELINE.md). The experiment
reuses existing model families/grids, caches identical feature tables/predictions,
and reuses deterministic HGB seed outputs only where early stopping and binning
subsampling cannot apply, after checking seed equivalence. No stochastic forest
or XGBoost seed is omitted.
'''
    (OUT/'README.md').write_text(text)
    print(chosen[['target','input_set','mae','existing_mae','persistence_mae','n']].to_string(index=False))

if __name__=='__main__':main()
