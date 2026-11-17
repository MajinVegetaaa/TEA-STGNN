"""Prepare raw calendar history and separately publication-filtered new inputs.

Uses the existing builder's full-grid feature function after its master gate.
Does not rebuild or change the validated master. Original temporal columns stay
unchanged. Raw history is retained for audit/evaluation; only available_* new
columns may enter models.
"""
from pathlib import Path
import json
import pandas as pd
import yaml
from build_master import temporal_features, YEARS
from new_sources import FEATURE_GROUPS, FEATURES
from validate_additions import require_master_gate, sha

ROOT = Path(__file__).resolve().parents[2]

def prepare():
    require_master_gate()
    p = ROOT / 'data/processed'
    m = pd.read_csv(p/'master_school_year.csv', dtype={'campus_id': str})
    old = pd.read_csv(p/'temporal_features.csv', dtype={'campus_id': str})
    config = yaml.safe_load((ROOT/'configs/feature_availability.yaml').read_text())
    raw = temporal_features(m, FEATURES)
    grid = pd.MultiIndex.from_product([sorted(m.campus_id.unique()), YEARS], names=['campus_id','year'])
    available = {}
    for group, fields in FEATURE_GROUPS.items():
        minimum = config['groups'][group]['minimum_lag']
        bounds = config['publication_bounds'][group]
        for field in fields:
            w = m.set_index(['campus_id','year'])[field].reindex(grid).unstack('year')
            lag = {}
            for k in range(minimum, minimum+3):
                value = w.shift(k, axis=1)
                for year in YEARS:
                    released = bounds.get(year-k)
                    if released is None or released > f'{year-1}-{config["cutoff_month_day"]}':
                        value[year] = float('nan')
                lag[k] = value
                available[f'available_{field}_lag{k}'] = value
            a,b,c = [lag[k] for k in range(minimum, minimum+3)]
            available[f'available_{field}_change'] = a-b
            available[f'available_{field}_ma3'] = (a+b+c)/3
            available[f'available_{field}_trend3'] = ((a-c)/2).where(b.notna())
    a = pd.concat({k:w.stack(future_stack=True) for k,w in available.items()},axis=1).reset_index()
    out = raw.merge(a,on=['campus_id','year'],validate='one_to_one',how='left')
    pd.testing.assert_frame_equal(out[old.columns],old,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    out[old.columns] = old
    destination = p/'expanded_temporal_features.csv'
    out.to_csv(destination,index=False)
    (p/'reports/expanded_feature_receipt.json').write_text(json.dumps({'master_sha256':sha(p/'master_school_year.csv'), 'availability_sha256':sha(ROOT/'configs/feature_availability.yaml'), 'rows':len(out),'columns':len(out.columns),'eligible_input_columns':list(available)},indent=2))
    print(f'Wrote {destination}: {out.shape}; {len(available)} publication-filtered inputs')

if __name__ == '__main__':
    prepare()
