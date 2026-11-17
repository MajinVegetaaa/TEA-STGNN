"""Audit expanded experiment coverage, selection and preserved artifacts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/expanded'
SETS=['existing','staff','ap_ib','staar','graduation','all']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    nodes=pd.read_csv(ROOT/'data/graph_expanded/nodes.csv',dtype={'campus_id':str})
    selected=pd.read_csv(OUT/'selection_locked.csv')
    final=pd.read_csv(OUT/'feature_selection_locked.csv')
    test=pd.read_csv(OUT/'selected_test_predictions.csv',dtype={'campus_id':str})
    assert len(selected)==60 and len(final)==10
    assert not selected.duplicated(['input_set','target']).any()
    assert not final.target.duplicated().any()
    checks=[]
    def verify_rows(p,year,expected_models):
        assert p.year.eq(year).all()
        assert np.isfinite(p[['y_true','y_pred']].to_numpy()).all()
        assert not p.duplicated(['target','model','seed','campus_id','year']).any()
        for target,g in p.groupby('target'):
            n=nodes[(nodes.year==year)&nodes[target].notna()&nodes[target+'_lag1'].notna()]
            truth=n.set_index('campus_id')[target].sort_index()
            assert g.model.nunique()==(expected_models-1 if target=="risk_index" and expected_models>1 else expected_models)
            for (model,seed),d in g.groupby(['model','seed']):
                actual=d.set_index('campus_id').y_true.sort_index()
                assert actual.index.equals(truth.index), (target,model,seed,'rows')
                np.testing.assert_allclose(actual,truth,rtol=0,atol=1e-12)
            for model,d in g.groupby('model'):
                assert set(d.seed)==set(range(5))
        assert set(p.target)==set(selected.target)
    for s in SETS:
        val=pd.read_csv(OUT/s/'level/predictions.csv',dtype={'campus_id':str})
        assert val.split.eq('val').all()
        verify_rows(val,2023,15)
        scores=val.assign(error=(val.y_pred-val.y_true).abs()).groupby(['target','model','seed']).error.mean().groupby(level=[0,1]).mean()
        for row in selected[selected.input_set==s].itertuples():
            assert np.isclose(scores.loc[(row.target,row.model)],scores.loc[row.target].min())
            assert np.isclose(row.val_mae,scores.loc[(row.target,row.model)])
        p=test[test.input_set==s]
        assert p.split.eq('test').all()
        verify_rows(p,2024,1)
        for target,g in p.groupby('target'):
            model=selected.loc[(selected.input_set==s)&(selected.target==target),'model'].item()
            assert set(g.model)=={model}
        checks.append({'input_set':s,'validation_rows':len(val),'test_rows':len(p),'all_checks_pass':True})
    for row in final.itertuples():
        assert np.isclose(row.val_mae,selected.loc[selected.target==row.target,'val_mae'].min())
        assert ((selected.input_set==row.input_set)&(selected.target==row.target)&(selected.model==row.model)).any()
    frozen=ROOT/'results/frozen/pre_expansion'
    manifest=json.loads((frozen/'manifest.json').read_text())['files']
    preserved=[p for p in manifest if p.startswith(('data/graph/','results/'))]
    for p in preserved:
        assert sha(ROOT/p)==manifest[p]['sha256'], f'Changed frozen output: {p}'
    edges=list((ROOT/'data/graph').glob('edges_*.csv'))
    for p in edges:
        assert sha(p)==sha(ROOT/'data/graph_expanded'/p.name),p.name
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'configs/feature_availability.yaml',ROOT/'configs/split.yaml',ROOT/'data/graph_expanded/nodes.csv',ROOT/'src/baselines/run_experiment.py',ROOT/'src/baselines/run_expansion.py',OUT/'selection_locked.csv',OUT/'feature_selection_locked.csv',OUT/'selected_test_predictions.csv']}
    (OUT/'validation.json').write_text(json.dumps({'passed':True,'input_sets':checks,'preserved_files':len(preserved),'unchanged_edge_files':len(edges),'fingerprints_at_validation':fingerprints},indent=2)+'\n')
    print(f'Passed all six comparisons, 10 targets, 5 seeds; {len(preserved)} frozen artifacts and {len(edges)} edge files unchanged')

if __name__=='__main__':main()
