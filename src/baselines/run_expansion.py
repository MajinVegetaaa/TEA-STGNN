"""Six approved input comparisons, validation selection, then fixed test scoring.

Reuses run_experiment's model families, grids, features and preprocessing. The
first expanded experiment fixes target representation to level. Selection uses
2023 only, includes persistence, and never reads 2024 predictions until locked.
"""
import os
os.environ['TEA_EXPANDED'] = '1'
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import run_experiment as B

ROOT = B.ROOT
SETS = ['existing','staff','ap_ib','staar','graduation','all']
OUT = ROOT/'results/expanded'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache = {}
    original_fit = B.fit_predict
    original_features = B.feature_table
    feature_cache = {}
    def cached_features(nodes,edges,target,edge_set):
        selected_set=B.INPUT_SET
        if target not in feature_cache:
            B.INPUT_SET='existing'
            feature_cache[target]=original_features(nodes,edges,target,edge_set)
            B.INPUT_SET=selected_set
        base,groups=feature_cache[target]
        df=base.copy()
        fg={k:list(v) for k,v in groups.items()}
        prefixes=[p for k,ps in B.ADDITION_GROUPS.items() if selected_set in (k,'all') for p in ps]
        added=[c for c in nodes if any(c.startswith('available_'+p) for p in prefixes)]
        if added:
            df=pd.concat([df,nodes[added]],axis=1)
            fg['ctx']+=added
        return df,fg
    B.feature_table=cached_features
    deterministic_gb_verified=False
    def cached(kind,params,seed,Xtr,ytr,Xte,target):
        nonlocal deterministic_gb_verified
        if kind=='gb' and len(Xtr)<10000:
            if not deterministic_gb_verified:
                a=original_fit(kind,params,0,Xtr,ytr,Xte,target)
                b=original_fit(kind,params,1,Xtr,ytr,Xte,target)
                np.testing.assert_array_equal(a,b)
                deterministic_gb_verified=True
                print('Verified deterministic HGB seeds with no early stopping or binning subsample',flush=True)
            seed=0
        digest = hashlib.sha256(repr((kind,params,seed,target,list(Xtr.columns))).encode())
        for x in (Xtr,ytr,Xte):
            digest.update(pd.util.hash_pandas_object(x,index=True).values.tobytes())
        key = digest.hexdigest()
        if key not in cache:
            cache[key] = original_fit(kind,params,seed,Xtr,ytr,Xte,target)
        return cache[key].copy()
    B.fit_predict = cached
    for input_set in SETS:
        sys.argv = ['run_experiment.py','--mode','level','--input-set',input_set,'--validation-only']
        B.main()
    predictions = pd.concat([pd.read_csv(OUT/s/'level/predictions.csv',dtype={'campus_id':str}).assign(input_set=s) for s in SETS],ignore_index=True)
    assert predictions.split.eq('val').all()
    predictions['error'] = (predictions.y_pred-predictions.y_true).abs()
    scores = predictions.groupby(['input_set','target','model','seed']).error.mean().groupby(level=[0,1,2]).mean().reset_index(name='val_mae')
    selected = scores.loc[scores.groupby(['input_set','target']).val_mae.idxmin()].reset_index(drop=True)
    tuning = {s:pd.read_csv(OUT/s/'level/tuning.csv') for s in SETS}
    params = []
    for row in selected.itertuples():
        t = tuning[row.input_set]
        t = t[(t.target==row.target)&(t.model==row.model)]
        params.append(t.loc[t.val_mae.idxmin(),'params'] if len(t) else '{}')
    selected['params'] = params
    # Persist selection before opening test rows for prediction/scoring.
    selected.to_csv(OUT/'selection_locked.csv',index=False)
    selected.loc[selected.groupby('target').val_mae.idxmin()].to_csv(OUT/'feature_selection_locked.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps({'mode':'level','input_sets':SETS,'split':B.SPLIT,'selection_year':2023,'test_year':2024,'selection_includes_persistence':True},indent=2))
    nodes = pd.read_csv(B.G/'nodes.csv',dtype=B.ID)
    edges = {l:pd.read_csv(B.G/f'edges_{l}.csv',dtype=B.ID) for l in B.LAYERS}
    sets = pd.read_csv(B.G/'academic_edge_sets.csv').set_index('target').edge_set
    outputs=[]
    for row in selected.itertuples():
        B.INPUT_SET = row.input_set
        df,fg = B.feature_table(nodes,edges,row.target,sets.get(row.target))
        eligible = df[df[row.target].notna()&df._raw_lag1.notna()]
        tr=eligible[eligible.year.isin(B.TRAIN)]
        refit=eligible[eligible.year.isin(B.TRAIN+B.VAL)]
        te=eligible[eligible.year.isin(B.TEST)]
        B.PREPROCESS_INDEX=tr.index
        for seed in B.SEEDS:
            if row.model.startswith('persistence'):
                prediction=te._raw_lag1.to_numpy()
            else:
                kind,groups=B.MODELS[row.model]
                cols=B.cols_for(groups,fg)
                prediction=original_fit(kind,ast.literal_eval(row.params),seed,refit[cols],refit._y_fit,te[cols],row.target)
            outputs.append(pd.DataFrame({'input_set':row.input_set,'model':row.model,'seed':seed,'campus_id':te.campus_id,'year':te.year,'target':row.target,'y_true':te[row.target],'y_pred':prediction,'split':'test'}))
        print('Fixed test evaluation:',row.input_set,row.target,row.model,flush=True)
    test=pd.concat(outputs,ignore_index=True)
    test.to_csv(OUT/'selected_test_predictions.csv',index=False)
    summaries=[]
    for (input_set,target),g in test.groupby(['input_set','target']):
        n=nodes[nodes.year.isin(B.TEST)]
        errors=[]
        for seed,d in g.groupby('seed'):
            e=d.y_pred-d.y_true
            errors.append((e.abs().mean(),np.sqrt((e**2).mean()),1-(e**2).sum()/((d.y_true-d.y_true.mean())**2).sum()))
        a=np.array(errors)
        summaries.append({'input_set':input_set,'target':target,'n':g.campus_id.nunique(),'eligible_pct':100*g.campus_id.nunique()/len(n),'mae':a[:,0].mean(),'mae_sd':a[:,0].std(),'rmse':a[:,1].mean(),'r2':a[:,2].mean()})
    pd.DataFrame(summaries).merge(selected,on=['input_set','target'],validate='one_to_one').to_csv(OUT/'comparison.csv',index=False)
    print('Expanded comparisons complete',flush=True)

if __name__=='__main__':
    main()
