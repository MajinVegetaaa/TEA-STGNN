"""Compare fixed expanded selections to frozen originals on identical campus rows."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/expanded'

def main():
    p=pd.read_csv(OUT/'selected_test_predictions.csv',dtype={'campus_id':str})
    oldroot=ROOT/'results/frozen/pre_expansion/results'
    selection=pd.read_csv(oldroot/'selection.csv')
    older={mode:pd.read_csv(oldroot/mode/'predictions.csv',dtype={'campus_id':str}) for mode in selection['mode'].unique()}
    rows=[]
    for chosen in selection.itertuples():
        old=older[chosen.mode]
        old=old[(old.target==chosen.target)&(old.model==chosen.model)&(old.split=='test')]
        old=old.groupby(['campus_id','year']).agg(y=('y_true','first'),old=('y_pred','mean'))
        for variant,g in p[p.target==chosen.target].groupby('input_set'):
            new=g.groupby(['campus_id','year']).agg(y_new=('y_true','first'),new=('y_pred','mean'))
            assert set(old.index)==set(new.index), f'Coverage differs: {chosen.target}/{variant}'
            j=old.join(new,validate='one_to_one')
            assert np.allclose(j.y,j.y_new)
            delta=(j.new-j.y).abs()-(j.old-j.y).abs()
            rng=np.random.default_rng(0)
            boot=delta.to_numpy()[rng.integers(0,len(j),size=(2000,len(j)))].mean(axis=1)
            rows.append({'target':chosen.target,'input_set':variant,'n_identical_rows':len(j),'old_model':chosen.model,'old_mode':chosen.mode,'old_mae':(j.old-j.y).abs().mean(),'new_mae':(j.new-j.y).abs().mean(),'mae_difference':delta.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975)})
    pd.DataFrame(rows).to_csv(OUT/'frozen_comparison.csv',index=False)
    print(f'Wrote {len(rows)} paired comparisons')

if __name__=='__main__':
    main()
