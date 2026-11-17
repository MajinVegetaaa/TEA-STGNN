"""Independent checks for publication-filtered calendar histories and model masks."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]

class ExpandedTests(unittest.TestCase):
    def test_all_available_inputs_against_calendar_and_release_dates(self):
        m=pd.read_csv(ROOT/'data/processed/master_school_year.csv',dtype={'campus_id':str}).set_index(['campus_id','year'])
        f=pd.read_csv(ROOT/'data/processed/expanded_temporal_features.csv',dtype={'campus_id':str}).set_index(['campus_id','year'])
        config=yaml.safe_load((ROOT/'configs/feature_availability.yaml').read_text())
        sys.path.insert(0,str(ROOT/'src/data'))
        from new_sources import FEATURE_GROUPS
        for group,fields in FEATURE_GROUPS.items():
            minimum=config['groups'][group]['minimum_lag']
            for field in fields:
                for k in range(minimum,minimum+3):
                    index=pd.MultiIndex.from_arrays([f.index.get_level_values(0),f.index.get_level_values(1)-k])
                    expected=m[field].reindex(index).to_numpy()
                    for i,(_,year) in enumerate(f.index):
                        release=config['publication_bounds'][group].get(year-k)
                        if release is None or release>f'{year-1}-08-01':expected[i]=np.nan
                    np.testing.assert_allclose(f[f'available_{field}_lag{k}'],expected,equal_nan=True)
                a,b,c=[f[f'available_{field}_lag{k}'] for k in range(minimum,minimum+3)]
                np.testing.assert_allclose(f[f'available_{field}_ma3'],(a+b+c)/3,equal_nan=True)
                np.testing.assert_allclose(f[f'available_{field}_trend3'],((a-c)/2).where(b.notna()),equal_nan=True)

    def test_original_temporal_values_preserved(self):
        old=pd.read_csv(ROOT/'data/processed/temporal_features.csv')
        new=pd.read_csv(ROOT/'data/processed/expanded_temporal_features.csv')
        pd.testing.assert_frame_equal(old,new[old.columns])

class PreprocessingTests(unittest.TestCase):
    def test_baseline_refit_uses_original_train_medians(self):
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'src/baselines'))
        import run_experiment as baseline
        class Capture:
            def fit(self,x,y):
                self.train=x.copy()
                return self
            def predict(self,x):
                self.test=x.copy()
                return np.zeros(len(x))
        model=Capture()
        # Row 2 is a validation row in the final refit. Its large value must
        # neither influence the median nor rescue a training-empty column.
        train=pd.DataFrame({'value':[2.,4.,10000.], 'empty':[np.nan,np.nan,9.]})
        test=pd.DataFrame({'value':[np.nan], 'empty':[5.]},index=[3])
        with patch.object(baseline,'EXPANDED',True), patch.object(baseline,'PREPROCESS_INDEX',pd.Index([0,1]),create=True), patch.object(baseline,'make',return_value=model):
            baseline.fit_predict('rf',{},0,train,pd.Series([1.,2.,3.]),test,'example')
        self.assertEqual(model.test.loc[3,'value'],3.)
        self.assertEqual(model.test.loc[3,'value_missing'],1.)
        self.assertNotIn('empty',model.train)

    def test_pyg_train_scaler_ignores_validation_and_test_values(self):
        import importlib.util
        if importlib.util.find_spec('torch_geometric') is None:
            self.skipTest('Run with .venv-expanded/bin/python for PyG checks')
        sys.path.insert(0,str(ROOT/'src/graphs'))
        import to_pyg as converter
        nodes=pd.read_csv(ROOT/'data/graph_expanded/nodes.csv',dtype={'campus_id':str})
        catalog=pd.read_csv(ROOT/'data/graph_expanded/node_feature_catalog.csv')
        cols=converter.input_columns(catalog)
        _,before=converter.build_features(nodes,cols)
        changed=nodes.copy()
        changed.loc[changed.year>=2023,cols]=999999
        _,after=converter.build_features(changed,cols)
        self.assertEqual(before,after)

    def test_fixed_campus_order_and_new_target_masks(self):
        import importlib.util
        if importlib.util.find_spec('torch_geometric') is None:
            self.skipTest('Run with .venv-expanded/bin/python for PyG checks')
        import torch
        data=torch.load(ROOT/'data/graph_expanded/pyg/snapshots.pt',weights_only=False)
        frozen=torch.load(ROOT/'results/frozen/pre_expansion/data/graph/pyg/snapshots.pt',weights_only=False)
        self.assertEqual(data['campus_ids'],frozen['campus_ids'])
        self.assertEqual(len(data['targets']),10)
        nodes=pd.read_csv(ROOT/'data/graph_expanded/nodes.csv',dtype={'campus_id':str})
        for snapshot in data['snapshots']:
            for target in ['sat_act_participation','graduation_4yr']:
                expected=set(nodes.loc[(nodes.year==snapshot.year)&nodes[target].notna()&nodes[target+'_lag1'].notna(),'campus_id'])
                mask=snapshot['campus'].eval_mask[:,data['targets'].index(target)]
                actual={data['campus_ids'][i] for i in mask.nonzero().flatten().tolist()}
                self.assertEqual(expected,actual)

if __name__=='__main__':unittest.main()
