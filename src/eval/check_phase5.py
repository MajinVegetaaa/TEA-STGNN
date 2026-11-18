"""Independently verify local or Colab Phase 5 outputs against Phase 3."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
KEY = ["target", "graph", "seed"]
ID = {"campus_id": str}


def check(directory, snapshot_path, phase3, require_cuda=False):
    protocol = json.loads((directory / "protocol.json").read_text())
    if require_cuda and protocol["device"] != "cuda":
        raise ValueError("Expected a CUDA run")
    with snapshot_path.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == protocol["snapshot_sha256"]
    data = torch.load(snapshot_path, weights_only=False, map_location="cpu")
    assert data["targets"] == protocol["targets"] and data["split"] == protocol["split"]
    test = data["snapshots"][-1]
    selected = pd.DataFrame(json.loads((directory / "selection_locked.json").read_text()))
    grid = pd.read_csv(directory / "validation_grid.csv")
    pred = pd.read_csv(directory / "predictions.csv", dtype={**ID, "y_pred": np.float32})
    ref = pd.read_csv(phase3 / "selected_test_predictions.csv", dtype=ID)
    ref = ref.merge(pd.read_csv(phase3 / "feature_selection_locked.csv")[["target","input_set"]], validate="many_to_one")
    assert pred.year.eq(data["years"][-1]).all() and pred.split.eq("test").all()
    assert set(pred.model) == {protocol["model"]}
    assert not pred.duplicated(KEY + ["campus_id"]).any()
    scores = []
    for j, target in enumerate(data["targets"]):
        g = grid[grid.target == target]
        eligible = {"combined","district","region","geographic","demographic"}
        if target in data["academic_relation"]:
            eligible.add(data["academic_relation"][target])
        assert set(g.graph) == eligible and len(g) == len(eligible)
        single = g[g.graph != "combined"].sort_values("val_mae", kind="stable").iloc[0]
        chosen = selected[selected.target == target]
        assert set(chosen.graph) == {"combined", single.graph} and len(chosen) == 2
        mask = test["campus"].eval_mask[:, j]
        ids = np.array(data["campus_ids"])[mask.numpy()]
        previous = test["campus"].y_lag1[mask, j].numpy()
        for row in chosen.itertuples():
            valrow = g[g.graph == row.graph].iloc[0]
            assert row.epochs == valrow.epochs and np.isclose(row.val_mae, valrow.val_mae)
            for seed in data["split"]["seeds"]:
                rows = pred[(pred.target == target) & (pred.graph == row.graph) & (pred.seed == seed)].set_index("campus_id")
                benchmark = ref[(ref.target == target) & (ref.seed == seed)].set_index("campus_id")
                assert set(rows.index) == set(ids) == set(benchmark.index)
                truth = benchmark.loc[ids].y_true.to_numpy()
                np.testing.assert_allclose(rows.loc[ids].y_true, truth, rtol=0, atol=1e-10)
                error = rows.loc[ids].y_pred.to_numpy() - truth
                assert np.isfinite(error).all()
                mae = np.abs(error).mean()
                scores.append(dict(target=target,graph=row.graph,seed=seed,n=len(ids),mae=mae,
                    rmse=np.mean(error**2)**0.5,r2=1-np.sum(error**2)/np.sum((truth-truth.mean())**2),
                    coverage_pct=100*len(ids)/int(test["campus"].present.sum()),
                    delta_vs_phase3=mae-np.abs(benchmark.loc[ids].y_pred-truth).mean(),
                    delta_vs_persistence=mae-np.abs(previous-truth).mean()))
    calculated = pd.DataFrame(scores).set_index(KEY).sort_index()
    stored = pd.read_csv(directory / "per_seed.csv").set_index(KEY).sort_index()
    assert calculated.index.equals(stored.index) and len(pred) == calculated.n.sum()
    np.testing.assert_allclose(stored[calculated.columns], calculated, atol=1e-8)
    mean = calculated.groupby(level=["target","graph"]).mean()
    mean["mae_sd"] = calculated.mae.groupby(level=["target","graph"]).std(ddof=0)
    comparison = pd.read_csv(directory / "comparison.csv").set_index(["target","graph"]).sort_index()
    assert mean.index.equals(comparison.index)
    np.testing.assert_allclose(comparison[mean.columns], mean, atol=1e-8)
    print(f"Verified {protocol['model']} on {protocol['device']}: {len(pred):,} predictions, {len(stored)} seed comparisons",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--require-cuda",action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        check(directory, ROOT/"data/graph_expanded/pyg/snapshots.pt", ROOT/"results/expanded", args.require_cuda)
