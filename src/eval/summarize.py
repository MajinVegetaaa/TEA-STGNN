"""Numbers to beat: pick each target's model on the 2023 validation year, then score it once on 2024.

For every target, candidates are every (target mode, model) pair in results/{level,relative,zscore}. The pick is
the lowest seed-averaged validation MAE among non-persistence models; persistence is the reference. The test
year never takes part in the choice. Writes (results/):
  selection.csv   per target: persistence and picked model with val/test MAE, R2, paired bootstrap vs persistence
  breakdowns.csv  test MAE of persistence vs the picked model by urbanicity, ESC region and school-size tertile
Run from the repo root (after run_experiment.py for each mode):  python3 src/eval/summarize.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate import PERSIST, metrics, paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
MODES = ["level", "relative", "zscore"]


def main() -> None:
    p = pd.concat([pd.read_csv(ROOT / "results" / m / "predictions.csv", dtype={"campus_id": str}).assign(mode=m)
                   for m in MODES], ignore_index=True)
    nodes = pd.read_csv(ROOT / "data" / "graph" / "nodes.csv", dtype={"campus_id": str, "esc_region": str},
                        usecols=["campus_id", "year", "urbanicity", "esc_region", "enrollment_lag1"])
    per_seed = p.groupby(["mode", "model", "target", "split", "seed"]).apply(metrics, include_groups=False).reset_index()
    m = per_seed.groupby(["mode", "model", "target", "split"])[["mae", "rmse", "r2", "school_mae"]].mean().reset_index()

    sel_rows, break_rows = [], []
    for t in m.target.unique():
        val = m[(m.target == t) & (m.split == "val") & (m.model != PERSIST)]
        best = val.loc[val.mae.idxmin()]
        mode, model = best["mode"], best["model"]
        tt = p[(p.target == t) & (p.split == "test") & (p["mode"] == mode)]
        pers, pick = tt[tt.model == PERSIST], tt[tt.model == model]

        def row(name, split):
            return m[(m["mode"] == mode) & (m.model == name) & (m.target == t) & (m.split == split)].iloc[0]
        pv, pt, mt = row(PERSIST, "val"), row(PERSIST, "test"), row(model, "test")
        bs = paired_bootstrap(pers, pick)
        sel_rows.append({"target": t, "mode": mode, "model": model, "n_test": bs["n"],
                         "val_mae_persistence": pv.mae, "val_mae_model": best.mae,
                         "test_mae_persistence": pt.mae, "test_mae_model": mt.mae,
                         "test_r2_persistence": pt.r2, "test_r2_model": mt.r2,
                         "test_rmse_persistence": pt.rmse, "test_rmse_model": mt.rmse,
                         "pct_change_vs_persistence": bs["pct_change"], "ci_low": bs["ci_low"],
                         "ci_high": bs["ci_high"], "beats_persistence": bool(bs["ci_high"] < 0)})

        # breakdowns: seed-averaged predictions, grouped by the campus's own attributes in the test year
        j = (pers.groupby("campus_id").agg(year=("year", "first"), y=("y_true", "first"), p_pers=("y_pred", "mean"))
             .join(pick.groupby("campus_id").y_pred.mean().rename("p_model")).reset_index()
             .merge(nodes, on=["campus_id", "year"], how="left"))
        j["size"] = pd.qcut(j.enrollment_lag1, 3, labels=["small", "medium", "large"])
        for dim in ("urbanicity", "esc_region", "size"):
            for grp, d in j.groupby(dim, observed=True):
                e0, e1 = (d.p_pers - d.y).abs(), (d.p_model - d.y).abs()
                break_rows.append({"target": t, "model": model, "dimension": dim, "group": grp, "n": len(d),
                                   "mae_persistence": e0.mean(), "mae_model": e1.mean(),
                                   "pct_change": 100 * (e1.mean() / e0.mean() - 1)})

    sel, brk = pd.DataFrame(sel_rows), pd.DataFrame(break_rows)
    sel.to_csv(ROOT / "results" / "selection.csv", index=False)
    brk.to_csv(ROOT / "results" / "breakdowns.csv", index=False)
    pd.set_option("display.width", 250)
    print(sel.round(3).to_string())
    print(brk.round(3).to_string())


if __name__ == "__main__":
    main()
