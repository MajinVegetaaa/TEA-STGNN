"""Score results/predictions.csv (hand-off 3 format) the same way for every model.

Usage: python3 src/eval/evaluate.py results/relative   (a folder holding predictions.csv)
Outputs <folder>/metrics.csv (per model, target, split; mean and sd over seeds), results/comparisons.csv
(paired bootstrap of test MAE differences on identical schools) and results/figures/*.png.
Run from the repo root:  python3 src/eval/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "results"
FIG = RES / "figures"
PERSIST = "persistence (last year's value)"
NO_GRAPH = "gradient_boosting (+context)"
GRAPH = "gradient_boosting (+context +graph)"
LABEL = {"act_composite": "ACT composite", "sat_total": "SAT total", "sat_act_above_criterion": "College-ready %",
         "dropout_9_12": "Dropout rate 9-12", "attendance_rate": "Attendance rate"}


def metrics(g: pd.DataFrame) -> pd.Series:
    e = g.y_pred - g.y_true
    ss_res, ss_tot = (e ** 2).sum(), ((g.y_true - g.y_true.mean()) ** 2).sum()
    return pd.Series({"mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean()), "r2": 1 - ss_res / ss_tot, "n": len(g)})


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """MAE(b) - MAE(a) on the same schools, seed-averaged predictions; negative = b is better."""
    pa = a.groupby("campus_id").agg(y=("y_true", "first"), p=("y_pred", "mean"))
    pb = b.groupby("campus_id").agg(p=("y_pred", "mean"))
    j = pa.join(pb, rsuffix="_b", how="inner")
    da, db = (j.p - j.y).abs().to_numpy(), (j.p_b - j.y).abs().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(j), size=(n_boot, len(j)))
    diffs = db[idx].mean(1) - da[idx].mean(1)
    return {"n": len(j), "mae_a": da.mean(), "mae_b": db.mean(), "diff": db.mean() - da.mean(),
            "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975),
            "pct_change": 100 * (db.mean() - da.mean()) / da.mean(), "p_b_not_better": (diffs >= 0).mean()}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(RES / "predictions.csv", dtype={"campus_id": str})
    # every model must be scored on identical rows
    keys = p.groupby(["target", "split", "model"]).apply(lambda g: frozenset(g.campus_id), include_groups=False)
    for (t, s), k in keys.groupby(level=[0, 1]):
        assert k.nunique() == 1, f"{t}/{s}: models scored on different rows"

    per_seed = p.groupby(["model", "target", "split", "seed"]).apply(metrics, include_groups=False).reset_index()
    m = per_seed.groupby(["model", "target", "split"]).agg(
        mae=("mae", "mean"), mae_sd=("mae", "std"), rmse=("rmse", "mean"), r2=("r2", "mean"),
        r2_sd=("r2", "std"), n=("n", "first")).reset_index()
    m.to_csv(RES / "metrics.csv", index=False)

    comps = []
    test = p[p.split == "test"]
    for t in test.target.unique():
        tt = test[test.target == t]
        for a, b in [(PERSIST, GRAPH), (NO_GRAPH, GRAPH), (PERSIST, "random_forest (own lags)"),
                     ("random_forest (+context)", "random_forest (+context +graph)")]:
            r = paired_bootstrap(tt[tt.model == a], tt[tt.model == b])
            comps.append({"target": t, "baseline": a, "model": b, **r})
    pd.DataFrame(comps).to_csv(RES / "comparisons.csv", index=False)

    # figure: test MAE relative to persistence, per target
    order = [PERSIST, "linear_regression (own lags)", "knn (own lags)", "random_forest (own lags)",
             "random_forest (+context)", NO_GRAPH, "random_forest (+context +graph)", GRAPH]
    tm = m[m.split == "test"]
    targets = list(LABEL)
    fig, axes = plt.subplots(1, len(targets), figsize=(18, 4.2), sharey=True)
    for ax, t in zip(axes, targets):
        d = tm[tm.target == t].set_index("model").reindex(order)
        rel = 100 * (d.mae / d.loc[PERSIST, "mae"] - 1)
        colors = ["#9e9e9e"] * 6 + ["#7aa6d6", "#1f5fa8"]
        ax.barh(range(len(order)), rel, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{LABEL[t]}\npersistence MAE {d.loc[PERSIST, 'mae']:.3g}", fontsize=10)
        ax.set_xlabel("% change in test MAE vs persistence")
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=9)
    fig.suptitle("2023 test year: error relative to 'same as last year' (left = better)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "test_mae_vs_persistence.png", dpi=150)

    print(m[m.split == "test"].pivot(index="model", columns="target", values="mae").reindex(order).round(3).to_string())
    print(pd.DataFrame(comps).round(4).to_string())


if __name__ == "__main__":
    main()
