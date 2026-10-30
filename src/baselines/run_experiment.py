"""First forecasting results: baselines vs. the same models with graph-neighbour features (no GNN yet).

Question: does information from related schools (district, nearby, academically similar, same ESC region)
improve next-year forecasts beyond each school's own history?

Split (configs/split.yaml): tune on train 2018-2022 -> validate 2023; refit on 2018-2023 -> test 2024.
Targets: ACT, SAT, college-ready %, dropout, attendance, grade-level retention, Academic Risk Index. Risk index
has no academic-similarity layer (every academic indicator is one of its components).
Every model is scored on exactly the same rows: target observed AND the school's previous-year value
observed (so the persistence baseline is defined).
Two target modes (--mode):
  level     models predict the raw value (the first run; COVID-year swings in 2020-21 leak into 2023 forecasts)
  relative  models predict the school's deviation from its year's statewide mean, from inputs expressed the same
            way; the forecast adds LAST year's statewide mean. Common year shocks (COVID) drop out of training,
            and nothing from the forecast year is used.
  zscore    as relative, but deviations are also divided by that year's statewide spread (sd), so a year whose
            spread collapsed (attendance 2020: sd 1.0 vs ~3.3) does not teach squashed differences. The forecast
            is last year's mean + last year's sd x predicted z.
Writes results/<mode>/predictions.csv (hand-off 3 format) and results/<mode>/tuning.csv.
Run from the repo root:  python3 src/baselines/run_experiment.py --mode relative
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "data" / "graph"
RES = ROOT / "results"
SPLIT = yaml.safe_load((ROOT / "configs" / "split.yaml").read_text())
TRAIN, VAL, TEST, SEEDS = SPLIT["train_years"], SPLIT["val_years"], SPLIT["test_years"], SPLIT["seeds"]
ID = {"campus_id": str, "src_campus_id": str, "dst_campus_id": str, "district_id": str, "esc_region": str}

TARGETS = ["act_composite", "sat_total", "sat_act_above_criterion", "dropout_9_12", "attendance_rate",
           "retention_rate", "risk_index", "tsi_both"]
# tsi_both (added 2026-10-25) is not a context input, so the other targets' Phase 3 inputs are unchanged
CONTEXT_VARS = ["sat_total", "act_composite", "sat_act_above_criterion", "sat_act_participation",
                "attendance_rate", "dropout_9_12", "chronic_absenteeism"]
DEMO_VARS = ["enrollment", "pct_econ_disadv", "pct_english_learner", "pct_special_ed", "pct_at_risk",
             "pct_african_american", "pct_hispanic", "pct_white", "pct_asian", "pct_american_indian",
             "pct_pacific_islander", "pct_two_or_more", "mobility_rate"]   # previous year's student make-up
LAYERS = ["district", "geographic", "academic", "demographic", "region"]


# ------------------------------------------------------------------------------------------- features
def neighbour_features(nodes: pd.DataFrame, edges: dict, target: str, edge_set: str | None) -> pd.DataFrame:
    """Weighted mean of neighbours' previous-year value and previous change, per layer.

    Uses <target>_lag1 and <target>_change_lag1 of neighbours, i.e. years t-1 and t-2 only.
    """
    vals = nodes.set_index(["campus_id", "year"])[[f"{target}_lag1", f"{target}_change_lag1"]]
    out = nodes[["campus_id", "year"]].copy()
    for layer in LAYERS:
        if layer == "academic" and edge_set is None:   # risk_index: every academic indicator is a component
            continue
        e = edges[layer]
        if layer == "academic":
            e = e[e.edge_set == edge_set]          # similarity profile built without this target's family
        both = pd.concat([e[["src_campus_id", "dst_campus_id", "year", "weight"]],
                          e.rename(columns={"src_campus_id": "dst_campus_id", "dst_campus_id": "src_campus_id"})
                          [["src_campus_id", "dst_campus_id", "year", "weight"]]], ignore_index=True)
        nb = both.join(vals, on=["dst_campus_id", "year"])
        for col, name in ((f"{target}_lag1", "lag1"), (f"{target}_change_lag1", "change")):
            ok = nb[col].notna()
            agg = (nb[ok].assign(wx=nb.loc[ok, col] * nb.loc[ok, "weight"])
                   .groupby(["src_campus_id", "year"])[["wx", "weight"]].sum())
            m = (agg.wx / agg.weight).rename(f"nbr_{layer}_{name}")
            out = out.merge(m, left_on=["campus_id", "year"], right_index=True, how="left")
    return out


def feature_table(nodes: pd.DataFrame, edges: dict, target: str, edge_set: str | None) -> tuple[pd.DataFrame, dict]:
    t = target
    df = nodes[["campus_id", "year", t, f"{t}_lag1", f"{t}_lag2", f"{t}_change_lag1", f"{t}_n_prior_obs",
                "covid_affected_year", "is_charter", "urbanicity", "esc_region"]].copy()
    own = [f"{t}_lag1", f"{t}_lag2", f"{t}_change_lag1", f"{t}_n_prior_obs"]
    ctx = []
    for v in CONTEXT_VARS:
        if v != t:
            df[f"ctx_{v}_lag1"] = nodes[f"{v}_lag1"]
            ctx.append(f"ctx_{v}_lag1")
    for v in DEMO_VARS:
        df[f"demo_{v}_lag1"] = nodes[f"{v}_lag1"]
        ctx.append(f"demo_{v}_lag1")
    for u in ["City", "Suburb", "Town", "Rural"]:
        df[f"urb_{u}"] = (df.urbanicity == u).astype(int)
    ctx += ["covid_affected_year", "is_charter"] + [f"urb_{u}" for u in ["City", "Suburb", "Town", "Rural"]]
    nb = neighbour_features(nodes, edges, t, edge_set)
    df = df.merge(nb, on=["campus_id", "year"], how="left", validate="one_to_one")
    graph = [c for c in nb.columns if c.startswith("nbr_")]
    df["_raw_lag1"] = df[f"{t}_lag1"]                 # persistence always uses the raw previous value
    df["_y_fit"], df["_offset"], df["_scale"] = df[t], 0.0, 1.0
    return df, {"own": own, "ctx": ctx, "graph": graph}


def make_relative(df: pd.DataFrame, nodes: pd.DataFrame, t: str, fg: dict, scale: bool = False) -> pd.DataFrame:
    """Express target and every level input as a deviation from that year's statewide (cohort) mean.

    Training target: y_t - mean_t. Forecast: mean_{t-1} + predicted deviation, so the forecast year's own
    mean is never used. Inputs from year t-k subtract mean_{t-k}; changes subtract the mean's change.
    """
    df = df.copy()
    cols = list(dict.fromkeys([t] + CONTEXT_VARS))
    mu, sd = nodes.groupby("year")[cols].mean(), nodes.groupby("year")[cols].std()
    m = lambda v, k: (df.year - k).map(mu[v])
    s = (lambda v, k: (df.year - k).map(sd[v])) if scale else (lambda v, k: 1.0)
    z = lambda x, v, k: (x - m(v, k)) / s(v, k)
    lag1, lag2 = df[f"{t}_lag1"].copy(), df[f"{t}_lag2"].copy()
    df["_y_fit"] = z(df[t], t, 0)
    df["_offset"], df["_scale"] = m(t, 1), s(t, 1)
    df[f"{t}_lag1"] = z(lag1, t, 1)
    df[f"{t}_lag2"] = z(lag2, t, 2)
    df[f"{t}_change_lag1"] = df[f"{t}_lag1"] - df[f"{t}_lag2"]
    for v in CONTEXT_VARS:
        if v != t:
            df[f"ctx_{v}_lag1"] = z(df[f"ctx_{v}_lag1"], v, 1)
    for c in fg["graph"]:
        if c.endswith("_lag1"):
            df[c] = z(df[c], t, 1)
        else:   # neighbours' change t-2 -> t-1, minus the statewide change, in t-1 spread units
            df[c] = (df[c] - (m(t, 1) - m(t, 2))) / s(t, 1)
    return df


def impute_for_linear(X: pd.DataFrame, target: str) -> pd.DataFrame:
    """LR/KNN can't take NaN: a missing lag2 is set to lag1 (no change) and flagged; other gaps use train medians."""
    X = X.copy()
    if f"{target}_lag2" in X:
        X[f"{target}_lag2_missing"] = X[f"{target}_lag2"].isna().astype(int)
        X[f"{target}_lag2"] = X[f"{target}_lag2"].fillna(X[f"{target}_lag1"])
        X[f"{target}_change_lag1"] = X[f"{target}_lag1"] - X[f"{target}_lag2"]
    return X


# --------------------------------------------------------------------------------------------- models
def make(kind: str, params: dict, seed: int):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1, **params)
    if kind == "gb":
        return HistGradientBoostingRegressor(max_iter=200, random_state=seed, **params)
    if kind == "xgb":
        return XGBRegressor(n_estimators=200, random_state=seed, n_jobs=-1, tree_method="hist", **params)
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(**params))
    if kind == "lr":
        return LinearRegression()
    raise ValueError(kind)


GRIDS = {
    # light grid (kept small so the full run finishes in minutes on a laptop CPU)
    "rf": [{"min_samples_leaf": a, "max_features": 0.5} for a in [5, 20]],
    "gb": [{"learning_rate": a, "max_leaf_nodes": b, "min_samples_leaf": 20} for a, b in [(0.1, 15), (0.05, 31)]],
    "xgb": [{"learning_rate": a, "max_depth": b, "min_child_weight": 20, "subsample": 0.8} for a, b in [(0.1, 4), (0.05, 6)]],
    "knn": [{"n_neighbors": k} for k in [10, 20, 40]],
    "lr": [{}],
}
# model name -> (kind, feature groups)
MODELS = {
    "linear_regression (own lags)": ("lr", ["own"]),
    "knn (own lags)": ("knn", ["own"]),
    "random_forest (own lags)": ("rf", ["own"]),
    "random_forest (+context)": ("rf", ["own", "ctx"]),
    "gradient_boosting (+context)": ("gb", ["own", "ctx"]),
    "random_forest (+context +graph)": ("rf", ["own", "ctx", "graph"]),
    "gradient_boosting (+context +graph)": ("gb", ["own", "ctx", "graph"]),
    "xgboost (+context)": ("xgb", ["own", "ctx"]),
    "xgboost (+context +graph)": ("xgb", ["own", "ctx", "graph"]),
}
for layer in LAYERS:   # single-layer ablation on the strongest model family
    MODELS[f"gradient_boosting (+context +{layer} only)"] = ("gb", ["own", "ctx", f"graph:{layer}"])


def cols_for(groups: list[str], fg: dict) -> list[str]:
    cols = []
    for g in groups:
        if g.startswith("graph:"):
            cols += [c for c in fg["graph"] if c.startswith(f"nbr_{g.split(':')[1]}_")]
        else:
            cols += fg[g]
    return cols


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def fit_predict(kind, params, seed, Xtr, ytr, Xte, target):
    if kind in ("lr", "knn"):
        Xtr, Xte = impute_for_linear(Xtr, target), impute_for_linear(Xte, target)
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    m = make(kind, params, seed).fit(Xtr, ytr)
    return m.predict(Xte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["level", "relative", "zscore"], default="zscore")
    ap.add_argument("--targets", nargs="*", default=TARGETS, help="subset of targets (smoke tests)")
    ap.add_argument("--out", default=None, help="output folder name under results/ (default: the mode)")
    args = ap.parse_args()
    mode = args.mode
    out_dir = RES / (args.out or mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"target mode: {mode} -> {out_dir}")
    nodes = pd.read_csv(G / "nodes.csv", dtype=ID)
    edges = {l: pd.read_csv(G / f"edges_{l}.csv", dtype=ID) for l in LAYERS}
    sets = pd.read_csv(G / "academic_edge_sets.csv").set_index("target").edge_set
    preds, tuning = [], []

    for target in args.targets:
        df, fg = feature_table(nodes, edges, target, sets.get(target))
        if mode in ("relative", "zscore"):
            df = make_relative(df, nodes, target, fg, scale=(mode == "zscore"))
        rows = df[df[target].notna() & df["_raw_lag1"].notna()]
        tr, va, te = (rows[rows.year.isin(y)] for y in (TRAIN, VAL, TEST))
        trva = rows[rows.year.isin(TRAIN + VAL)]
        print(f"\n== {target}: train {len(tr)} (years {sorted(tr.year.unique())}), val {len(va)}, test {len(te)}")

        # persistence: next year = this year
        for split_df, split in ((va, "val"), (te, "test")):
            for s in SEEDS:
                preds.append(pd.DataFrame({"model": "persistence (last year's value)", "seed": s,
                                           "campus_id": split_df.campus_id, "year": split_df.year,
                                           "target": target, "y_true": split_df[target],
                                           "y_pred": split_df["_raw_lag1"], "split": split}))
        for name, (kind, groups) in MODELS.items():
            X = cols_for(groups, fg)
            if any(g.startswith("graph:") for g in groups) and not any(c in X for c in fg["graph"]):
                continue                      # layer absent for this target (risk_index has no academic layer)
            # tune on the validation year with seed 0
            best, best_mae = None, np.inf
            for params in GRIDS[kind]:
                p = fit_predict(kind, params, 0, tr[X], tr["_y_fit"], va[X], target) * va["_scale"].to_numpy() + va["_offset"].to_numpy()
                score = mae(va[target], p)
                tuning.append({"target": target, "model": name, "params": str(params), "val_mae": score})
                if score < best_mae:
                    best, best_mae = params, score
            seeds = SEEDS if kind in ("rf", "gb", "xgb") else [SEEDS[0]]
            for s in seeds:
                pv = fit_predict(kind, best, s, tr[X], tr["_y_fit"], va[X], target) * va["_scale"].to_numpy() + va["_offset"].to_numpy()
                pt = fit_predict(kind, best, s, trva[X], trva["_y_fit"], te[X], target) * te["_scale"].to_numpy() + te["_offset"].to_numpy()
                for split_df, p, split in ((va, pv, "val"), (te, pt, "test")):
                    preds.append(pd.DataFrame({"model": name, "seed": s, "campus_id": split_df.campus_id,
                                               "year": split_df.year, "target": target,
                                               "y_true": split_df[target], "y_pred": p, "split": split}))
            if kind not in ("rf", "gb", "xgb"):          # deterministic: copy seed 0 so every model has 5 seeds
                last = [d for d in preds[-2:]]
                for s in SEEDS[1:]:
                    preds += [d.assign(seed=s) for d in last]
            print(f"  {name:45s} best {best} val MAE {best_mae:.3f}", flush=True)

    pd.concat(preds, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(out_dir / "tuning.csv", index=False)


if __name__ == "__main__":
    main()
