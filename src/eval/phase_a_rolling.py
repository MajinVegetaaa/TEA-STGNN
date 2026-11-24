"""Phase A2-A4: rolling-origin tree residuals, graph-signal tests and shrinkage baselines (2020-2023 only).

The 2023 bar was chosen on 2023 alone. This script replays one PRE-SPECIFIED tree at every forecast origin
s = 2020..2023, trained only on target years before s, and asks whether the graph carries forecastable signal.

Base tree (fixed, not tuned per origin): HistGradientBoosting, learning_rate 0.05, max_leaf_nodes 31,
min_samples_leaf 20, 200 iterations, seed 0; S1c "all" inputs (own history + context + every D1 source, all
lag 1), zscore mode (deviations from each year's statewide mean / sd, so the forecast uses last year's mean
and sd); NO graph features. Rows: target observed and last year's value observed (the protocol rows).

Outputs (results/phase_a/):
  rolling_residuals.csv  campus, year, target, y, persistence, tree prediction, residual e = y - prediction
  rolling_scores.csv     per target and origin: tree, multi-year means and shrinkage baselines vs persistence
  moran.csv              same-year Moran's I of tree residuals per relation (descriptive), permutation p
  lag_test.csv           the deciding test: does the weighted mean of neighbours' LAST-YEAR residuals improve a
                         correction of this year's residual beyond the school's own last-year residual?
                         Fitted on earlier origins only, scored forward at 2022 and 2023; permutation null.
  audit.csv              usable rows per target and year (training transitions per origin)
  gate.csv               decision rule, fixed before the full run: a relation carries graph signal for a target
                         only if, at BOTH forward origins (2022, 2023), adding neighbours' last-year residuals
                         beats the tree alone AND the own-residual correction, with permutation p <= 0.05.

Leakage rules: nodes and edges are read only up to 2023 (files are sorted by year; reading stops at the first
later row). A year-s edge is built from year s-1 profiles (docs/GRAPH.md). Residuals of year s-1 are known
when forecasting s (D1). Every coefficient is fitted on earlier origins than the one it is scored on.

    .venv-expanded/bin/python src/eval/phase_a_rolling.py [--targets attendance_rate ...] [--perms 199]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "baselines"))
os.environ.pop("TEA_EXPANDED", None)
import run_experiment as B  # noqa: E402
import run_s1c as S  # noqa: E402

GRAPH = ROOT / "data" / "graph_v2"
OUT = ROOT / "results" / "phase_a"
CUTOFF = 2023
ORIGINS = [2020, 2021, 2022, 2023]
FORWARD = [2022, 2023]                       # lag test is scored forward at these origins
RELATIONS = ["district", "geographic", "demographic", "region", "academic"]
TARGETS = S.TARGETS
TREE = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, max_iter=200, random_state=0)
ID_COLS = {"campus_id", "src_campus_id", "dst_campus_id", "district_id", "esc_region"}
TEXT_COLS = ID_COLS | {"edge_set", "edge_type", "method", "urbanicity", "campus_name", "district_name"}


# ------------------------------------------------------------------------------------------- reading
def read_through(path: Path, cutoff: int = CUTOFF) -> pd.DataFrame:
    """Rows with year <= cutoff from a CSV sorted by year; stops at the first later row, refuses unsorted files."""
    rows, last = [], None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        yi = header.index("year")
        for row in reader:
            year = int(row[yi])
            if last is not None and year < last:
                raise ValueError(f"{path.name} is not sorted by year; refusing to read past a cutoff")
            last = year
            if year > cutoff:
                break
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        if col == "year":
            df[col] = df[col].astype(int)
        elif col not in TEXT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load():
    nodes = read_through(GRAPH / "nodes.csv")
    edges = {layer: read_through(GRAPH / f"edges_{layer}.csv") for layer in RELATIONS}
    sets = pd.read_csv(GRAPH / "academic_edge_sets.csv").set_index("target").edge_set
    if nodes.year.max() != CUTOFF or any(e.year.max() > CUTOFF for e in edges.values()):
        raise AssertionError("data beyond the cutoff was read")
    return nodes, edges, sets


# ------------------------------------------------------------------------------------------- graph tools
def relation_edges(edges: dict, relation: str, edge_set: str | None, year: int) -> pd.DataFrame | None:
    e = edges[relation]
    if relation == "academic":
        if edge_set is None:
            return None                       # risk_index has no academic relation (all indicators are components)
        e = e[e.edge_set == edge_set]
    e = e[e.year == year]
    both = pd.concat([e[["src_campus_id", "dst_campus_id", "weight"]],
                      e.rename(columns={"src_campus_id": "dst_campus_id", "dst_campus_id": "src_campus_id"})
                      [["src_campus_id", "dst_campus_id", "weight"]]], ignore_index=True)
    both = both[both.src_campus_id != both.dst_campus_id]
    return both.groupby(["src_campus_id", "dst_campus_id"], as_index=False).weight.max()


def neighbour_mean(edge_list: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Weighted mean of neighbours' values (self excluded, missing neighbours dropped, weights renormalised)."""
    nb = edge_list.join(values.rename("v"), on="dst_campus_id")
    nb = nb[nb.v.notna() & (nb.src_campus_id != nb.dst_campus_id)]
    agg = nb.assign(wv=nb.v * nb.weight).groupby("src_campus_id")[["wv", "weight"]].sum()
    return (agg.wv / agg.weight).rename("nbr")


def morans_i(edge_list: pd.DataFrame, values: pd.Series, perms: int, rng: np.random.Generator) -> tuple[float, float, int]:
    """Moran's I of `values` on a symmetric weighted graph, with a two-sided permutation p-value."""
    values = values.dropna()
    idx = {c: i for i, c in enumerate(values.index)}
    e = edge_list[edge_list.src_campus_id.isin(idx) & edge_list.dst_campus_id.isin(idx)]
    if e.empty:
        return np.nan, np.nan, 0
    n = len(values)
    w = sparse.csr_matrix((e.weight.to_numpy(float), (e.src_campus_id.map(idx), e.dst_campus_id.map(idx))), shape=(n, n))
    z0 = values.to_numpy(float) - values.mean()

    def stat(z):
        return n / w.sum() * float(z @ (w @ z)) / float(z @ z)
    observed = stat(z0)
    null = np.array([stat(rng.permutation(z0)) for _ in range(perms)])
    p = (1 + np.sum(np.abs(null - null.mean()) >= abs(observed - null.mean()))) / (perms + 1)
    return observed, p, int(len(e) // 2)


# ------------------------------------------------------------------------------------------- base tree
def target_table(nodes, edges, target, edge_set) -> pd.DataFrame:
    df, fg = S.d1_feature_table(nodes, edges, target, edge_set)
    df = B.make_relative(df, nodes, target, fg, scale=True)            # zscore mode
    df.attrs["features"] = fg["own"] + fg["ctx"]                       # no graph features in the base tree
    return df[df[target].notna() & df["_raw_lag1"].notna()]


def rolling_tree(df: pd.DataFrame, target: str) -> pd.DataFrame:
    feats, out = df.attrs["features"], []
    for s in ORIGINS:
        train, test = df[df.year < s], df[df.year == s]
        if train.empty or test.empty:
            continue
        model = HistGradientBoostingRegressor(**TREE).fit(train[feats], train["_y_fit"])
        pred = model.predict(test[feats]) * test["_scale"].to_numpy() + test["_offset"].to_numpy()
        out.append(pd.DataFrame({"campus_id": test.campus_id.to_numpy(), "year": s, "target": target,
                                 "y": test[target].to_numpy(), "persistence": test["_raw_lag1"].to_numpy(),
                                 "tree": pred, "district_id": test.district_id.to_numpy()
                                 if "district_id" in test else None}))
    res = pd.concat(out, ignore_index=True)
    res["resid"] = res.y - res.tree
    return res


# ------------------------------------------------------------------------------------------- baselines
def baselines(nodes: pd.DataFrame, res: pd.DataFrame, target: str) -> pd.DataFrame:
    """Multi-year means and two shrinkage forecasts, coefficients fitted on years before each origin."""
    cols = ["campus_id", "year", "district_id", "enrollment_lag1", f"{target}_lag1", f"{target}_lag2", f"{target}_lag3"]
    n = nodes[[c for c in cols if c in nodes]].rename(columns={f"{target}_lag1": "l1", f"{target}_lag2": "l2",
                                                                 f"{target}_lag3": "l3"})
    g = n.groupby(["district_id", "year"]).l1
    s, c = g.transform("sum"), g.transform("count")
    n["dist_loo"] = (s - n.l1.fillna(0)) / (c - n.l1.notna().astype(int))
    n["ma2"], n["ma3"] = n[["l1", "l2"]].mean(axis=1), n[["l1", "l2", "l3"]].mean(axis=1)
    d = res.merge(n.drop(columns=["district_id"]), on=["campus_id", "year"], how="left")
    rows = []
    for s in ORIGINS:
        cur = d[d.year == s]
        if cur.empty:
            continue
        hist = d[d.year < s]
        pers = np.abs(cur.persistence - cur.y).mean()
        out = {"target": target, "origin": s, "n": len(cur), "persistence_mae": pers,
               "tree": np.abs(cur.tree - cur.y).mean() / pers,
               "mean_2yr": np.abs(cur.ma2 - cur.y).mean() / pers, "mean_3yr": np.abs(cur.ma3 - cur.y).mean() / pers}
        for name, anchor in (("shrink_to_own_3yr", "ma3"), ("shrink_to_district", "dist_loo")):
            # y = lag1 + beta * (anchor - lag1); beta per enrollment tercile, fitted on earlier years only
            cur_pred = cur.persistence.copy()
            if not hist.empty:
                cuts = hist.enrollment_lag1.quantile([1 / 3, 2 / 3]).to_numpy()
                band = lambda f: np.digitize(f.enrollment_lag1.fillna(hist.enrollment_lag1.median()), cuts)
                hb, cb = band(hist), band(cur)
                for k in range(3):
                    h = hist[(hb == k) & hist[anchor].notna()]
                    x, yv = (h[anchor] - h.persistence).to_numpy(), (h.y - h.persistence).to_numpy()
                    beta = float(np.clip(x @ yv / (x @ x), 0, 1)) if len(h) > 10 and x @ x > 0 else 0.0
                    m = (cb == k) & cur[anchor].notna().to_numpy()
                    cur_pred[m] = cur.persistence[m] + beta * (cur[anchor][m] - cur.persistence[m])
            out[name] = np.abs(cur_pred - cur.y).mean() / pers
        rows.append(out)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- lag test
def lag_design(res: pd.DataFrame, edges, edge_set, relation: str) -> pd.DataFrame:
    """Per school and origin: this year's residual, own last-year residual, neighbours' last-year residual."""
    frames = []
    for s in ORIGINS[1:]:
        cur = res[res.year == s].set_index("campus_id")
        prev = res[res.year == s - 1].set_index("campus_id").resid
        el = relation_edges(edges, relation, edge_set, s)
        if el is None:
            return pd.DataFrame()
        frames.append(pd.DataFrame({"year": s, "e": cur.resid, "own": prev.reindex(cur.index),
                                    "nbr": neighbour_mean(el, prev).reindex(cur.index),
                                    "pers_err": (cur.persistence - cur.y).abs()}))
    return pd.concat(frames).fillna({"own": 0.0, "nbr": 0.0})          # no history -> zero correction


def fit_score(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> float:
    """MAE of this year's residual after a per-school correction fitted on earlier origins.

    The question is per-school (peer) signal, not the year-level shift: residuals and predictors are centred
    within each year and there is no intercept, so a past year's average residual (a COVID-era regime
    artefact) is never carried forward. Predictors at the scored origin are centred with that origin's own
    last-year values, which are known at forecast time. cols=[] means no correction (the tree alone).
    """
    if not cols:
        return float(np.abs(test.e.to_numpy()).mean())
    centre = lambda f, c: f[c] - f.groupby("year")[c].transform("mean")
    X = np.column_stack([centre(train, c) for c in cols])
    beta, *_ = np.linalg.lstsq(X, centre(train, "e").to_numpy(), rcond=None)
    Xt = np.column_stack([centre(test, c) for c in cols])
    return float(np.abs(test.e.to_numpy() - Xt @ beta).mean())


def lag_test(design: pd.DataFrame, target: str, relation: str, perms: int, rng) -> list[dict]:
    rows = []
    for s in FORWARD:
        train, test = design[design.year < s], design[design.year == s]
        pers = test.pers_err.mean()
        base, own, full = (fit_score(train, test, cols) for cols in ([], ["own"], ["own", "nbr"]))
        null = []
        for _ in range(perms):
            tr, te = train.copy(), test.copy()
            for frame in (tr, te):        # shuffle neighbour values within each year: breaks the graph, keeps marginals
                frame["nbr"] = frame.groupby("year").nbr.transform(lambda v: rng.permutation(v.to_numpy()))
            null.append(own - fit_score(tr, te, ["own", "nbr"]))
        gain = own - full
        rows.append({"target": target, "relation": relation, "origin": s, "n": len(test),
                     "tree_ratio": base / pers, "own_corr_ratio": own / pers,
                     "own_nbr_corr_ratio": full / pers, "nbr_gain_ratio_pts": gain / pers,
                     "perm_p": (1 + np.sum(np.array(null) >= gain)) / (perms + 1),
                     "beats_tree_and_own": bool(full < min(base, own)),
                     "corr_e_nbr": float(np.corrcoef(test.e, test.nbr)[0, 1]) if test.nbr.std() > 0 else np.nan})
    return rows


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--perms", type=int, default=199)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    nodes, edges, sets = load()
    B.G = GRAPH
    S.INPUT_SET = "all"
    residuals, scores, moran, lags, audit = [], [], [], [], []
    for target in args.targets:
        edge_set = sets.get(target)
        df = target_table(nodes, edges, target, edge_set)
        audit += [{"target": target, "year": y, "usable_rows": int(n)} for y, n in df.groupby("year").size().items()]
        res = rolling_tree(df, target)
        residuals.append(res)
        scores.append(baselines(nodes, res, target))
        for relation in RELATIONS:
            for s in ORIGINS:
                el = relation_edges(edges, relation, edge_set, s)
                if el is None:
                    continue
                cur = res[res.year == s].set_index("campus_id").resid
                i, p, m = morans_i(el, cur, args.perms, rng)
                moran.append({"target": target, "relation": relation, "year": s, "morans_i": i, "perm_p": p, "edges": m})
            design = lag_design(res, edges, edge_set, relation)
            if not design.empty:
                lags += lag_test(design, target, relation, args.perms, rng)
        print(f"done {target}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(residuals).to_csv(OUT / "rolling_residuals.csv", index=False)
    lag = pd.DataFrame(lags)
    lag["passes"] = lag.beats_tree_and_own & (lag.perm_p <= 0.05)
    gate = (lag.groupby(["target", "relation"]).agg(origins_passed=("passes", "sum"),
                                                     mean_gain_pts=("nbr_gain_ratio_pts", "mean")).reset_index())
    gate["graph_signal"] = gate.origins_passed == len(FORWARD)          # decision rule fixed before the run
    gate.to_csv(OUT / "gate.csv", index=False)
    for name, rows in (("rolling_scores", scores), ("moran", moran), ("lag_test", lags), ("audit", audit)):
        (pd.concat(rows) if name == "rolling_scores" else pd.DataFrame(rows)).to_csv(OUT / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
