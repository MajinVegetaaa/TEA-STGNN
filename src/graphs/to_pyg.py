"""Convert the yearly graph CSVs (data/graph/) into PyTorch Geometric snapshots, one HeteroData per year.

Every snapshot uses the same campus order (all 1,670 cohort campuses, sorted by campus_id), so node i is the same
school in every year and a recurrent model can run across years; `present` marks campuses with a row that year.
Inputs are chosen by role in node_feature_catalog.csv (history features, COVID flag, static school info), never
same-year observations. Missing inputs get a 0/1 indicator; imputation medians and scaling are fit on the training
years only (configs/split.yaml). Targets are kept raw, with masks:
  y_mask     target observed for a present campus (use for the training loss)
  eval_mask  target AND its previous-year value observed: exactly the rows Phase 3 scored (numbers to beat)
year_stats holds each year's statewide mean and sd per target, for the relative / zscore target versions
(a forecast for year t may use year t-1's stats only).
Relations: one ('campus', <layer>, 'campus') edge type per layer, undirected edges stored in both directions;
academic edges are one relation per edge_set (academic_excl_sat, ...). Temporal edges are implied by the fixed order. Every snapshot has every
relation (empty where a year has none, e.g. 2018 academic and demographic).
Writes data/graph/pyg/snapshots.pt. Run from the repo root:  python3 src/graphs/to_pyg.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "data" / "graph"
OUT = G / "pyg" / "snapshots.pt"
SPLIT = yaml.safe_load((ROOT / "configs" / "split.yaml").read_text())
ID = {"campus_id": str, "src_campus_id": str, "dst_campus_id": str, "district_id": str, "esc_region": str}

TARGETS = ["act_composite", "sat_total", "sat_act_above_criterion", "dropout_9_12", "attendance_rate",
           "retention_rate", "risk_index", "tsi_both"]
LAYERS = ["district", "geographic", "demographic", "region", "academic"]
STATIC_NUM = ["is_charter", "is_magnet", "is_virtual_or_hybrid", "is_residential_facility", "serves_full_9_12",
              "latitude", "longitude"]
STATIC_CAT = ["urbanicity", "esc_region"]


def input_columns(cat: pd.DataFrame) -> list[str]:
    temporal = cat.loc[cat.role == "input feature (temporal)", "column"].tolist()
    known = cat.loc[cat.role == "input feature", "column"].tolist()          # covid_affected_year
    banned = set(cat.loc[cat.role.str.startswith("same-year observation"), "column"])
    cols = temporal + known + STATIC_NUM
    assert not banned & set(cols), f"same-year observations in inputs: {banned & set(cols)}"
    return cols


def build_features(nodes: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Numeric inputs + missing indicators + one-hot categoricals; medians and scaling from training years only."""
    train = nodes.year.isin(SPLIT["train_years"]).to_numpy()
    # an input never observed in the training years can't be scaled or learned from (e.g. chronic_absenteeism_lag3,
    # first available in 2023): drop it rather than show the model unscaled values at validation/test time
    empty = [c for c in cols if nodes.loc[train, c].isna().all()]
    cols = [c for c in cols if c not in empty]
    if empty:
        print(f"dropped {len(empty)} inputs with no training-year values: " + ", ".join(empty))
    X = nodes[cols].astype(float)
    miss = [c for c in cols if X[c].isna().any()]
    ind = X[miss].isna().astype(float).add_suffix("_missing")
    med = X[train].median()
    X = X.fillna(med)
    mu, sd = X[train].mean(), X[train].std().replace(0, 1).fillna(1)
    X = (X - mu) / sd
    onehot = pd.get_dummies(nodes[STATIC_CAT].astype(str), prefix=STATIC_CAT, dtype=float)
    out = pd.concat([X, ind, onehot], axis=1)
    scaler = {"columns": cols, "median": med.to_dict(), "mean": mu.to_dict(), "sd": sd.to_dict(),
              "fit_years": SPLIT["train_years"]}
    return out, scaler


def main() -> None:
    nodes = pd.read_csv(G / "nodes.csv", dtype=ID, low_memory=False)
    cat = pd.read_csv(G / "node_feature_catalog.csv")
    sets = pd.read_csv(G / "academic_edge_sets.csv").set_index("target").edge_set
    years = sorted(nodes.year.unique())
    campuses = sorted(nodes.campus_id.unique())
    pos = {c: i for i, c in enumerate(campuses)}
    n = len(campuses)

    feats, scaler = build_features(nodes, input_columns(cat))
    names = feats.columns.tolist()
    y_stats = {t: {"mean": nodes.groupby("year")[t].mean().to_dict(), "sd": nodes.groupby("year")[t].std().to_dict()}
               for t in TARGETS}
    edges = {l: pd.read_csv(G / f"edges_{l}.csv", dtype=ID) for l in LAYERS}

    relations = [l for l in LAYERS if l != "academic"] + \
        [f"academic_{es.removeprefix('excl_')}" for es in sorted(edges["academic"].edge_set.unique())]
    snaps = []
    for y in years:
        rows = nodes.year == y
        idx = torch.tensor(nodes.loc[rows, "campus_id"].map(pos).to_numpy())
        d = HeteroData()
        x = torch.zeros(n, len(names))
        x[idx] = torch.tensor(feats[rows].to_numpy(), dtype=torch.float32)
        present = torch.zeros(n, dtype=torch.bool)
        present[idx] = True
        yv = torch.full((n, len(TARGETS)), float("nan"))
        yv[idx] = torch.tensor(nodes.loc[rows, TARGETS].to_numpy(dtype=float), dtype=torch.float32)
        lag1 = torch.full((n, len(TARGETS)), float("nan"))
        lag1[idx] = torch.tensor(nodes.loc[rows, [f"{t}_lag1" for t in TARGETS]].to_numpy(dtype=float),
                                 dtype=torch.float32)
        d["campus"].x, d["campus"].present, d["campus"].y, d["campus"].y_lag1 = x, present, yv, lag1
        d["campus"].y_mask = ~torch.isnan(yv) & present[:, None]
        d["campus"].eval_mask = d["campus"].y_mask & ~torch.isnan(lag1)
        d.year = int(y)
        for layer, e in edges.items():
            e = e[e.year == y]
            groups = e.groupby("edge_set") if layer == "academic" else [(None, e)]
            for es, g in groups:
                rel = f"academic_{es.removeprefix('excl_')}" if es else layer
                src, dst = g.src_campus_id.map(pos), g.dst_campus_id.map(pos)
                assert src.notna().all() and dst.notna().all(), f"{y} {rel}: endpoint not in campus index"
                s, t_ = torch.tensor(src.to_numpy()), torch.tensor(dst.to_numpy())
                w = torch.tensor(g.weight.to_numpy(), dtype=torch.float32)
                d["campus", rel, "campus"].edge_index = torch.stack([torch.cat([s, t_]), torch.cat([t_, s])])
                d["campus", rel, "campus"].edge_weight = torch.cat([w, w])
        for rel in relations:      # same relation set every year (2018 has no academic/demographic edges: no prior year)
            if ("campus", rel, "campus") not in d.edge_types:
                d["campus", rel, "campus"].edge_index = torch.zeros(2, 0, dtype=torch.long)
                d["campus", rel, "campus"].edge_weight = torch.zeros(0)
        snaps.append(d)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"years": [int(y) for y in years], "campus_ids": campuses, "feature_names": names, "targets": TARGETS,
                "academic_relation": {t: f"academic_{sets[t].removeprefix('excl_')}" for t in TARGETS if t in sets},
                "split": SPLIT, "scaler": scaler, "year_stats": y_stats, "snapshots": snaps}, OUT)
    check(snaps, nodes, edges, campuses, names)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(snaps)} years x {n} campuses x {len(names)} features")


def check(snaps, nodes, edges, campuses, names) -> None:
    """Self-checks: counts, round-trips, and the evaluation rows equal Phase 3's."""
    ok = []
    rec = lambda name, cond, detail="": (ok.append(bool(cond)), print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}"))
    rec("one fixed campus order (cohort size)", len(campuses) == nodes.campus_id.nunique(), str(len(campuses)))
    rec("no NaN in inputs", all(not torch.isnan(s["campus"].x).any() for s in snaps))
    rec("no same-year target in inputs", not set(TARGETS) & set(names))
    for s in snaps:
        y = s.year
        rec(f"{y}: present nodes = rows in nodes.csv", int(s["campus"].present.sum()) == int((nodes.year == y).sum()))
        for layer, e in edges.items():
            ne = int((e.year == y).sum())
            got = sum(s[k].edge_index.shape[1] for k in s.edge_types if k[1] == layer or k[1].startswith(layer + "_"))
            rec(f"{y}: {layer} edges = 2 x CSV rows", got == 2 * ne, f"{got} vs 2x{ne}")
    nd = nodes.set_index(["campus_id", "year"])
    pos = {c: i for i, c in enumerate(campuses)}
    smp = nodes.sample(300, random_state=0)
    err = 0
    for _, r in smp.iterrows():
        s = snaps[[x.year for x in snaps].index(r.year)]
        for j, t in enumerate(TARGETS):
            a, b = float(s["campus"].y[pos[r.campus_id], j]), nd.at[(r.campus_id, r.year), t]
            err += not ((np.isnan(a) and pd.isna(b)) or abs(a - b) < 1e-3)
    rec("y round-trips against nodes.csv (300 sampled rows x 7 targets)", err == 0, f"{err} mismatches")
    pred = ROOT / "results" / "level" / "predictions.csv"
    if pred.exists():
        p = pd.read_csv(pred, dtype={"campus_id": str}, usecols=["model", "seed", "campus_id", "year", "target"])
        p = p[(p.model == "persistence (last year's value)") & (p.seed == 0)]
        for t in TARGETS:
            for y in SPLIT["val_years"] + SPLIT["test_years"]:
                s = snaps[[x.year for x in snaps].index(y)]
                m = s["campus"].eval_mask[:, TARGETS.index(t)]
                mine = {campuses[i] for i in m.nonzero().flatten().tolist()}
                theirs = set(p[(p.target == t) & (p.year == y)].campus_id)
                rec(f"eval rows = Phase 3 rows: {t} {y}", mine == theirs, f"{len(mine)} vs {len(theirs)}")
    else:
        print(f"SKIP  eval rows = Phase 3 rows ({len(TARGETS) * len(SPLIT['val_years'] + SPLIT['test_years'])} checks): "
              f"{pred.relative_to(ROOT)} not found (gitignored; upload it or run Phase 3 to enable)")
    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    assert all(ok), "converter self-checks failed"


if __name__ == "__main__":
    main()
