"""Build graph node and edge tables (one snapshot per year; years from configs/data_sources.yaml) from data/processed/.

Run from the repo root:  python3 src/graphs/build_graph_data.py
Writes data/graph/*.csv. See GRAPH_DATA_DOCUMENTATION.md for every rule and parameter.
No model is trained here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
OUT = ROOT / "data" / "graph"
ID = {"campus_id": str, "district_id": str, "esc_region": str, "county_id": str, "nces_school_id": str,
      "nces_locale_code": str, "cbsa_code": str}
YEARS = [int(y) for y in yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "data_sources.yaml").read_text())["years"]]
EARTH_KM = 6371.0088

# Parameters live in configs/graph.yaml and are PENDING PROFESSOR REVIEW (Questions PDF Q10)
CFG = yaml.safe_load((ROOT / "configs" / "graph.yaml").read_text())
PARAM_STATUS = CFG["status"]
GEO_K = int(CFG["geographic"]["k"])
GEO_MAX_KM = float(CFG["geographic"]["max_km"])
GEO_TAU_KM = float(CFG["geographic"]["tau_km"])
ACAD_K = int(CFG["academic"]["k"])
ACAD_MIN_SHARED = int(CFG["academic"]["min_shared"])
# Academic profile, always taken from year t-1 for the year-t graph (never from year t)
ACAD_FEATURES = list(CFG["academic"]["features"])
DEMO_K = int(CFG["demographic"]["k"])
DEMO_MIN_SHARED = int(CFG["demographic"]["min_shared"])
DEMO_FEATURES = list(CFG["demographic"]["features"])

# One academic edge set per target family: the predicted variable, and anything it is arithmetically
# part of or derived from the same raw counts, is left out of the similarity profile. Otherwise the
# academic layer would be a nearest-neighbour lookup on the target's own history.
ACAD_EDGE_SETS = {
    "excl_sat":           ["sat_total"],                                 # SAT total = ERW + Math
    "excl_act":           ["act_composite"],                             # composite averages the subjects
    "excl_criterion":     ["sat_act_above_criterion"],
    "excl_participation": ["sat_act_participation"],
    "excl_attendance":    ["attendance_rate", "chronic_absenteeism"],    # both come from absence days
    "excl_dropout":       ["dropout_9_12"],
    "excl_retention":     ["retention_rate"],    # retention is not in the profile; full profile used
}
TARGET_EDGE_SET = {
    "sat_total": "excl_sat", "sat_erw": "excl_sat", "sat_math": "excl_sat",
    "act_composite": "excl_act", "act_english": "excl_act", "act_math": "excl_act",
    "sat_act_above_criterion": "excl_criterion", "sat_act_participation": "excl_participation",
    "attendance_rate": "excl_attendance", "chronic_absenteeism": "excl_attendance",
    "dropout_9_12": "excl_dropout", "retention_rate": "excl_retention",
    # TSI criteria can be met with the same SAT/ACT scores that make up the college-ready %, so TSI shares its family
    "tsi_both": "excl_criterion",
}

EDGE_COLS = ["src_campus_id", "dst_campus_id", "src_year", "dst_year", "year", "edge_type", "weight",
             "directed", "distance", "method"]
ACAD_COLS = EDGE_COLS[:6] + ["edge_set"] + EDGE_COLS[6:]

STATIC_META = ["district_id", "district_name", "district_type", "county_id", "county_name", "esc_region",
               "charter_type", "is_charter", "is_magnet", "is_virtual_or_hybrid", "is_residential_facility",
               "serves_full_9_12", "lowest_grade", "grade_range", "latitude", "longitude", "nces_locale_code",
               "urbanicity", "cbsa_code", "campus_name"]
LEAKY_META = ["enrollment_oct2025"]   # measured after the study period; kept out of nodes.csv


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def undirected(src: np.ndarray, dst: np.ndarray, **cols) -> pd.DataFrame:
    """Store each unordered pair once with src < dst; drop self-loops and duplicates (keep first)."""
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    df = pd.DataFrame({"src_campus_id": a, "dst_campus_id": b, **cols})
    df = df[df.src_campus_id != df.dst_campus_id]
    return df.drop_duplicates(["src_campus_id", "dst_campus_id"]).reset_index(drop=True)


def finish(df: pd.DataFrame, year: int, edge_type: str, directed: int, method: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["src_year"] = df["dst_year"] = year
    df["edge_type"] = edge_type
    df["directed"] = directed
    df["method"] = method
    if "weight" not in df:
        df["weight"] = 1.0
    if "distance" not in df:
        df["distance"] = np.nan
    return df[EDGE_COLS]


def clique_edges(nodes: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    src, dst = [], []
    for _, g in nodes.groupby(key):
        ids = np.sort(g.campus_id.to_numpy())
        i, j = np.triu_indices(len(ids), k=1)
        src.append(ids[i]); dst.append(ids[j])
    return (np.concatenate(src), np.concatenate(dst)) if src else (np.array([]), np.array([]))


# ---------------------------------------------------------------------------------------------- edges
def district_edges(nodes_y: pd.DataFrame, year: int) -> pd.DataFrame:
    s, d = clique_edges(nodes_y, "district_id")
    return finish(undirected(s, d), year, "district", 0, "same 6-digit district_id (Directory); weight 1")


def region_edges(nodes_y: pd.DataFrame, year: int) -> pd.DataFrame:
    s, d = clique_edges(nodes_y, "esc_region")
    return finish(undirected(s, d), year, "region", 0, "same ESC region (Directory, ESC Region PEIMS); weight 1")


def geographic_edges(nodes_y: pd.DataFrame, year: int) -> pd.DataFrame:
    g = nodes_y[(nodes_y.is_virtual_or_hybrid == 0) & nodes_y.latitude.notna()].reset_index(drop=True)
    lat, lon = g.latitude.to_numpy(), g.longitude.to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1, kind="stable")[:, :GEO_K]
    rows = np.repeat(np.arange(len(g)), GEO_K)
    cols = nn.ravel()
    dk = dist[rows, cols]
    keep = dk <= GEO_MAX_KM
    rows, cols, dk = rows[keep], cols[keep], dk[keep]
    ids = g.campus_id.to_numpy()
    e = undirected(ids[rows], ids[cols], distance=dk, weight=np.exp(-dk / GEO_TAU_KM))
    return finish(e, year, "geographic", 0,
                  f"{GEO_K} nearest (haversine, NCES 2024-25 coords) within {GEO_MAX_KM:g} km, union; "
                  f"weight exp(-km/{GEO_TAU_KM:g}); virtual/hybrid campuses excluded")


def academic_profile(master: pd.DataFrame, year: int, campus_ids: np.ndarray,
                     exclude: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """z-scored profile from year-1 values only. Scaler statistics use year-1 data only."""
    prev = master[master.year == year - 1].set_index("campus_id")
    # chronic absenteeism is absent for 2019; excluded features belong to the predicted target's family
    feats = [f for f in ACAD_FEATURES if f not in exclude and prev[f].notna().any()]
    X = prev.reindex(campus_ids)[feats]
    mu, sd = prev[feats].mean(), prev[feats].std(ddof=0)
    return (X - mu) / sd, feats


def academic_distances(Z: np.ndarray, min_shared: int = ACAD_MIN_SHARED) -> tuple[np.ndarray, np.ndarray]:
    """RMS z-difference over jointly observed features; inf where fewer than min_shared are shared."""
    # exact pairwise differences (the a^2 + b^2 - 2ab shortcut loses precision for near-identical rows)
    diff = Z[:, None, :] - Z[None, :, :]                  # NaN wherever either school lacks the feature
    shared = (~np.isnan(diff)).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt(np.nansum(diff ** 2, axis=2) / shared)
    d[shared < min_shared] = np.inf
    np.fill_diagonal(d, np.inf)
    return d, shared


def academic_edges(master: pd.DataFrame, nodes_y: pd.DataFrame, year: int,
                   edge_set: str) -> tuple[pd.DataFrame, dict]:
    ids = np.sort(nodes_y.campus_id.to_numpy())
    if year - 1 not in master.year.values:
        return pd.DataFrame(columns=ACAD_COLS), {"features": [], "sigma": np.nan, "eligible": 0}
    Z, feats = academic_profile(master, year, ids, ACAD_EDGE_SETS[edge_set])
    d, _ = academic_distances(Z.to_numpy())
    eligible = np.isfinite(d).any(axis=1)
    nn = np.argsort(d, axis=1, kind="stable")[:, :ACAD_K]
    rows = np.repeat(np.arange(len(ids)), ACAD_K)
    cols = nn.ravel()
    dk = d[rows, cols]
    keep = np.isfinite(dk)
    rows, cols, dk = rows[keep], cols[keep], dk[keep]
    e = undirected(ids[rows], ids[cols], distance=dk)
    sigma = float(np.median(e.distance))               # kernel width: median distance of this year's edges
    e["weight"] = np.exp(-(e.distance ** 2) / (2 * sigma ** 2))
    e = finish(e, year, "academic", 0,
               f"{ACAD_K}-NN on z-scored year-{year-1} profile ({', '.join(feats)}); RMS distance over >= "
               f"{ACAD_MIN_SHARED} shared features; weight exp(-d^2/(2*{sigma:.3f}^2))")
    e.insert(6, "edge_set", edge_set)
    return e, {"features": feats, "sigma": sigma, "eligible": int(eligible.sum())}


def demographic_profile(master: pd.DataFrame, year: int, campus_ids: np.ndarray) -> pd.DataFrame:
    """z-scored student-composition profile from year-1 TAPR membership; scaler statistics from year-1 only."""
    prev = master[master.year == year - 1].set_index("campus_id")[DEMO_FEATURES].copy()
    prev["enrollment"] = np.log1p(prev["enrollment"])            # school size on a log scale
    X = prev.reindex(campus_ids)
    return (X - prev.mean()) / prev.std(ddof=0)


def demographic_edges(master: pd.DataFrame, nodes_y: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    ids = np.sort(nodes_y.campus_id.to_numpy())
    if year - 1 not in master.year.values:
        return pd.DataFrame(columns=EDGE_COLS), {"sigma": np.nan, "eligible": 0}
    d, _ = academic_distances(demographic_profile(master, year, ids).to_numpy(), DEMO_MIN_SHARED)
    eligible = np.isfinite(d).any(axis=1)
    nn = np.argsort(d, axis=1, kind="stable")[:, :DEMO_K]
    rows = np.repeat(np.arange(len(ids)), DEMO_K)
    cols = nn.ravel()
    dk = d[rows, cols]
    keep = np.isfinite(dk)
    e = undirected(ids[rows[keep]], ids[cols[keep]], distance=dk[keep])
    sigma = float(np.median(e.distance))
    e["weight"] = np.exp(-(e.distance ** 2) / (2 * sigma ** 2))
    e = finish(e, year, "demographic", 0,
               f"{DEMO_K}-NN on z-scored year-{year-1} student composition ({len(DEMO_FEATURES)} features, "
               f"log enrollment); RMS distance over >= {DEMO_MIN_SHARED} shared; weight exp(-d^2/(2*{sigma:.3f}^2))")
    return e, {"sigma": sigma, "eligible": int(eligible.sum())}


def temporal_edges(nodes: pd.DataFrame) -> pd.DataFrame:
    k = nodes[["campus_id", "year"]]
    nxt = k.assign(year=k.year + 1)
    e = k.merge(nxt, on=["campus_id", "year"])          # (campus, t) exists and (campus, t-1) exists
    df = pd.DataFrame({"src_campus_id": e.campus_id, "dst_campus_id": e.campus_id,
                       "src_year": e.year - 1, "dst_year": e.year, "year": e.year,
                       "edge_type": "temporal", "weight": 1.0, "directed": 1, "distance": np.nan,
                       "method": "same campus, year t-1 -> t (consecutive years only)"})
    return df[EDGE_COLS]


# ----------------------------------------------------------------------------------------------- main
def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(P / "master_school_year.csv", dtype=ID)
    meta = pd.read_csv(P / "school_metadata.csv", dtype=ID)
    tf = pd.read_csv(P / "temporal_features.csv", dtype=ID)

    # ---------------- nodes: one row per campus-year, annual values + temporal features + static metadata
    meta_cols = ["campus_id"] + [c for c in STATIC_META if c not in ("district_id", "esc_region")]
    nodes = (master.merge(tf, on=["campus_id", "year"], how="left", validate="one_to_one")
                   .merge(meta[meta_cols], on="campus_id", how="left", validate="many_to_one"))
    nodes = nodes.drop(columns=["source_ccad", "source_cdrop", "source_demographics", "source_retention", "source_tsi"]) \
                 .sort_values(["year", "campus_id"])
    nodes.insert(2, "node_index", nodes.groupby("year").cumcount())     # 0..n-1 within each yearly snapshot
    nodes.to_csv(OUT / "nodes.csv", index=False)

    # feature catalogue: static vs annual
    annual = set(master.columns) - {"campus_id", "year", "district_id", "esc_region", "source_ccad", "source_cdrop",
                                    "source_demographics", "source_retention", "source_tsi"}
    demog = {v for v in master.columns if v.startswith("pct_")} | {"enrollment", "mobility_rate"}
    cat = []
    for c in nodes.columns:
        if c in ("campus_id", "year", "node_index"):
            kind, role = "key", "identifier"
        elif c in tf.columns:
            kind, role = "annual (derived from years < t)", "input feature (temporal)"
        elif c == "covid_affected_year":
            kind, role = "annual (known in advance)", "input feature"
        elif c in demog:
            kind = "annual (fall membership snapshot of year t)"
            role = "same-year observation: not an input for year t (use its _lag1)"
        elif c in annual:
            # observed from year-t outcome files: a target (or target mask), never an input for year t
            kind = "annual (observed in year t)"
            role = ("same-year observation: target mask only, not an input for year t" if c.endswith("_supp")
                    or c == "has_graduates_reported" else "same-year observation: target, not an input for year t")
        else:
            kind, role = "static (Directory Oct 2025 / NCES 2024-25)", "input feature (static) or label"
        cat.append({"column": c, "variation": kind, "role": role})
    pd.DataFrame(cat).to_csv(OUT / "node_feature_catalog.csv", index=False)

    # ---------------- edges
    out = {k: [] for k in ["district", "geographic", "academic", "region", "demographic"]}
    acad_info = {}
    for y in YEARS:
        ny = nodes[nodes.year == y]
        out["district"].append(district_edges(ny, y))
        out["region"].append(region_edges(ny, y))
        out["geographic"].append(geographic_edges(ny, y))
        e, info = demographic_edges(master, ny, y)
        out["demographic"].append(e)
        acad_info[(y, "demographic")] = info
        for es in ACAD_EDGE_SETS:
            e, info = academic_edges(master, ny, y, es)
            out["academic"].append(e)
            acad_info[(y, es)] = info
    edges = {k: pd.concat([e for e in v if len(e)], ignore_index=True) for k, v in out.items()}
    edges["temporal"] = temporal_edges(nodes)
    # unavailable: header-only files (no fabricated edges)
    edges["transition"] = pd.DataFrame(columns=EDGE_COLS)
    for k, df in edges.items():
        df.to_csv(OUT / f"edges_{k}.csv", index=False, float_format="%.9g")
    pd.DataFrame([{"target": t, "edge_set": es, "excluded_features": " ".join(ACAD_EDGE_SETS[es]),
                   "profile_features": " ".join(f for f in ACAD_FEATURES if f not in ACAD_EDGE_SETS[es])}
                  for t, es in TARGET_EDGE_SET.items()]).to_csv(OUT / "academic_edge_sets.csv", index=False)
    return {"nodes": nodes, "edges": edges, "master": master, "meta": meta, "tf": tf, "acad_info": acad_info}


if __name__ == "__main__":
    r = build()
    for k, df in r["edges"].items():
        print(k, len(df))
    print(f"graph parameters status: {PARAM_STATUS}")
    for (y, es), i in r["acad_info"].items():
        if i.get("features") or es == "demographic" and i["eligible"]:
            print(y, es, len(i.get("features", DEMO_FEATURES)), "features, sigma", round(i["sigma"], 3), "eligible", i["eligible"])
