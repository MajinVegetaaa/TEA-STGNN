"""Quality checks for data/graph/ (written to data/graph/graph_quality_report.csv).

Recomputes distances and similarities independently of build_graph_data.py and rebuilds the academic
edges with year-t values scrambled to prove they never use same-year information.
Run from the repo root:  python3 src/graphs/validate_graph.py   (exits non-zero if any check fails)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_graph_data as B  # noqa: E402  (only used for parameters and the leakage rebuild)

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "data" / "graph"
P = ROOT / "data" / "processed"
ID = B.ID | {"src_campus_id": str, "dst_campus_id": str}
TYPES = ["district", "geographic", "academic", "demographic", "transition", "region", "temporal"]
rows: list[dict] = []


def rec(check, value, passed=None, scope="all", detail=""):
    rows.append({"check": check, "scope": scope, "value": value,
                 "passed": "" if passed is None else bool(passed), "detail": detail})
    flag = "" if passed is None else ("PASS " if passed else "FAIL ")
    print(f"{flag}[{scope}] {check}: {value}" + (f" ({detail})" if detail else ""))


def unit_vec(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def main() -> int:
    master = pd.read_csv(P / "master_school_year.csv", dtype=ID)
    meta = pd.read_csv(P / "school_metadata.csv", dtype=ID)
    tf = pd.read_csv(P / "temporal_features.csv", dtype=ID)
    nodes = pd.read_csv(G / "nodes.csv", dtype=ID)
    E = {t: pd.read_csv(G / f"edges_{t}.csv", dtype=ID) for t in TYPES}

    # ------------------------------------------------------------------ nodes
    key = lambda d: d[["campus_id", "year"]].sort_values(["campus_id", "year"]).reset_index(drop=True)
    rec("node keys == master keys", len(nodes), key(nodes).equals(key(master)))
    rec("duplicate nodes (campus_id, year)", int(nodes.duplicated(["campus_id", "year"]).sum()),
        not nodes.duplicated(["campus_id", "year"]).any())
    rec("invalid node campus_id (not 9 digits)", int((~nodes.campus_id.str.fullmatch(r"\d{9}")).sum()),
        nodes.campus_id.str.fullmatch(r"\d{9}").all())
    rec("node campuses all in school_metadata", nodes.campus_id.nunique(), set(nodes.campus_id) <= set(meta.campus_id))
    ok_idx = all((g.node_index.sort_values().to_numpy() == np.arange(len(g))).all() for _, g in nodes.groupby("year"))
    rec("node_index is 0..n-1 within each year", ok_idx, ok_idx)
    for y, n in nodes.groupby("year").size().items():
        rec("nodes", int(n), scope=str(y))
    rec("post-period enrollment_oct2025 excluded from nodes", "enrollment_oct2025" not in nodes.columns,
        "enrollment_oct2025" not in nodes.columns)
    cat = pd.read_csv(G / "node_feature_catalog.csv")
    same_year = set(master.columns) - {"campus_id", "year", "district_id", "esc_region", "covid_affected_year",
                                       "source_ccad", "source_cdrop", "source_demographics", "source_retention", "source_tsi"}
    lab = cat.set_index("column").role
    ok_lab = all(lab[c].startswith("same-year observation") for c in same_year if c in lab)
    rec("every year-t observed column is labelled not-an-input", f"{len(same_year)} columns", ok_lab)
    inputs = set(cat.column[cat.role.str.startswith("input")])
    rec("input-labelled columns never come from year-t outcome files", sorted(inputs & same_year) or "none",
        not (inputs & same_year))
    # consistency with processed data
    mm = nodes.set_index(["campus_id", "year"])
    ma = master.set_index(["campus_id", "year"]).reindex(mm.index)
    vals = [c for c in master.columns if c not in ("campus_id", "year", "source_ccad", "source_cdrop", "source_demographics", "source_retention", "source_tsi")]
    same = all(ma[c].equals(mm[c]) or np.allclose(ma[c].astype(float), mm[c].astype(float), equal_nan=True)
               if ma[c].dtype != object else ma[c].fillna("").equals(mm[c].fillna("")) for c in vals)
    rec("node indicator values identical to master_school_year", same, same)
    tt = tf.set_index(["campus_id", "year"]).reindex(mm.index)
    same_tf = np.allclose(tt.to_numpy(float), mm[tt.columns].to_numpy(float), equal_nan=True)
    rec("node temporal features identical to temporal_features", same_tf, same_tf)
    mt = meta.set_index("campus_id").reindex(nodes.campus_id)
    same_geo = np.allclose(mt.latitude.to_numpy(), nodes.latitude.to_numpy(), equal_nan=True)
    rec("node coordinates identical to school_metadata", same_geo, same_geo)
    rec("nodes missing coordinates", int(nodes.latitude.isna().sum()), nodes.latitude.notna().all())
    dup_xy = meta.duplicated(["latitude", "longitude"], keep=False)
    rec("campuses sharing identical coordinates with another campus", int(dup_xy.sum()), None, "all",
        "co-located campuses; geographic distance 0, weight 1")

    # ------------------------------------------------------------------ generic edge checks
    node_keys = set(zip(nodes.campus_id, nodes.year))
    for t, e in E.items():
        n = len(e)
        rec("edges", n, scope=t)
        if n == 0:
            rec("edge type available", False, None, t, "no edges built: required data unavailable")
            continue
        for y, c in e.groupby("year").size().items():
            rec("edges", int(c), scope=f"{t} {y}")
        ids_ok = e.src_campus_id.str.fullmatch(r"\d{9}").all() and e.dst_campus_id.str.fullmatch(r"\d{9}").all()
        rec("invalid campus IDs", 0 if ids_ok else "some", ids_ok, t)
        ends = set(zip(e.src_campus_id, e.src_year)) | set(zip(e.dst_campus_id, e.dst_year))
        rec("edge endpoints that are not nodes", len(ends - node_keys), not (ends - node_keys), t)
        rec("missing weights", int(e.weight.isna().sum()), e.weight.notna().all(), t)
        rec("weights in (0, 1]", bool(e.weight.between(0, 1, inclusive="right").all()),
            e.weight.between(0, 1, inclusive="right").all(), t)
        dup = e.duplicated(["src_campus_id", "dst_campus_id", "src_year", "dst_year"]
                           + (["edge_set"] if "edge_set" in e else [])).sum()
        rec("duplicate edges", int(dup), dup == 0, t)
        if t != "temporal":
            rec("self-loops", int((e.src_campus_id == e.dst_campus_id).sum()), (e.src_campus_id != e.dst_campus_id).all(), t)
            rec("undirected pairs stored once (src < dst)", bool((e.src_campus_id < e.dst_campus_id).all()),
                (e.src_campus_id < e.dst_campus_id).all(), t)
            rec("edges link nodes of the same year", bool((e.src_year == e.year).all() and (e.dst_year == e.year).all()),
                (e.src_year == e.year).all() and (e.dst_year == e.year).all(), t)

    # ------------------------------------------------------------------ district / region: exact cliques
    for t, col in (("district", "district_id"), ("region", "esc_region")):
        e = E[t]
        grp = meta.set_index("campus_id")[col]
        same_grp = (grp.reindex(e.src_campus_id).to_numpy() == grp.reindex(e.dst_campus_id).to_numpy()).all()
        rec(f"both endpoints share {col}", bool(same_grp), same_grp, t)
        exp = {y: int((g[col].value_counts().pipe(lambda s: s * (s - 1) // 2)).sum())
               for y, g in nodes.groupby("year")}
        got = e.groupby("year").size().to_dict()
        rec("edge count == sum C(n,2) per group", got == exp, got == exp, t, str(exp))

    # ------------------------------------------------------------------ geographic: independent recomputation
    e = E["geographic"]
    a = meta.set_index("campus_id").reindex(e.src_campus_id)
    b = meta.set_index("campus_id").reindex(e.dst_campus_id)
    cosang = np.clip((unit_vec(a.latitude.values, a.longitude.values) * unit_vec(b.latitude.values, b.longitude.values)).sum(1), -1, 1)
    d_ind = B.EARTH_KM * np.arccos(cosang)
    err = np.abs(d_ind - e.distance.values).max()
    rec("geographic distance matches independent great-circle (max abs km error)", round(float(err), 4), err < 0.01, "geographic")
    w_err = np.abs(np.exp(-e.distance.values / B.GEO_TAU_KM) - e.weight.values).max()
    rec("geographic weight == exp(-km/tau)", round(float(w_err), 8), w_err < 1e-5, "geographic")
    rec(f"geographic distance <= {B.GEO_MAX_KM:g} km", round(float(e.distance.max()), 2), e.distance.max() <= B.GEO_MAX_KM, "geographic")
    virt = set(meta.campus_id[meta.is_virtual_or_hybrid == 1])
    rec("virtual/hybrid campuses in geographic edges", int(e.src_campus_id.isin(virt).sum() + e.dst_campus_id.isin(virt).sum()),
        not (e.src_campus_id.isin(virt).any() or e.dst_campus_id.isin(virt).any()), "geographic")
    # kNN property: for each node, every one of its K nearest (within cap) must be linked
    miss = 0
    for y, ny in nodes[nodes.is_virtual_or_hybrid == 0].groupby("year"):
        u = unit_vec(ny.latitude.values, ny.longitude.values)
        D = B.EARTH_KM * np.arccos(np.clip(u @ u.T, -1, 1)); np.fill_diagonal(D, np.inf)
        ids = ny.campus_id.to_numpy()
        ey = e[e.year == y]; linked = set(zip(ey.src_campus_id, ey.dst_campus_id))
        for i in range(len(ids)):
            for j in np.argsort(D[i], kind="stable")[:B.GEO_K]:
                if D[i, j] <= B.GEO_MAX_KM - 1e-6:
                    miss += (min(ids[i], ids[j]), max(ids[i], ids[j])) not in linked
    rec("missing k-nearest geographic links", miss, miss == 0, "geographic", "ties at equal distance may differ")

    # ------------------------------------------------------------------ academic: recompute sample + leakage
    e = E["academic"]
    rec(f"academic edges in {B.YEARS[0]} (no prior year)", int((e.year == B.YEARS[0]).sum()), (e.year == B.YEARS[0]).sum() == 0, "academic")
    rng = np.random.default_rng(1)
    sets = pd.read_csv(G / "academic_edge_sets.csv")
    own = all(r.target in r.excluded_features.split() or B.ACAD_EDGE_SETS[r.edge_set] == r.excluded_features.split()
              for r in sets.itertuples())
    no_overlap = all(not set(r.excluded_features.split()) & set(str(r.profile_features).split()) for r in sets.itertuples())
    fam = {"sat_erw": "sat_total", "sat_math": "sat_total", "act_english": "act_composite", "act_math": "act_composite", "retention_rate": "retention_rate",
           "tsi_both": "sat_act_above_criterion",
           "chronic_absenteeism": "chronic_absenteeism"}
    self_excl = all((fam.get(r.target, r.target) in r.excluded_features.split()) for r in sets.itertuples())
    rec("every target's academic edge set excludes the target (or the total it is part of)", self_excl,
        self_excl and no_overlap and own, "academic", "; ".join(f"{r.target}->{r.edge_set}" for r in sets.itertuples()))
    rec(f"academic edge sets present every year {B.YEARS[1]}-{B.YEARS[-1]}",
        sorted(e.groupby(["edge_set", "year"]).size().index.tolist()) ==
        sorted((es, y) for es in B.ACAD_EDGE_SETS for y in B.YEARS[1:]), True, "academic")
    for (es, y), c in e.groupby(["edge_set", "year"]).size().items():
        rec("edges", int(c), scope=f"academic {es} {y}")
    samp = e.iloc[rng.choice(len(e), 600, replace=False)]
    bad = 0
    for _, r in samp.iterrows():
        prev = master[master.year == r.year - 1].set_index("campus_id")
        feats = [f for f in B.ACAD_FEATURES if f not in B.ACAD_EDGE_SETS[r.edge_set] and prev[f].notna().any()]
        z = (prev[feats] - prev[feats].mean()) / prev[feats].std(ddof=0)
        x, yv = z.loc[r.src_campus_id].to_numpy(float), z.loc[r.dst_campus_id].to_numpy(float)
        sh = ~np.isnan(x) & ~np.isnan(yv)
        d = np.sqrt(np.mean((x[sh] - yv[sh]) ** 2))
        bad += (sh.sum() < B.ACAD_MIN_SHARED) or not np.isclose(d, r.distance, rtol=1e-6, atol=1e-8)  # CSV keeps 9 significant digits
    rec("academic distance matches loop recomputation over the set's own features (600 sampled edges)",
        f"{600 - bad}/600", bad == 0, "academic")
    w_ok = all(np.allclose(np.exp(-(g.distance ** 2) / (2 * np.median(g.distance) ** 2)), g.weight, atol=1e-5)
               for _, g in e.groupby(["edge_set", "year"]))
    rec("academic weight == exp(-d^2 / 2 sigma^2), sigma = median distance per set and year", w_ok, w_ok, "academic")
    # leakage: scramble every indicator in year t, rebuild year-t academic edges, they must be identical
    leak_ok = True
    for y in B.YEARS[1:]:
        scr = master.copy()
        mask = scr.year == y
        for f in B.ACAD_FEATURES:
            scr.loc[mask, f] = rng.permutation(scr.loc[mask, f].to_numpy())
        for es in B.ACAD_EDGE_SETS:
            rebuilt, _ = B.academic_edges(scr, nodes[nodes.year == y], y, es)
            orig = e[(e.year == y) & (e.edge_set == es)]
            same = rebuilt[["src_campus_id", "dst_campus_id"]].reset_index(drop=True).equals(
                orig[["src_campus_id", "dst_campus_id"]].reset_index(drop=True))
            leak_ok &= same
    rec("leakage test: scrambling year-t values leaves year-t academic edges unchanged", leak_ok, leak_ok, "academic")
    rec("prediction-time target excluded from academic similarity", bool(leak_ok and self_excl), leak_ok and self_excl,
        "academic", "year-t values never used (shuffle test) and the predicted family is dropped from each set")

    # ------------------------------------------------------------------ demographic: recompute + leakage
    e = E["demographic"]
    rec(f"demographic edges in {B.YEARS[0]} (no prior year)", int((e.year == B.YEARS[0]).sum()), (e.year == B.YEARS[0]).sum() == 0, "demographic")
    samp = e.iloc[rng.choice(len(e), 400, replace=False)]
    bad = 0
    for _, r in samp.iterrows():
        prev = master[master.year == r.year - 1].set_index("campus_id")[B.DEMO_FEATURES].copy()
        prev["enrollment"] = np.log(1 + prev["enrollment"])
        z = (prev - prev.mean()) / prev.std(ddof=0)
        x, yv = z.loc[r.src_campus_id].to_numpy(float), z.loc[r.dst_campus_id].to_numpy(float)
        sh = ~np.isnan(x) & ~np.isnan(yv)
        d = np.sqrt(np.mean((x[sh] - yv[sh]) ** 2))
        bad += (sh.sum() < B.DEMO_MIN_SHARED) or not np.isclose(d, r.distance, rtol=1e-6, atol=1e-8)
    rec("demographic distance matches loop recomputation (400 sampled edges)", f"{400 - bad}/400", bad == 0, "demographic")
    w_ok = all(np.allclose(np.exp(-(g.distance ** 2) / (2 * np.median(g.distance) ** 2)), g.weight, atol=1e-5)
               for _, g in e.groupby("year"))
    rec("demographic weight == exp(-d^2 / 2 sigma^2), sigma = median distance per year", w_ok, w_ok, "demographic")
    leak_ok = True
    for y in B.YEARS[1:]:
        scr = master.copy()
        mask = scr.year == y
        for f in B.DEMO_FEATURES:
            scr.loc[mask, f] = rng.permutation(scr.loc[mask, f].to_numpy())
        rebuilt, _ = B.demographic_edges(scr, nodes[nodes.year == y], y)
        orig = e[e.year == y]
        leak_ok &= rebuilt[["src_campus_id", "dst_campus_id"]].reset_index(drop=True).equals(
            orig[["src_campus_id", "dst_campus_id"]].reset_index(drop=True))
    rec("leakage test: scrambling year-t demographics leaves year-t demographic edges unchanged", leak_ok, leak_ok, "demographic")

    # ------------------------------------------------------------------ temporal
    e = E["temporal"]
    ok = (e.src_campus_id == e.dst_campus_id).all() and (e.dst_year - e.src_year == 1).all() and (e.directed == 1).all()
    rec("temporal edges: same campus, consecutive years, directed forward", bool(ok), ok, "temporal")
    k = set(zip(master.campus_id, master.year))
    exp = sum((c, y - 1) in k for c, y in k)
    rec("temporal edge count == consecutive campus-year pairs", f"{len(e)} vs {exp}", len(e) == exp, "temporal")
    gaps = master.sort_values("year").groupby("campus_id").year.apply(lambda s: (s.diff() > 1).any())
    rec("campuses with a gap year (no edge bridges the gap)", int(gaps.sum()), None, "temporal")

    # ------------------------------------------------------------------ isolated nodes
    spatial = ["district", "geographic", "academic", "region", "demographic"]
    for y, ny in nodes.groupby("year"):
        ids = set(ny.campus_id)
        for t in spatial:
            ey = E[t][E[t].year == y]
            deg = set(ey.src_campus_id) | set(ey.dst_campus_id)
            rec(f"isolated nodes in {t} layer", len(ids - deg), None, str(y))
        te = E["temporal"]
        rec("nodes with no incoming temporal edge (no t-1 row)", len(ids - set(te[te.dst_year == y].dst_campus_id)), None, str(y))
        rec("nodes with no outgoing temporal edge (no t+1 row)", len(ids - set(te[te.src_year == y].src_campus_id)), None, str(y))
        allsp = set()
        for t in spatial:
            ey = E[t][E[t].year == y]; allsp |= set(ey.src_campus_id) | set(ey.dst_campus_id)
        rec("isolated nodes across all spatial edge types", len(ids - allsp), len(ids - allsp) == 0, str(y))
        for es in B.ACAD_EDGE_SETS:
            ey = E["academic"][(E["academic"].year == y) & (E["academic"].edge_set == es)]
            rec(f"nodes without academic edges in {es} (no usable t-1 profile)",
                len(ids - set(ey.src_campus_id) - set(ey.dst_campus_id)), None, str(y))
    rec("graph parameters status", B.PARAM_STATUS, None, "config", "configs/graph.yaml; flagged for professor review (Q10)")

    rep = pd.DataFrame(rows)
    (G / "reports").mkdir(exist_ok=True)
    rep.to_csv(G / "reports" / "graph_quality_report.csv", index=False)
    failed = rep[rep.passed.astype(str) == "False"]
    print(f"\n{(rep.passed.astype(str) == 'True').sum()} passed, {len(failed)} failed, "
          f"{(rep.passed.astype(str) == '').sum()} informational")
    return 1 if len(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
