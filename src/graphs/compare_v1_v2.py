"""S1b: what graph v2 changes relative to v1 (data/graph_expanded/ vs data/graph_v2/). Writes results/s1b/.

  edges_identical.csv     per relation: same (src, dst, year) edge set in both graphs? (v2 only re-weights)
  geographic_weights.csv  mean geographic edge weight by locale, v1 vs v2
  similarity_weights.csv  median academic/demographic weight per year, v1 vs v2
Run from the repo root:  python3 src/graphs/compare_v1_v2.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V1, V2, OUT = ROOT / "data/graph_expanded", ROOT / "data/graph_v2", ROOT / "results/s1b"
ID = {"src_campus_id": str, "dst_campus_id": str, "campus_id": str}
KEY = ["src_campus_id", "dst_campus_id", "year"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    same = []
    for rel in ["district", "region", "geographic", "academic", "demographic", "temporal", "transition"]:
        a, b = (pd.read_csv(g / f"edges_{rel}.csv", dtype=ID) for g in (V1, V2))
        key = KEY + (["edge_set"] if rel == "academic" else [])
        identical = a[key].reset_index(drop=True).equals(b[key].reset_index(drop=True))
        weights_equal = a.weight.reset_index(drop=True).equals(b.weight.reset_index(drop=True))
        same.append({"relation": rel, "edges": len(a), "same_edges": identical, "same_weights": weights_equal})
    pd.DataFrame(same).to_csv(OUT / "edges_identical.csv", index=False)
    print(pd.DataFrame(same).to_string(index=False))

    nodes = pd.read_csv(V2 / "nodes.csv", dtype=ID, usecols=["campus_id", "year", "urbanicity"])
    geo = []
    for name, g in (("v1", V1), ("v2", V2)):
        e = pd.read_csv(g / "edges_geographic.csv", dtype=ID)
        both = pd.concat([e.rename(columns={"src_campus_id": "campus_id"}), e.rename(columns={"dst_campus_id": "campus_id"})])
        both = both.merge(nodes, on=["campus_id", "year"])
        geo.append(both.groupby("urbanicity").weight.mean().rename(name))
    geo = pd.concat(geo, axis=1).round(3)
    geo.to_csv(OUT / "geographic_weights.csv")
    print("\nmean geographic edge weight by locale\n", geo.to_string())

    sim = []
    for rel in ("academic", "demographic"):
        for name, g in (("v1", V1), ("v2", V2)):
            e = pd.read_csv(g / f"edges_{rel}.csv", dtype=ID)
            sim.append(e.groupby("year").weight.median().rename(f"{rel}_{name}"))
    sim = pd.concat(sim, axis=1).round(3)
    sim.to_csv(OUT / "similarity_weights.csv")
    print("\nmedian similarity weight per year (v1 re-fits sigma each year, so it is ~0.61 by construction)\n",
          sim.to_string())


if __name__ == "__main__":
    main()
