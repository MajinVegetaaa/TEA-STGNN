"""Step 4 report: rolling-origin comparison on identical rows (origins 2022 and 2023), with the pre-registered claims.

Per origin and target, on the STGNN prediction rows that also have a Phase A tree forecast:
persistence (1.0 by construction), shrinkage anchor (fitted on years < origin), fixed tree, honest bar (tree or
shrinkage chosen per target on earlier origins only), V0 and V1 STGNN (shrinkage anchor, nested selection; mean
over 5 seeds). The S1c bar (picked on 2023) is shown for 2023 only. Writes results/step4/step4_*.csv.

    .venv-expanded/bin/python src/eval/report_step4.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/step4"
ORIGINS, SEEDS = [2022, 2023], range(5)
EARLIER = {2022: [2021], 2023: [2021, 2022]}


def mae_ratio(pred, y, lag) -> float:
    return float(np.abs(pred - y).mean() / np.abs(lag - y).mean())


def main() -> None:
    tree = pd.read_csv(ROOT / "results/phase_a/rolling_residuals.csv", dtype={"campus_id": str})
    phase_a = pd.read_csv(ROOT / "results/phase_a/rolling_scores.csv")
    s1c = pd.read_csv(ROOT / "results/phase_a/shift_decomposition.csv")
    s1c = s1c[s1c.model.str.startswith("TABULAR_BAR")].set_index("target").ratio
    rows, seed_rows = [], []
    for s in ORIGINS:
        for m in ("v0", "v1"):
            status = json.loads((OUT / m / f"origin_{s}" / "status.json").read_text())
            if status["status"] != "completed":
                raise ValueError(f"{m} origin {s} not completed")
        for target in sorted({p.stem.rsplit("_seed_", 1)[0] for p in (OUT / "v0" / f"origin_{s}" / "predictions").glob("*.csv")}):
            preds = {(m, k): pd.read_csv(OUT / m / f"origin_{s}" / "predictions" / f"{target}_seed_{k}.csv",
                                         dtype={"campus_id": str}).set_index("campus_id") for m in ("v0", "v1") for k in SEEDS}
            ref = preds[("v0", 0)]
            t = tree[(tree.year == s) & (tree.target == target)].set_index("campus_id")
            ids = ref.index.intersection(t.index)
            r = ref.loc[ids]
            if not np.allclose(t.loc[ids, "y"], r.y_true) or not np.allclose(t.loc[ids, "persistence"], r.persistence_pred):
                raise ValueError(f"{target} {s}: tree rows disagree with STGNN rows")
            y, lag = r.y_true.to_numpy(), r.persistence_pred.to_numpy()
            score = {"shrinkage": mae_ratio(r.anchor_pred, y, lag), "tree": mae_ratio(t.loc[ids, "tree"], y, lag)}
            hist = phase_a[(phase_a.target == target) & phase_a.origin.isin(EARLIER[s])]
            pick = min(("tree", "shrink_to_own_3yr"), key=lambda c: hist[c].mean())
            score["honest_bar"] = score["tree" if pick == "tree" else "shrinkage"]
            for m in ("v0", "v1"):
                per = [mae_ratio(preds[(m, k)].loc[ids, "y_pred"], y, lag) for k in SEEDS]
                score[m] = float(np.mean(per))
                seed_rows += [{"origin": s, "model": m, "target": target, "seed": k, "ratio": v} for k, v in zip(SEEDS, per)]
            if s == 2023:
                score["s1c_bar_selected_on_2023"] = float(s1c[target])
            rows.append({"origin": s, "target": target, "n": len(ids), "n_stgnn_rows": len(ref),
                         "bar_pick": pick, **score})
    per_target = pd.DataFrame(rows)
    seeds = pd.DataFrame(seed_rows)
    macro_seed = seeds.groupby(["origin", "model", "seed"]).ratio.mean().groupby(["origin", "model"]).agg(["mean", "std"])
    cols = ["shrinkage", "tree", "honest_bar", "v0", "v1", "s1c_bar_selected_on_2023"]
    summary = per_target.groupby("origin")[cols].mean()
    summary["v0_sd"] = macro_seed.xs("v0", level="model")["std"]
    summary["v1_sd"] = macro_seed.xs("v1", level="model")["std"]
    summary = summary.reset_index()
    beats_bar = bool((summary.v1 < summary.honest_bar).all())
    graph = summary.v1 - summary.v0
    graph_helps = bool((graph < -summary[["v0_sd", "v1_sd"]].max(axis=1)).all())
    step1_v1 = pd.read_csv(ROOT / "results/s5_round2/step1/step1_summary.csv").query("model == 'v1' and arm == 'E_shrink_anchor'").mean_ratio.iloc[0]
    claims = pd.DataFrame([{"claim_1_v1_beats_honest_bar_both_origins": beats_bar,
                            "claim_2_graph_helps_both_origins": graph_helps,
                            "v1_minus_v0_2022": graph.iloc[0], "v1_minus_v0_2023": graph.iloc[1],
                            "selection_optimism_2023": float(summary.v1.iloc[1] - step1_v1),
                            "step1_v1_tuned_on_2023": step1_v1}])
    for name, f in (("per_target", per_target), ("summary", summary), ("claims", claims)):
        f.to_csv(OUT / f"step4_{name}.csv", index=False)
        print(f"\n== {name}\n{f.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
