"""S7 post-test ablation report: all STGNN versions x both setups x origins 2022-2024, plus the fixed comparators.

Per origin and target, every model is scored on the same schools (rows present in every model's forecasts and the
tree's). Score = MAE / persistence MAE; macro = mean over targets; seeded models: mean and SD over 5 seeds.
Contrasts fixed before the runs (DECISIONS 2026-11-26), each a paired bootstrap of the macro difference on
seed-averaged forecasts (2000 resamples, seed 0), per origin:
  A. Plan A vs r2 for each STGNN version;
  B. within each setup: v1 vs v0 (graph), v1s4 vs v1 (shared relations), v2 vs v1 (gate), v3 vs v2 (multi-task);
  C. the best STGNN version, chosen by its mean over 2022-2023 only, vs MTGNN, tree, honest bar and S1c bar.
2024 is post-test (S6 was already seen); the table does not change the S6 result.

    .venv-expanded/bin/python src/s7/ablation_report.py [--partial]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/s7"), str(ROOT / "src/s6")]
from ablation_run import ORIGINS, SETUPS, STGNN, done, run_dir  # noqa: E402
from report import bootstrap  # noqa: E402

OUT = ROOT / "results/s7_ablation"
SEEDS = range(5)


def seeded_runs(partial: bool) -> list[tuple[str, str]]:
    runs = [(s, m) for s in SETUPS for m in STGNN] + [("r2", "mtgnn")]
    missing = [(s, m, o) for s, m in runs for o in ORIGINS if not done(s, m, o)]
    if missing and not partial:
        raise SystemExit(f"{len(missing)} runs are not complete, e.g. {missing[:3]}; use --partial to inspect")
    return [(s, m) for s, m in runs if all(done(s, m, o) for o in ORIGINS)]


def load_preds(setup: str, model: str, origin: int) -> pd.DataFrame:
    d = run_dir(setup, model, origin) / "predictions"
    frames = [pd.read_csv(p, dtype={"campus_id": str}).assign(seed=int(p.stem.rsplit("_seed_", 1)[1]))
              for p in sorted(d.glob("*_seed_*.csv"))]
    out = pd.concat(frames, ignore_index=True)
    if set(out.year) != {origin} or out.target.nunique() != 10 or out.groupby("target").seed.nunique().ne(5).any():
        raise SystemExit(f"{setup}/{model}/{origin}: expected 10 targets x 5 seeds for {origin}")
    return out


def tree_preds(origin: int) -> pd.DataFrame:
    if origin == 2024:
        return pd.read_csv(ROOT / "results/s6/tree/predictions.csv", dtype={"campus_id": str})
    t = pd.read_csv(ROOT / "results/phase_a/rolling_residuals.csv", dtype={"campus_id": str}).query("year == @origin")
    return t.rename(columns={"y": "y_true", "persistence": "persistence_pred", "tree": "y_pred"})


def bar_picks(origin: int) -> dict:
    if origin == 2024:
        return json.loads((ROOT / "configs/s6_lock.json").read_text())["models"]["honest_bar"]["picks"]
    r = pd.read_csv(ROOT / "results/phase_a/rolling_scores.csv").query("origin < @origin and origin >= 2021")
    m = r.groupby("target")[["tree", "shrink_to_own_3yr"]].mean()
    return {t: "tree" if row.tree < row.shrink_to_own_3yr else "shrinkage" for t, row in m.iterrows()}


def s1c_preds(origin: int) -> pd.DataFrame | None:
    path = {2023: ROOT / "results/s6_rehearsal/s1c_bar/predictions.csv",
            2024: ROOT / "results/s6/s1c_bar/predictions.csv"}.get(origin)
    return pd.read_csv(path, dtype={"campus_id": str}) if path and path.exists() else None


def origin_tables(origin: int, runs: list[tuple[str, str]]):
    preds = {f"{s}:{m}": load_preds(s, m, origin) for s, m in runs}
    shrink_src = load_preds("planA", "v0", origin).query("seed == 0")
    tree, picks, s1c = tree_preds(origin), bar_picks(origin), s1c_preds(origin)
    rows, aligned = [], {}
    for target in sorted(shrink_src.target.unique()):
        ids = pd.Index(tree.query("target == @target").campus_id)
        for f in preds.values():
            ids = ids.intersection(f.query("target == @target and seed == 0").campus_id)
        ids = ids.sort_values()
        ref = shrink_src.query("target == @target").set_index("campus_id").loc[ids]
        y, lag = ref.y_true.to_numpy(), ref.persistence_pred.to_numpy()
        t = tree.query("target == @target").set_index("campus_id").loc[ids]
        if not (np.allclose(t.y_true, y) and np.allclose(t.persistence_pred, lag)):
            raise SystemExit(f"{origin}/{target}: tree rows disagree")
        pers = np.abs(lag - y).mean()
        fixed = {"persistence": lag, "shrinkage": ref.anchor_pred.to_numpy(), "tree": t.y_pred.to_numpy()}
        fixed["honest_bar"] = fixed["tree" if picks[target] == "tree" else "shrinkage"]
        if s1c is not None:
            sc = s1c[s1c.target == target]
            if (sc.model == "persistence").all():
                fixed["s1c_bar"] = lag
            else:
                sc = sc.groupby("campus_id").y_pred.mean()
                common = ids.intersection(sc.index)
                if len(common) != len(ids):
                    raise SystemExit(f"{origin}/{target}: S1c rows do not cover the common rows")
                fixed["s1c_bar"] = sc.loc[ids].to_numpy()
        for name, p in fixed.items():
            ratio = np.abs(p - y).mean() / pers
            if name == "s1c_bar" and not (s1c[s1c.target == target].model == "persistence").all():
                # as in the S6 report: mean of the per-seed scores (the seed-averaged forecast is used for bootstraps)
                sw = s1c[s1c.target == target].pivot_table(index="campus_id", columns="seed", values="y_pred").loc[ids]
                ratio = float(np.mean([np.abs(sw[k].to_numpy() - y).mean() / pers for k in sw.columns]))
            rows.append({"origin": origin, "target": target, "model": name, "n": len(ids), "ratio": ratio})
        seeded = {}
        for name, f in preds.items():
            wide = f[f.target == target].pivot_table(index="campus_id", columns="seed", values="y_pred").loc[ids]
            truth = f[(f.target == target) & (f.seed == 0)].set_index("campus_id").y_true.loc[ids]
            if not np.allclose(truth, y):
                raise SystemExit(f"{origin}/{target}/{name}: truths disagree")
            per = [np.abs(wide[k].to_numpy() - y).mean() / pers for k in SEEDS]
            rows += [{"origin": origin, "target": target, "model": name, "n": len(ids), "seed": k, "ratio": v}
                     for k, v in zip(SEEDS, per)]
            seeded[name] = wide.mean(axis=1).to_numpy()
        aligned[target] = {"y": y, "pers": pers, **fixed, **seeded}
    return pd.DataFrame(rows), aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="report only the complete setup/model runs")
    args = ap.parse_args()
    runs = seeded_runs(args.partial)
    long, aligned = [], {}
    for origin in ORIGINS:
        frame, aligned[origin] = origin_tables(origin, runs)
        long.append(frame)
    long = pd.concat(long, ignore_index=True)
    seeded = long[long.seed.notna()]
    macro_seed = seeded.groupby(["origin", "model", "seed"]).ratio.mean()
    summary = long.groupby(["origin", "model", "target"]).ratio.mean().groupby(["origin", "model"]).mean().unstack(0)
    summary.columns = [f"y{c}" for c in summary.columns]
    sd = macro_seed.groupby(["origin", "model"]).std().unstack(0)
    for c in sd.columns:
        summary[f"sd{c}"] = sd[c]
    summary["mean_2022_2023"] = summary[["y2022", "y2023"]].mean(axis=1)
    summary["mean_2022_2024"] = summary[["y2022", "y2023", "y2024"]].mean(axis=1)
    summary = summary.sort_values("mean_2022_2024").reset_index().rename(columns={"index": "model"})

    names = {f"{s}:{m}" for s, m in runs}
    pairs = [(f"planA:{m}", f"r2:{m}", "A: Plan A vs r2") for m in STGNN]
    for s in SETUPS:
        pairs += [(f"{s}:v1", f"{s}:v0", "B: graph"), (f"{s}:v1s4", f"{s}:v1", "B: shared relations"),
                  (f"{s}:v2", f"{s}:v1", "B: gate"), (f"{s}:v3", f"{s}:v2", "B: multi-task")]
    stgnn = summary[summary.model.isin([n for n in names if not n.endswith("mtgnn")])]
    best = stgnn.sort_values("mean_2022_2023").model.iloc[0] if not stgnn.empty else None
    if best:
        pairs += [(best, b, "C: best STGNN (chosen on 2022-2023)") for b in ("r2:mtgnn", "tree", "honest_bar", "s1c_bar")]
    contrasts = []
    for a, b, family in pairs:
        for origin in ORIGINS:
            first = next(iter(aligned[origin].values()))
            if a not in first or b not in first:
                continue
            _, m = bootstrap(aligned[origin], a, b, 2000, 0)
            contrasts.append({"family": family, "origin": origin, **m})
    contrasts = pd.DataFrame(contrasts)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in (("per_target", long), ("summary", summary), ("contrasts", contrasts)):
        frame.to_csv(OUT / f"ablation_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"best STGNN version by mean over 2022-2023: {best}\n")
    print(summary.round(3).to_string(index=False))
    print("\n" + contrasts.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
