"""S6 step 3: the 2024 test table, written once after every part of src/s6/run.py is complete.

Per target, every model is scored on the same schools (2024 target and its 2023 value observed, present in every
model's predictions). Score = MAE / persistence MAE; macro = mean over the 10 targets (seeded models: mean and SD
over the 5 seeds). Paired bootstrap (seed-averaged predictions, identical schools, 2000 resamples, seed 0) for the
comparisons fixed in configs/s6_lock.json, per target and for the macro score.

    .venv-expanded/bin/python src/s6/report.py              # results/s6/s6_*.csv
    .venv-expanded/bin/python src/s6/report.py --rehearse   # results/s6_rehearsal/ (2023, settings chosen on 2022)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDED = ["v0", "v1", "mtgnn"]
SEEDS = range(5)


def load(root: Path, year: int, with_s1c: bool) -> dict[str, pd.DataFrame]:
    """Long frames (target, campus_id, seed, y_true, y_pred) per model, plus persistence and shrinkage."""
    out = {}
    for model in SEEDED:
        d = root / "neural" / model / f"origin_{year}"
        if json.loads((d / "status.json").read_text())["status"] != "completed":
            raise SystemExit(f"{model} is not complete")
        frames = [pd.read_csv(p, dtype={"campus_id": str}).assign(seed=int(p.stem.rsplit("_seed_", 1)[1]))
                  for p in sorted((d / "predictions").glob("*_seed_*.csv"))]
        out[model] = pd.concat(frames, ignore_index=True)
        if out[model].groupby("target").seed.nunique().ne(len(SEEDS)).any() or out[model].target.nunique() != 10:
            raise SystemExit(f"{model}: expected 10 targets x 5 seeds")
    base = out["v0"][out["v0"].seed == 0]
    out["persistence"] = base.assign(y_pred=base.persistence_pred)
    out["shrinkage"] = base.assign(y_pred=base.anchor_pred)
    tree = pd.read_csv(root / "tree" / "predictions.csv", dtype={"campus_id": str})
    out["tree"] = tree.assign(seed=0)
    if with_s1c:
        s1c = pd.read_csv(root / "s1c_bar" / "predictions.csv", dtype={"campus_id": str})
        out["s1c_bar"] = s1c[s1c.model != "persistence"]
        out["_s1c_persistence_targets"] = s1c[s1c.model == "persistence"]
    return out


def common_rows(frames: dict, target: str, lock: dict, with_s1c: bool) -> pd.Index:
    ids = None
    for name, f in frames.items():
        if name.startswith("_"):
            continue
        sub = f[f.target == target]
        if name == "s1c_bar" and sub.empty:          # this target's S1c pick is persistence itself
            continue
        cur = pd.Index(sub.campus_id.unique())
        ids = cur if ids is None else ids.intersection(cur)
    return ids.sort_values()


def per_target_tables(frames: dict, lock: dict, with_s1c: bool):
    picks = lock["models"]["honest_bar"]["picks"]
    rows, seed_rows, aligned = [], [], {}
    for target in sorted(frames["v0"].target.unique()):
        ids = common_rows(frames, target, lock, with_s1c)
        ref = frames["persistence"].query("target == @target").set_index("campus_id").loc[ids]
        y, lag = ref.y_true.to_numpy(), ref.persistence_pred.to_numpy()
        tree = frames["tree"].query("target == @target").set_index("campus_id").loc[ids]
        if not (np.allclose(tree.y_true, y) and np.allclose(tree.persistence_pred, lag)):
            raise SystemExit(f"{target}: tree rows disagree with the STGNN rows")
        pers = np.abs(lag - y).mean()
        shrink = frames["shrinkage"].query("target == @target").set_index("campus_id").loc[ids]
        preds = {"persistence": lag, "shrinkage": shrink.y_pred.to_numpy(), "tree": tree.y_pred.to_numpy()}
        preds["honest_bar"] = preds["tree" if picks[target] == "tree" else "shrinkage"]
        seeded = {m: frames[m].query("target == @target") for m in SEEDED}
        if with_s1c:
            s = frames["s1c_bar"].query("target == @target")
            seeded["s1c_bar"] = s if not s.empty else None
        row = {"target": target, "n": len(ids), "persistence_mae": pers, "honest_bar_pick": picks[target]}
        for name in ("persistence", "shrinkage", "tree", "honest_bar"):
            row[name] = np.abs(preds[name] - y).mean() / pers
        for name, f in seeded.items():
            if f is None:                                  # S1c pick = persistence
                preds[name] = lag
                row[name] = 1.0
                continue
            wide = f.pivot_table(index="campus_id", columns="seed", values="y_pred").loc[ids]
            per_seed = [np.abs(wide[k].to_numpy() - y).mean() / pers for k in wide.columns]
            row[name] = float(np.mean(per_seed))
            seed_rows += [{"target": target, "model": name, "seed": int(k), "ratio": v}
                          for k, v in zip(wide.columns, per_seed)]
            preds[name] = wide.mean(axis=1).to_numpy()     # seed-averaged, for the bootstrap
        rows.append(row)
        aligned[target] = {"y": y, "pers": pers, **preds}
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), aligned


def bootstrap(aligned: dict, a: str, b: str, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """(MAE_a - MAE_b) / persistence MAE per target and macro; negative = a is better."""
    rng = np.random.default_rng(seed)
    rows, macro = [], np.zeros(n_boot)
    for target, d in aligned.items():
        ea, eb = np.abs(d[a] - d["y"]), np.abs(d[b] - d["y"])
        idx = rng.integers(0, len(ea), size=(n_boot, len(ea)))
        diffs = (ea[idx].mean(1) - eb[idx].mean(1)) / d["pers"]
        macro += diffs / len(aligned)
        rows.append({"model": a, "versus": b, "target": target, "diff": (ea.mean() - eb.mean()) / d["pers"],
                     "ci_low": np.quantile(diffs, 0.025), "ci_high": np.quantile(diffs, 0.975),
                     "share_a_not_better": float((diffs >= 0).mean())})
    point = float(np.mean([r["diff"] for r in rows]))
    return pd.DataFrame(rows), {"model": a, "versus": b, "macro_diff": point,
                                "ci_low": float(np.quantile(macro, 0.025)), "ci_high": float(np.quantile(macro, 0.975)),
                                "share_a_not_better": float((macro >= 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true")
    args = ap.parse_args()
    lock = json.loads((ROOT / "configs/s6_lock.json").read_text())
    if args.rehearse:   # the honest-bar picks valid for a 2023 forecast are chosen on origins 2021-2022 only
        r = pd.read_csv(ROOT / "results/phase_a/rolling_scores.csv").query("origin in [2021, 2022]")
        m = r.groupby("target")[["tree", "shrink_to_own_3yr"]].mean()
        lock["models"]["honest_bar"]["picks"] = {t: "tree" if row.tree < row.shrink_to_own_3yr else "shrinkage"
                                                 for t, row in m.iterrows()}
    root = ROOT / ("results/s6_rehearsal" if args.rehearse else "results/s6")
    year = 2023 if args.rehearse else lock["test_year"]
    with_s1c = not args.rehearse
    frames = load(root, year, with_s1c)
    per_target, seeds, aligned = per_target_tables(frames, lock, with_s1c)
    models = ["persistence", "shrinkage", "tree", "honest_bar"] + (["s1c_bar"] if with_s1c else []) + SEEDED
    summary = pd.DataFrame({"model": models, "macro_ratio": [per_target[m].mean() for m in models]})
    sd = seeds.groupby(["model", "seed"]).ratio.mean().groupby("model").std()
    summary["sd_over_seeds"] = summary.model.map(sd)
    boot = lock["scoring"]["bootstrap"]
    comps = [c for c in lock["scoring"]["comparisons"] if with_s1c or "s1c_bar" not in c]
    tables = [bootstrap(aligned, a, b, boot["n_boot"], boot["seed"]) for a, b in comps]
    out = {"per_target": per_target, "summary": summary, "per_seed": seeds,
           "bootstrap_per_target": pd.concat([t for t, _ in tables], ignore_index=True),
           "bootstrap_macro": pd.DataFrame([m for _, m in tables])}
    for name, frame in out.items():
        frame.to_csv(root / f"s6_{name}.csv", index=False)
    for name in ("summary", "bootstrap_macro"):
        print(f"\n== {name}\n{out[name].round(4).to_string(index=False)}")
    print(f"\n== per_target\n{per_target.drop(columns=['persistence_mae']).round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
