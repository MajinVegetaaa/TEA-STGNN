"""S1c: the numbers to beat. For each target, pick the best candidate on 2023 only -- every tabular model (input set x
target version x model) AND persistence itself -- then report it on 2024 once, with a paired bootstrap against
persistence (configs/protocol.yaml, selection.bar). The best non-persistence model is reported alongside.

Reads results/s1c/{existing,all}/{level,relative,zscore}/predictions.csv. Writes results/s1c/:
  candidates.csv        every candidate's seed-mean validation and test MAE (the full picture)
  numbers_to_beat.csv   per target: the pick, its 2023 and 2024 scores, persistence, bootstrap CI
  bar.json              the final-model score of the bar on 2023 (mean of MAE / persistence MAE), what S5 must beat
Run from the repo root:  python3 src/eval/numbers_to_beat.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from protocol import PERSIST, load, metrics, paired_bootstrap, selection_score

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "s1c"
SETS, MODES = ["existing", "all"], ["level", "relative", "zscore"]


def read(input_set: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / input_set / mode / "predictions.csv", dtype={"campus_id": str})


def main() -> None:
    cfg = load()
    per_seed = []
    for s in SETS:
        for m in MODES:
            p = read(s, m)
            g = p.groupby(["model", "target", "split", "seed"]).apply(metrics, include_groups=False).reset_index()
            per_seed.append(g.assign(input_set=s, mode=m))
    per_seed = pd.concat(per_seed, ignore_index=True)
    key = ["input_set", "mode", "model", "target", "split"]
    cand = per_seed.groupby(key)[["mae", "rmse", "r2"]].mean().reset_index()
    wide = cand.pivot_table(index=["input_set", "mode", "model", "target"], columns="split", values="mae").reset_index()
    wide.rename(columns={"val": "val_mae", "test": "test_mae"}).to_csv(RUNS / "candidates.csv", index=False)

    rows, bar = [], {}
    for t in cfg["targets"]:
        c = cand[(cand.target == t) & (cand.split == "val")]
        tab = c[c.model != PERSIST].loc[lambda d: d.mae.idxmin()]           # best tabular model
        best = c.loc[c.mae.idxmin()]                                      # persistence counts as a candidate
        pick = (best.input_set, best["mode"], best.model)
        p = read(best.input_set, best["mode"])
        p = p[p.target == t]
        pers_val = cand[(cand.target == t) & (cand.model == PERSIST) & (cand.split == "val")].mae.iloc[0]
        test = p[p.split == "test"]
        bs = paired_bootstrap(test[test.model == PERSIST], test[test.model == best.model],
                              n_boot=cfg["significance"]["n_boot"], seed=cfg["significance"]["seed"])
        tm = test[test.model == best.model].groupby("seed").apply(metrics, include_groups=False).mean()
        pm = test[test.model == PERSIST].groupby("seed").apply(metrics, include_groups=False).mean()
        tab_test = cand[(cand.target == t) & (cand.split == "test") & (cand.input_set == tab.input_set)
                        & (cand["mode"] == tab["mode"]) & (cand.model == tab.model)].mae.iloc[0]
        is_pers = pick[2] == PERSIST                      # persistence is the same in every input set and version
        rows.append({"target": t, "input_set": "n/a" if is_pers else pick[0], "mode": "n/a" if is_pers else pick[1],
                     "model": pick[2], "bar_is_persistence": is_pers,
                     "best_tabular": f"{tab.model} | {tab['mode']} | {tab.input_set}",
                     "best_tabular_val_ratio": tab.mae / pers_val,
                     "best_tabular_test_mae": tab_test,
                     "val_mae_persistence": pers_val, "val_mae_model": best.mae,
                     "val_ratio": best.mae / pers_val,
                     "test_mae_persistence": pm.mae, "test_mae_model": tm.mae, "test_r2_model": tm.r2,
                     "test_rmse_model": tm.rmse, "pct_change_vs_persistence": bs["pct_change"],
                     "ci_low": bs["ci_low"], "ci_high": bs["ci_high"], "beats_persistence": bool(bs["ci_high"] < 0),
                     "n_test": bs["n"]})
        seeds = per_seed[(per_seed.target == t) & (per_seed.split == "val") & (per_seed.input_set == pick[0])
                         & (per_seed["mode"] == pick[1]) & (per_seed.model == pick[2])].set_index("seed").mae
        bar[t] = {"seeds": seeds.sort_index().tolist(), "persistence": pers_val}
    out = pd.DataFrame(rows)
    out.to_csv(RUNS / "numbers_to_beat.csv", index=False)

    model_mae = pd.DataFrame({t: v["seeds"] for t, v in bar.items()})
    score, sd = selection_score(model_mae, pd.Series({t: v["persistence"] for t, v in bar.items()}))
    (RUNS / "bar.json").write_text(json.dumps({"score_2023": score, "sd_over_seeds": sd,
                                               "meaning": "mean over the 10 targets of (MAE of the 2023 pick, persistence "
                                                          "included as a candidate) / persistence MAE, on 2023; "
                                                          "S4/S5 models are compared with this",
                                               "picks": {r["target"]: f'{r["model"]} | {r["mode"]} | {r["input_set"]}'
                                                         for r in rows}}, indent=2) + "\n")
    pd.set_option("display.width", 250)
    print(out[["target", "input_set", "mode", "model", "val_ratio", "test_mae_persistence", "test_mae_model",
               "pct_change_vs_persistence", "beats_persistence", "best_tabular_val_ratio"]].round(3).to_string(index=False))
    print(f"\nBar on 2023 (10 targets): {score:.4f} +/- {sd:.4f}  (1.0 = persistence)")


if __name__ == "__main__":
    main()
