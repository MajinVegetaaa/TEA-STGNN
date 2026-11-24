"""Phase A1: split each model's 2023 error into a statewide shift (bias) and a per-school part (spread).

Question: when a model beats persistence on a target, is that because it forecast the common year-to-year shift
(one number per target and year), or because it ranked and spaced individual schools better? Only the second is
evidence that peer or school-level information helps.

For each target and model, on the identical 2023 rows (target and last year's value observed):
  ratio           model MAE / persistence MAE
  bias            mean(prediction - truth)
  spread_mae      mean |error - bias|  (per-school error once the common shift is removed)
  shifted_pers    MAE of (last year's value + the model's own average predicted change) / persistence MAE
  oracle_shift    MAE of (last year's value + the true average change) / persistence MAE: the best any pure
                  shift could do, i.e. how much of the target is a common shift at all
Seeds are scored separately and averaged.

Reads only 2023 validation predictions: S1c bar picks (rows with split == "val") and the S5 Round 2 runs. Nothing
from 2024 is used or printed. Writes results/phase_a/shift_decomposition.csv.

    .venv-expanded/bin/python src/eval/phase_a_shift.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase_a"
S5_RUNS = {"V0": "v0_temporal_control", "V1_original": "v1_original_attention",
           "V1_query": "v1_query_attention", "V1_shared4_single": "v3_single_shared_relations",
           "V2_original": "v2_original_gate", "V2_positive": "v2_positive_gate",
           "V1_shared4_multi": "v3_multi_shared_relations", "MTGNN": "mtgnn_comparator"}
VAL_YEAR = 2023


def decompose(y: np.ndarray, lag: np.ndarray, pred: np.ndarray) -> dict:
    """Bias/spread decomposition of one prediction vector against persistence on the same rows."""
    y, lag, pred = (np.asarray(a, dtype=float) for a in (y, lag, pred))
    if not (len(y) == len(lag) == len(pred)) or len(y) == 0:
        raise ValueError("y, lag and pred must be non-empty and the same length")
    pers = np.abs(lag - y).mean()
    if not np.isfinite(pers) or pers <= 0:
        raise ValueError("persistence MAE must be positive")
    err = pred - y
    bias = err.mean()
    return {"ratio": np.abs(err).mean() / pers, "bias": bias, "spread_mae": np.abs(err - bias).mean(),
            "pers_bias": (lag - y).mean(), "pers_spread_mae": np.abs((lag - y) - (lag - y).mean()).mean(),
            "shifted_pers": np.abs(lag + (pred - lag).mean() - y).mean() / pers,
            "oracle_shift": np.abs(lag + (y - lag).mean() - y).mean() / pers, "n": len(y)}


def s1c_val_predictions(path: Path, model: str, target: str) -> pd.DataFrame:
    """2023 validation rows of one S1c model and target; test rows are filtered while reading."""
    keep = []
    for chunk in pd.read_csv(path, dtype={"campus_id": str}, chunksize=500_000):
        chunk = chunk[(chunk.split == "val") & (chunk.model == model) & (chunk.target == target)]
        keep.append(chunk)
    out = pd.concat(keep, ignore_index=True)
    if out.empty or set(out.year) != {VAL_YEAR}:
        raise ValueError(f"no 2023 validation rows for {model} / {target} in {path}")
    return out


def main() -> None:
    picks = json.loads((ROOT / "results/s1c/bar.json").read_text())["picks"]
    rows = []
    for target, pick in picks.items():
        # persistence rows (and lag) come from the S5 V0 run, which shares the exact 2023 rows with S1c
        base = {k: pd.read_csv(ROOT / f"results/s5_round2/{S5_RUNS['V0']}/predictions/{target}_seed_{k}.csv",
                               dtype={"campus_id": str}).set_index("campus_id") for k in range(5)}
        y, lag = base[0].y_true, base[0].persistence_pred
        rows.append({"target": target, "model": "persistence", **decompose(y, lag, lag)})
        model, _, rest = pick.partition(" | ")
        if model.startswith("persistence"):
            rows.append({"target": target, "model": "TABULAR_BAR (= persistence)", **decompose(y, lag, lag)})
        else:
            mode, input_set = rest.split(" | ")
            tab = s1c_val_predictions(ROOT / f"results/s1c/{input_set}/{mode}/predictions.csv", model, target)
            per_seed = []
            for _, g in tab.groupby("seed"):
                p = g.set_index("campus_id").y_pred.reindex(y.index)
                if p.isna().any():
                    raise ValueError(f"S1c rows differ from S5 rows for {target}")
                per_seed.append(decompose(y, lag, p))
            rows.append({"target": target, "model": f"TABULAR_BAR ({pick})",
                         **pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()})
        for name, folder in S5_RUNS.items():
            per_seed = []
            for k in range(5):
                pred_dir = ROOT / f"results/s5_round2/{folder}/predictions"
                single = pred_dir / f"{target}_seed_{k}.csv"
                if single.exists():
                    f = pd.read_csv(single, dtype={"campus_id": str})
                else:   # multi-task runs write every target into one file per seed
                    f = pd.read_csv(pred_dir / f"all_targets_seed_{k}.csv", dtype={"campus_id": str})
                    f = f[f.target == target]
                f = f.set_index("campus_id")
                if set(f.year) != {VAL_YEAR} or not f.index.sort_values().equals(y.index.sort_values()):
                    raise ValueError(f"{name}/{target}/seed {k}: rows differ from the reference rows")
                per_seed.append(decompose(f.y_true.reindex(y.index), lag, f.y_pred.reindex(y.index)))
            rows.append({"target": target, "model": name, **pd.DataFrame(per_seed).mean(numeric_only=True).to_dict()})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "shift_decomposition.csv", index=False)
    show = table[table.model.isin(["persistence", "V0", "V1_original", "V1_shared4_single", "MTGNN"])
                 | table.model.str.startswith("TABULAR_BAR")].copy()
    show["model"] = show.model.str.replace(r"TABULAR_BAR \(.*\)", "TABULAR_BAR", regex=True)
    cols = ["ratio", "bias", "spread_mae", "shifted_pers", "oracle_shift"]
    for target, g in show.groupby("target", sort=False):
        print(f"\n== {target} (n={int(g.n.iloc[0])}, oracle pure-shift ratio {g.oracle_shift.iloc[0]:.3f})")
        print(g.set_index("model")[cols[:-1]].round(3).to_string())


if __name__ == "__main__":
    main()
