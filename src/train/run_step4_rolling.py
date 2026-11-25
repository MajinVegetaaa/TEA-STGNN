"""Step 4: rolling-origin STGNN runs with nested selection (no year is used to choose settings and then scored).

For each origin s in ORIGINS and each model (V0 / V1_original, shrinkage anchor as adopted in Step 1):
  1. tuning: train on label years < s-1, choose hidden/dropout/epoch on year s-1 (budget B grid, seed 0, patience);
     the shrinkage anchor is fitted on label years < s-1.
  2. scoring: refit on label years < s with the chosen settings for 5 seeds (anchor fitted on label years < s),
     forecast year s.
Origin 2021 is not run: its tuning fit would have a single label year (2019) and no 3-year average for the anchor.
Uses the 2018-2023 validation bundle only; 2024 is never loaded.

    .venv-expanded/bin/python src/train/run_step4_rolling.py --model v0 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/train"), str(ROOT / "src/models")]
import run_s5_round2 as runner  # noqa: E402
from data_s5_round2 import load_validation_bundle  # noqa: E402

ORIGINS = [2022, 2023]
MODELS = {"v0": ("v0_temporal", "none"), "v1": ("v1_original", "target_specific")}
BUNDLE = ROOT / "data/graph_v2/pyg/s5_validation"
OUT = ROOT / "results/step4"
OUTS = {"change": OUT, "change_relative": ROOT / "results/step4_relative"}   # the second is the post-hoc plan A


def view(panel, through: int):
    """The panel with training years < `through`, so `val_index` points at `through` (later years unused)."""
    v = SimpleNamespace(**{k: getattr(panel, k) for k in ("years", "snapshots", "targets", "feature_names",
                                                         "campus_ids", "fingerprint")})
    v.split = dict(panel.split, train_years=[y for y in panel.years if y < through], val_years=[through])
    v.val_index = panel.years.index(through)
    v.relations = panel.relations
    return v


def anchor_for(panel, y, lag, scaler, through: int):
    anchor, coef = runner.shrinkage_anchor(view(panel, through), y, lag, scaler)
    return anchor, coef


def run(model: str, device: str, origins: list[int], target_mode: str = "change") -> None:
    variant, policy = MODELS[model]
    panel = load_validation_bundle(BUNDLE)
    scaler = json.loads((BUNDLE / "metadata.json").read_text())["scaler"]
    budget = yaml.safe_load((ROOT / "configs/budget.yaml").read_text())
    runner.cfg_model_args = {}
    runner.campus_ids = panel.campus_ids
    y, lag, mask = runner._target_tensors(panel, device)
    grid = [{"hidden": int(h), "dropout": float(d)} for h in budget["grid"]["hidden"] for d in budget["grid"]["dropout"]]
    common = dict(target_mode=target_mode, variant=variant, y=y, lag=lag, eval_mask=mask, lr=budget["lr"],
                  weight_decay=budget["weight_decay"], device=device, gradient_clip_norm=None, config_hash="step4")
    for s in origins:
        out = OUTS[target_mode] / model / f"origin_{s}"
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        tune_panel, score_panel = view(panel, s - 1), view(panel, s)
        tune_anchor, tune_coef = anchor_for(panel, y, lag, scaler, s - 1)
        score_anchor, score_coef = anchor_for(panel, y, lag, scaler, s)
        (out / "anchor_coefficients.json").write_text(json.dumps({"tune": tune_coef, "score": score_coef}, indent=1))
        sel_path = out / "selection.json"
        selection = json.loads(sel_path.read_text()) if sel_path.exists() else {}
        for target in panel.targets:
            idx = [panel.targets.index(target)]
            relations = runner._group_relations(panel, [target], policy)
            x = runner._prepare_inputs(panel, relations, variant, device)
            if target not in selection:
                runner.stop_year = s - 1
                cands = []
                for point in grid:
                    fit = runner.fit_trial(**common, x=x, target_indices=idx, target_names=[target], panel=tune_panel,
                                           relations=relations, epochs=budget["max_epochs"],
                                           patience=budget["patience"], seed=0, tune=True, anchor=tune_anchor, **point)
                    cands.append({**point, "epochs": fit["best_epoch"], "val_ratio": fit["val_ratio"]})
                selection[target] = {"chosen": min(cands, key=lambda c: (c["val_ratio"], c["epochs"])),
                                     "candidates": cands, "tuned_on": s - 1}
                sel_path.write_text(json.dumps(selection, indent=1))
            chosen = selection[target]["chosen"]
            runner.stop_year = s
            for seed in panel.split["seeds"]:
                path = out / "predictions" / f"{target}_seed_{seed}.csv"
                if path.exists():
                    continue
                fit = runner.fit_trial(**common, x=x, target_indices=idx, target_names=[target], panel=score_panel,
                                       relations=relations, hidden=chosen["hidden"], dropout=chosen["dropout"],
                                       epochs=chosen["epochs"], patience=None, seed=seed, tune=False, anchor=score_anchor)
                frame = pd.DataFrame(fit["prediction_rows"])
                if set(frame.year) != {s}:
                    raise ValueError(f"{target}: predictions are not for origin {s}")
                frame.to_csv(path.with_suffix(".tmp"), index=False)
                path.with_suffix(".tmp").rename(path)
            print(f"{model} {s} {target} done (tuned on {s - 1}: {chosen})", flush=True)
        (out / "status.json").write_text(json.dumps({"status": "completed", "fingerprint": panel.fingerprint}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--origins", type=int, nargs="+", default=ORIGINS)
    ap.add_argument("--target-mode", choices=sorted(OUTS), default="change")
    args = ap.parse_args()
    if not set(args.origins) <= set(ORIGINS):
        raise SystemExit(f"origins must be within {ORIGINS}")
    run(args.model, args.device, args.origins, args.target_mode)


if __name__ == "__main__":
    main()
