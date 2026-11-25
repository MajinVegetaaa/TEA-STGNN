"""S6 step 1: lock every model's settings for the 2024 test, using the 2018-2023 validation bundle only.

    tune   choose each STGNN's hidden/dropout/epochs per target on 2023 after training on label years <= 2022
           (Plan A: shrinkage anchor + relative-change target), exactly the Step 4 tuning stage at origin 2024.
    write  assemble configs/s6_lock.json from the tuned settings and the earlier results (see build_lock).

Nothing in this file opens a 2024 value: the validation bundle stops at 2023 and its loader rejects later years.

    .venv-expanded/bin/python src/s6/lock.py tune --model v0
    .venv-expanded/bin/python src/s6/lock.py write
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/train"), str(ROOT / "src/models"), str(ROOT / "src/s6")]
import run_s5_round2 as runner  # noqa: E402
import run_step4_rolling as rolling  # noqa: E402
from s6_data import content_hash  # noqa: E402
from data_s5_round2 import load_validation_bundle  # noqa: E402
from source_digest import digest_through, file_sha256  # noqa: E402

TUNED = ROOT / "results/s6_lock"
LOCK = ROOT / "configs/s6_lock.json"
TREE = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, max_iter=200, random_state=0)  # Phase A
TEST_YEAR = 2024
TARGET_MODE = "change_relative"           # Plan A, the best STGNN configuration under honest selection


def tune(model: str, device: str) -> None:
    rolling.run(model, device, [TEST_YEAR], TARGET_MODE, out_root=TUNED, tune_only=True)


def locked_betas(panel) -> list[dict]:
    """Shrinkage coefficients for the 2024 forecast: fitted on label years 2019-2023 (all of them are pre-2024)."""
    y, lag, _ = runner._target_tensors(panel, "cpu")
    full = SimpleNamespace(**{k: getattr(panel, k) for k in ("years", "snapshots", "targets", "feature_names")})
    full.split = dict(panel.split, train_years=list(panel.years))
    return runner.shrinkage_anchor(full, y, lag, panel.scaler)[1]


def locked_shrinkage(panel) -> dict:
    """Everything the 2024 shrinkage forecast uses, applied as is in S6 (nothing is re-fitted on the test bundle)."""
    coef = locked_betas(panel)
    return {"formula": "lag1 + beta * (ma3 - lag1); beta per target x enrollment tercile, least squares clipped "
                       "to [0, 1]", "fit_label_years": [2019, 2020, 2021, 2022, 2023],
            "cuts": coef[0]["cuts"], "fill_enrollment": coef[0]["fill_enrollment"],
            "betas": [{k: c[k] for k in ("target", "tercile", "beta", "n_fit")} for c in coef]}


def source_digests() -> dict:
    """Digests of the CSV sources the tree reads, through 2023 (equal to those recorded in the validation bundle)."""
    recorded = json.loads((ROOT / "data/graph_v2/pyg/s5_validation/metadata.json").read_text())["source_hashes_through_cutoff"]
    out = {}
    for layer in ("nodes", "edges_district", "edges_geographic", "edges_demographic", "edges_region", "edges_academic"):
        got = digest_through(ROOT / f"data/graph_v2/{layer}.csv", 2023)
        if got != recorded[f"{layer}.csv:through_cutoff"]:
            raise ValueError(f"{layer}.csv no longer matches the validation bundle")
        out[f"{layer}.csv:through_2023"] = got
    out["academic_edge_sets.csv"] = file_sha256(ROOT / "data/graph_v2/academic_edge_sets.csv")
    if out["academic_edge_sets.csv"] != recorded["academic_edge_sets.csv"]:
        raise ValueError("academic_edge_sets.csv no longer matches the validation bundle")
    return out


def honest_bar_picks() -> dict:
    """Tree or own-history shrinkage per target, by mean ratio over origins 2021-2023 (2020: no 3-year history)."""
    r = pd.read_csv(ROOT / "results/phase_a/rolling_scores.csv")
    r = r[r.origin.isin([2021, 2022, 2023])]
    means = r.groupby("target")[["tree", "shrink_to_own_3yr"]].mean()
    return {t: ("tree" if row.tree < row.shrink_to_own_3yr else "shrinkage") for t, row in means.iterrows()}


def stgnn_settings(model: str) -> dict:
    sel = json.loads((TUNED / model / "origin_2024" / "selection.json").read_text())
    status = json.loads((TUNED / model / "origin_2024" / "status.json").read_text())
    if status["status"] != "tuned" or len(sel) != 10:
        raise ValueError(f"{model}: tuning on 2023 is not complete")
    return {t: {k: v["chosen"][k] for k in ("hidden", "dropout", "epochs")} for t, v in sel.items()}


def build_lock() -> dict:
    panel = load_validation_bundle()
    mtgnn_cfg = yaml.safe_load((ROOT / "configs/experiments/s5_round2_mtgnn_comparator.yaml").read_text())
    mtgnn_sel = json.loads((ROOT / "results/s5_round2/mtgnn_comparator/selected_configs.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    s1c = json.loads((ROOT / "results/s1c/bar.json").read_text())
    stgnn = {"anchor": "shrink_own_3yr", "target_mode": TARGET_MODE, "tasks": "single",
             "train_label_years": [2019, 2020, 2021, 2022, 2023],
             "settings_chosen_on": "2023, after training on label years 2019-2022 (budget-B grid, seed 0, patience 20)"}
    return {
        "test_year": TEST_YEAR,
        "locked_at_commit": commit,
        "data": {"validation_bundle_fingerprint": panel.fingerprint,
                 "content_hash_2018_2023": content_hash(panel),
                 "preprocessing_fit_years": panel.scaler["fit_years"],
                 "source_digests_2018_2023": source_digests(),
                 "note": "the 2024 bundle must reproduce content_hash_2018_2023 before any model is fitted"},
        "seeds": panel.split["seeds"],
        "budget": yaml.safe_load((ROOT / "configs/budget.yaml").read_text()),
        "final_model": {"model": "v0", "rule": "final-model rule on the honest (nested) 2022/2023 results: Plan A V0 "
                        "1.014, Plan A V1 1.016, within one seed SD, so the simpler V0 wins; V1 is reported as the "
                        "proposed graph model"},
        "models": {
            "v0": dict(stgnn, variant="v0_temporal", relation_policy="none", settings=stgnn_settings("v0")),
            "v1": dict(stgnn, variant="v1_original", relation_policy="target_specific", settings=stgnn_settings("v1")),
            "mtgnn": {"variant": "mtgnn", "relation_policy": "target_specific", "anchor": "persistence",
                      "target_mode": "change", "model_args": mtgnn_cfg["model_args"],
                      "settings_chosen_on": "2023 in S5 Round 2 (results/s5_round2/mtgnn_comparator)",
                      "settings": {t: {k: v[k] for k in ("hidden", "dropout", "epochs")} for t, v in mtgnn_sel.items()}},
            "shrinkage": locked_shrinkage(panel),
            "tree": {"params": TREE, "inputs": "S1c 'all' inputs, zscore mode, no graph features (Phase A)",
                     "train_years": "all years before 2024"},
            "honest_bar": {"picks": honest_bar_picks(), "chosen_on": "origins 2021-2023, results/phase_a/rolling_scores.csv"},
            "s1c_bar": {"picks": s1c["picks"], "note": "selected on 2023 among ~90 candidates; refit on 2018-2023 and "
                        "2024 predictions written in S1c, before S6; reported, not re-run"},
        },
        "scoring": {"rows": "per target, schools with the 2024 target and its 2023 value observed, present in every "
                    "model's predictions", "score": "MAE / persistence MAE, mean over targets; seeds: mean and SD",
                    "bootstrap": {"method": "paired, seed-averaged predictions, identical schools", "n_boot": 2000, "seed": 0},
                    "comparisons": [["v0", "mtgnn"], ["v0", "honest_bar"], ["v0", "s1c_bar"],
                                    ["v1", "mtgnn"], ["v1", "honest_bar"], ["v1", "s1c_bar"], ["v1", "v0"]]},
    }


def write_lock() -> None:
    if LOCK.exists():
        raise FileExistsError(f"{LOCK} exists; the lock is written once")
    LOCK.write_text(json.dumps(build_lock(), indent=1) + "\n")
    print(f"wrote {LOCK}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--model", choices=sorted(rolling.MODELS), required=True)
    t.add_argument("--device", default="cpu")
    sub.add_parser("write")
    args = ap.parse_args()
    if args.cmd == "tune":
        tune(args.model, args.device)
    else:
        write_lock()


if __name__ == "__main__":
    main()
