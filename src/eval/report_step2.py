"""Step 2 comparison: does the STGNN improve with the relative-change target (and dist_2019 input)?

Reads the six pre-registered runs under results/s5_round2/step2/ (V0 and V1_original x arms A/B/C, all on the
same local bundle) and applies the adoption rule written in DECISIONS.md (2026-11-25) before any run:
an arm is adopted for the STGNN if V1's macro ratio improves on arm A by more than arm A's seed SD AND V1's
per-school spread (phase_a_shift.decompose spread_mae) improves on at least 6 of 10 targets.
Also reports V1 - V0 under each arm (the graph contrast). 2023 validation only.

    .venv-expanded/bin/python src/eval/report_step2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/eval"))
from phase_a_shift import decompose  # noqa: E402

STEP2 = ROOT / "results/s5_round2/step2"
MODELS, ARMS = ["v0", "v1"], ["A_change", "B_relative", "C_relative_dist2019"]
SEEDS = [0, 1, 2, 3, 4]


def load_run(model: str, arm: str, root: Path = STEP2) -> pd.DataFrame:
    run = root / f"{model}_{arm}"
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"run not completed: {run}")
    rows = []
    for path in sorted((run / "predictions").glob("*_seed_*.csv")):
        target, seed = path.stem.rsplit("_seed_", 1)
        d = pd.read_csv(path)
        if set(d.year) != {2023}:
            raise ValueError(f"{path}: not 2023-only")
        rows.append({"model": model, "arm": arm, "target": target, "seed": int(seed),
                     "fingerprint": manifest["snapshot_fingerprint"],
                     **decompose(d.y_true, d.persistence_pred, d.y_pred)})
    frame = pd.DataFrame(rows)
    if frame.groupby("target").seed.nunique().ne(len(SEEDS)).any() or frame.target.nunique() != 10:
        raise ValueError(f"expected 10 targets x 5 seeds in {run}")
    return frame


def build(root: Path = STEP2) -> dict[str, pd.DataFrame]:
    long = pd.concat([load_run(m, a, root) for m in MODELS for a in ARMS], ignore_index=True)
    if long.fingerprint.nunique() != 1:
        raise ValueError("Step 2 runs do not share one bundle fingerprint")
    per_seed = long.groupby(["model", "arm", "seed"], as_index=False).ratio.mean()
    summary = per_seed.groupby(["model", "arm"]).ratio.agg(mean_ratio="mean", sd_over_seeds="std").reset_index()
    per_target = long.groupby(["model", "arm", "target"], as_index=False)[["ratio", "bias", "spread_mae"]].mean()
    v1 = per_target[per_target.model == "v1"].pivot(index="target", columns="arm")
    base = summary.set_index(["model", "arm"]).loc[("v1", "A_change")]
    decisions = []
    for arm in ARMS[1:]:
        macro = summary.set_index(["model", "arm"]).loc[("v1", arm), "mean_ratio"]
        better_spread = int((v1["spread_mae"][arm] < v1["spread_mae"]["A_change"]).sum())
        decisions.append({"arm": arm, "v1_macro": macro, "v1_macro_change_vs_A": macro - base.mean_ratio,
                          "arm_A_sd": base.sd_over_seeds, "targets_with_better_spread": better_spread,
                          "adopted": bool(base.mean_ratio - macro > base.sd_over_seeds and better_spread >= 6)})
    graph = summary.pivot(index="arm", columns="model", values="mean_ratio")
    graph["v1_minus_v0"] = graph["v1"] - graph["v0"]
    return {"summary": summary, "per_target": per_target, "decision": pd.DataFrame(decisions),
            "graph_contrast": graph.reset_index()}


def main() -> None:
    out = build()
    for name, frame in out.items():
        frame.to_csv(STEP2 / f"step2_{name}.csv", index=False)
        print(f"\n== {name}\n{frame.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
