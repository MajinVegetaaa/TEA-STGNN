"""Step 1 comparison: does the STGNN improve when it predicts the change from the shrinkage forecast?

Applies the rule pre-registered in DECISIONS.md (2026-11-25, Step 1) before the runs: arm E (anchor
shrink_own_3yr, results/s5_round2/step1/) is adopted if V1's macro ratio improves on V1 arm A (Step 2 control,
same local bundle) by more than arm A's seed SD AND V1's per-school spread improves on at least 6 of 10 targets.
Also reports the anchor alone (zero network output) and V1 - V0 under E. 2023 validation only.

    .venv-expanded/bin/python src/eval/report_step1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/eval"))
from phase_a_shift import decompose  # noqa: E402
from report_step2 import STEP2, load_run  # noqa: E402

STEP1 = ROOT / "results/s5_round2/step1"
ARM = "E_shrink_anchor"


def anchor_alone(root: Path = STEP1) -> pd.DataFrame:
    """The shrinkage anchor scored by itself on the same rows (identical for every seed and model)."""
    rows = []
    for path in sorted((root / f"v0_{ARM}" / "predictions").glob("*_seed_0.csv")):
        d = pd.read_csv(path)
        rows.append({"target": path.stem.rsplit("_seed_", 1)[0],
                     **decompose(d.y_true, d.persistence_pred, d.anchor_pred)})
    return pd.DataFrame(rows)


def build() -> dict[str, pd.DataFrame]:
    long = pd.concat([load_run(m, "A_change", STEP2) for m in ("v0", "v1")]
                     + [load_run(m, ARM, STEP1) for m in ("v0", "v1")], ignore_index=True)
    if long.fingerprint.nunique() != 1:
        raise ValueError("Step 1 and the Step 2 control do not share one bundle fingerprint")
    per_seed = long.groupby(["model", "arm", "seed"], as_index=False).ratio.mean()
    summary = per_seed.groupby(["model", "arm"]).ratio.agg(mean_ratio="mean", sd_over_seeds="std").reset_index()
    alone = anchor_alone()
    summary = pd.concat([summary, pd.DataFrame([{"model": "anchor_alone", "arm": ARM,
                                                 "mean_ratio": alone.ratio.mean(), "sd_over_seeds": 0.0}])])
    per_target = long.groupby(["model", "arm", "target"], as_index=False)[["ratio", "bias", "spread_mae"]].mean()
    per_target = pd.concat([per_target, alone.assign(model="anchor_alone", arm=ARM)[per_target.columns]])
    s = summary.set_index(["model", "arm"])
    base, e = s.loc[("v1", "A_change")], s.loc[("v1", ARM)]
    v1 = per_target[per_target.model == "v1"].pivot(index="target", columns="arm")
    better = int((v1["spread_mae"][ARM] < v1["spread_mae"]["A_change"]).sum())
    decision = pd.DataFrame([{"arm": ARM, "v1_macro": e.mean_ratio, "v1_macro_change_vs_A": e.mean_ratio - base.mean_ratio,
                              "arm_A_sd": base.sd_over_seeds, "targets_with_better_spread": better,
                              "adopted": bool(base.mean_ratio - e.mean_ratio > base.sd_over_seeds and better >= 6),
                              "anchor_alone": alone.ratio.mean(),
                              "v1_minus_anchor_alone": e.mean_ratio - alone.ratio.mean(),
                              "v1_minus_v0_under_E": e.mean_ratio - s.loc[("v0", ARM)].mean_ratio}])
    return {"summary": summary, "per_target": per_target, "decision": decision}


def main() -> None:
    out = build()
    for name, frame in out.items():
        frame.to_csv(STEP1 / f"step1_{name}.csv", index=False)
        print(f"\n== {name}\n{frame.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
