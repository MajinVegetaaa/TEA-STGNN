"""S1 check: the protocol module reproduces the stored baseline numbers exactly, and every model was scored on the
same rows. Writes results/s1/protocol_check.csv and results/s1/phase3_bar.csv. Exit code 1 on any mismatch.

Run from the repo root:  python3 src/eval/check_protocol.py   (or: make protocol)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import PERSIST, load, metrics, paired_bootstrap, selection_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "s1"
TOL = 1e-9
rows: list[dict] = []


def rec(check: str, ok: bool, detail: str = "") -> None:
    rows.append({"check": check, "passed": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {check}" + (f"  ({detail})" if detail else ""))


def seed_mean(p: pd.DataFrame) -> pd.Series:
    return p.groupby("seed").apply(metrics, include_groups=False).mean()


def main() -> int:
    cfg = load()
    split = cfg["split"]
    rec("config: 10 targets, test year disjoint from train/validation",
        len(cfg["targets"]) == 10 and not set(split["test_years"]) & set(split["train_years"] + split["val_years"]),
        f"targets {len(cfg['targets'])}; train {split['train_years']}, val {split['val_years']}, test {split['test_years']}")

    # 1. Original Phase 3 (results/selection.csv): 8 targets, best of level / relative / zscore
    sel = pd.read_csv(ROOT / "results/selection.csv")
    preds = {m: pd.read_csv(ROOT / "results" / m / "predictions.csv", dtype={"campus_id": str})
             for m in sel["mode"].unique()}
    bar = []
    for r in sel.itertuples():
        p = preds[r.mode]
        p = p[p.target == r.target]
        got = {}
        for name, model in (("persistence", PERSIST), ("model", r.model)):
            for s in ("val", "test"):
                got[(name, s)] = seed_mean(p[(p.model == model) & (p.split == s)])
        stored = {("persistence", "val"): r.val_mae_persistence, ("model", "val"): r.val_mae_model,
                  ("persistence", "test"): r.test_mae_persistence, ("model", "test"): r.test_mae_model}
        diff = max(abs(got[k].mae - v) for k, v in stored.items())
        diff = max(diff, abs(got[("model", "test")].r2 - r.test_r2_model), abs(got[("model", "test")].rmse - r.test_rmse_model))
        t = p[p.split == "test"]
        bs = paired_bootstrap(t[t.model == PERSIST], t[t.model == r.model])
        diff = max(diff, abs(bs["ci_low"] - r.ci_low), abs(bs["ci_high"] - r.ci_high))
        rec(f"Phase 3 {r.target}: MAE, RMSE, R2 and bootstrap CI reproduce", diff < TOL, f"max abs diff {diff:.1e}")

        # every model on identical rows, per split and seed
        keys = p.groupby(["split", "seed", "model"]).campus_id.apply(lambda s: tuple(sorted(s)))
        same = all(g.nunique() == 1 for _, g in keys.groupby(level=["split", "seed"]))
        rec(f"Phase 3 {r.target}: every model scored on the same schools", same)

        per_seed = p[(p.model == r.model) & (p.split == "val")].groupby("seed").apply(metrics, include_groups=False).mae
        bar.append({"target": r.target, "mode": r.mode, "model": r.model,
                    "val_mae_persistence": got[("persistence", "val")].mae,
                    **{f"val_mae_seed{s}": v for s, v in per_seed.items()}})

    # the bar in the new score (8 targets; S1c recomputes it for all 10 under D1 and graph v2)
    b = pd.DataFrame(bar).set_index("target")
    seeds = [c for c in b if c.startswith("val_mae_seed")]
    score, sd = selection_score(b[seeds].T.reset_index(drop=True), b.val_mae_persistence)
    b.assign(ratio=b[seeds].mean(axis=1) / b.val_mae_persistence).to_csv(OUT / "phase3_bar.csv")
    rec("Phase 3 bar in the final-model score (2023, 8 targets)", np.isfinite(score),
        f"score {score:.4f} +/- {sd:.4f} (1.0 = persistence)")

    # 2. The expanded 10-target selection that Phase 5 compared against
    rep = pd.read_csv(ROOT / "results/expanded/final_selection_report.csv")
    ex = pd.read_csv(ROOT / "results/expanded/selected_test_predictions.csv", dtype={"campus_id": str})
    worst = 0.0
    for r in rep.itertuples():
        q = ex[(ex.input_set == r.input_set) & (ex.target == r.target) & (ex.model == r.model)]
        worst = max(worst, abs(seed_mean(q).mae - r.mae))
    rec("expanded 10-target selection (Phase 5 comparator): test MAE reproduces", worst < TOL,
        f"{len(rep)} targets; max abs diff {worst:.1e}")

    pd.DataFrame(rows).to_csv(OUT / "protocol_check.csv", index=False)
    failed = sum(not r["passed"] for r in rows)
    print(f"\n{len(rows) - failed}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sys.exit(main())
