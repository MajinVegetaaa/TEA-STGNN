"""S2 done-when: the harness reproduces the Phase 5 GCN + LSTM benchmark.

Compares results/s2/gcn_lstm_phase5/ with results/phase5/gcn_lstm/benchmark/: same snapshot, every validation
candidate's 2023 MAE and chosen epochs, the locked choices, and every 2024 per-seed MAE.
Run from the repo root:  python src/train/check_reproduction.py   (exit 1 on any mismatch)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NEW, OLD = ROOT / "results/s2/gcn_lstm_phase5", ROOT / "results/phase5/gcn_lstm/benchmark"
TOL = 1e-5
ok = []


def rec(name: str, cond: bool, detail: str = "") -> None:
    ok.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    a, b = json.loads((NEW / "protocol.json").read_text()), json.loads((OLD / "protocol.json").read_text())
    rec("same snapshot as Phase 5", a["snapshot_sha256"] == b["snapshot_sha256"])
    rec("same hyperparameters", (a["config"]["hidden"], a["config"]["lr"], a["config"]["max_epochs"])
        == (b["hidden"], b["learning_rate"], b["epochs"]))

    g = pd.read_csv(NEW / "validation_grid.csv").merge(pd.read_csv(OLD / "validation_grid.csv"),
                                                        on=["target", "graph"], suffixes=("", "_p5"))
    rec("every validation candidate present", len(g) == len(pd.read_csv(OLD / "validation_grid.csv")), f"{len(g)}")
    diff = (g.val_mae - g.val_mae_p5).abs().max()
    rec("2023 MAE of every candidate", diff < TOL, f"max abs diff {diff:.1e}")
    rec("chosen epochs of every candidate", (g.epochs == g.epochs_p5).all())

    new_lock = {(t, r["graph"]) for r in json.loads((NEW / "selection_locked.json").read_text()) for t in r["targets"]}
    old_lock = {(r["target"], r["graph"]) for r in json.loads((OLD / "selection_locked.json").read_text())}
    rec("locked choices (combined + best single relation)", new_lock == old_lock, f"{len(new_lock)}")

    s = pd.read_csv(NEW / "per_seed.csv").merge(pd.read_csv(OLD / "per_seed.csv"), on=["target", "graph", "seed"],
                                                 suffixes=("", "_p5"))
    rec("every 2024 target/graph/seed present", len(s) == len(pd.read_csv(OLD / "per_seed.csv")), f"{len(s)}")
    diff = (s.mae - s.mae_p5).abs().max()
    rec("2024 MAE of every seed", diff < TOL, f"max abs diff {diff:.1e}")
    rec("same scored schools", (s.n == s.n_p5).all())
    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
