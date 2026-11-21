"""S1c: tabular numbers to beat under D1, on graph v2. Reuses the Phase 3 runner (run_experiment.py) unchanged and
only adds inputs, so any difference from Phase 3 comes from the inputs and the graph, not from the code.

Input sets
  existing  Phase 3 inputs: own history, other indicators (lag 1), demographics (lag 1), school type, graph features
  all       existing + every added source under D1: staff, AP/IB, STAAR, graduation and finance per student, each as
            last year's value and last year's change. Finance enters as log(1 + USD) (a few tiny schools report
            up to $2.3M per student).

    python3 src/baselines/run_s1c.py --input-set all --mode level [--targets act_composite ...]
    python3 src/baselines/run_s1c.py --graph original --input-set existing --mode level --targets act_composite \\
        --out s1c/regression/level      # must reproduce results/level/predictions.csv for that target
Writes results/<out>/predictions.csv and tuning.csv (default out: s1c/<input-set>/<mode>).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.pop("TEA_EXPANDED", None)          # Phase 3 preprocessing path of run_experiment.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
import run_experiment as B  # noqa: E402
from new_sources import FEATURES, FINANCE_PER_STUDENT  # noqa: E402

TARGETS = ["act_composite", "sat_total", "sat_act_above_criterion", "sat_act_participation", "dropout_9_12",
           "attendance_rate", "retention_rate", "graduation_4yr", "risk_index", "tsi_both"]   # configs/protocol.yaml
GRAPHS = {"v2": ROOT / "data" / "graph_v2", "original": ROOT / "data" / "graph"}
ADDED = FEATURES + FINANCE_PER_STUDENT
_phase3_feature_table = B.feature_table


def d1_feature_table(nodes, edges, target, edge_set):
    """Phase 3 features, plus (input set `all`) last year's value and change of every added source."""
    df, fg = _phase3_feature_table(nodes, edges, target, edge_set)
    if INPUT_SET == "all":
        for v in ADDED:
            if v == target:
                continue                                       # already in the target's own history
            lag1, lag2 = nodes[f"{v}_lag1"], nodes[f"{v}_lag2"]
            if v in FINANCE_PER_STUDENT:
                lag1, lag2 = np.log1p(lag1), np.log1p(lag2)
            df[f"d1_{v}_lag1"], df[f"d1_{v}_change_lag1"] = lag1.to_numpy(), (lag1 - lag2).to_numpy()
            fg["ctx"] += [f"d1_{v}_lag1", f"d1_{v}_change_lag1"]
    return df, fg


def main() -> None:
    global INPUT_SET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", choices=sorted(GRAPHS), default="v2")
    ap.add_argument("--input-set", choices=["existing", "all"], default="all")
    ap.add_argument("--mode", choices=["level", "relative", "zscore"], required=True)
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    INPUT_SET = args.input_set
    B.G = GRAPHS[args.graph]
    B.feature_table = d1_feature_table
    out = args.out or f"s1c/{args.input_set}/{args.mode}"
    sys.argv = [sys.argv[0], "--mode", args.mode, "--out", out, "--targets", *args.targets]
    B.main()


INPUT_SET = "all"
if __name__ == "__main__":
    main()
