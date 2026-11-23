"""2023 scores of neural runs next to the bar (S3 go/no-go, S4 MTGNN, later S5 versions). 2024 is never read.

For every run (results/<run>/validation_seeds.csv) and target, the locked graph choice with the lower 5-seed mean
2023 MAE is kept (the same "best on 2023" rule as the bar). Score = protocol selection score: mean over targets of
MAE / persistence MAE, mean and SD over seeds.

    python3 src/eval/report_neural.py --step s3      # writes results/s3/summary.csv, per_target.csv
    python3 src/eval/report_neural.py --step s4      # writes results/s4/summary.csv, per_target.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from protocol import load, selection_score

ROOT = Path(__file__).resolve().parents[2]
STEPS = {  # step -> {run folder under results/: (model, setting)}
    "s3": {"s3/gconvgru_phase5": ("GConvGRU", "A: v1, Phase 5 inputs, level"),
           "s3/gconvgru_v2_level": ("GConvGRU", "B: v2, D1 inputs, level"),
           "s3/gconvgru_v2_change": ("GConvGRU", "C: v2, D1 inputs, change"),
           "s3/graphwavenet_phase5": ("GraphWaveNet", "A: v1, Phase 5 inputs, level"),
           "s3/graphwavenet_v2_level": ("GraphWaveNet", "B: v2, D1 inputs, level"),
           "s3/graphwavenet_v2_change": ("GraphWaveNet", "C: v2, D1 inputs, change")},
    "s4": {"s3/graphwavenet_v2_change": ("GraphWaveNet", "S3 best: v2, D1 inputs, change"),
           "s4/mtgnn": ("MTGNN", "v2 + learned graph, D1 inputs, change, budget B")},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=sorted(STEPS), required=True)
    step = ap.parse_args().step
    targets = load()["targets"]
    bar = json.loads((ROOT / "results/s1c/bar.json").read_text())["score_2023"]
    bar_t = pd.read_csv(ROOT / "results/s1c/numbers_to_beat.csv").set_index("target").val_ratio
    summary, per_target = [], []
    for run, (model, setting) in STEPS[step].items():
        v = pd.read_csv(ROOT / "results" / run / "validation_seeds.csv")
        mean = v.groupby(["target", "graph"]).val_mae.mean()
        pick = mean.groupby(level="target").idxmin().map(lambda ix: ix[1])          # best locked graph per target
        chosen = v[v.apply(lambda r: pick[r.target] == r.graph, axis=1)]
        model_mae = chosen.pivot(index="seed", columns="target", values="val_mae")[targets]
        persistence = chosen.groupby("target").persistence_mae.first()[targets]
        score, sd = selection_score(model_mae, persistence)
        combined = v[v.graph == "combined"].pivot(index="seed", columns="target", values="val_mae")[targets]
        c_score, _ = selection_score(combined, persistence)
        ratios = model_mae.mean() / persistence
        summary.append({"model": model, "setting": setting, "score_2023": score, "sd_over_seeds": sd,
                        "score_combined_graph_only": c_score, "bar": bar, "gap_to_bar": score - bar,
                        "targets_beating_persistence": int((ratios < 1).sum()),
                        "targets_at_or_better_than_bar": int((ratios <= bar_t[targets]).sum())})
        for t in targets:
            per_target.append({"model": model, "setting": setting, "target": t, "graph": pick[t],
                               "ratio_2023": ratios[t], "bar_ratio_2023": bar_t[t]})
    out = ROOT / "results" / step
    s, pt = pd.DataFrame(summary), pd.DataFrame(per_target)
    s.to_csv(out / "summary.csv", index=False)
    pt.to_csv(out / "per_target.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))
    print("\nratio to persistence on 2023 per target (bar in the last column)")
    w = pt.pivot_table(index="target", columns=["model", "setting"], values="ratio_2023").round(3)
    w[("bar", "S1c")] = bar_t.round(3)
    print(w.loc[targets].to_string())


if __name__ == "__main__":
    main()
