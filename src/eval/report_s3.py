"""S3 go/no-go: can graph models get close to the tabular bar on 2023?

For every S3 run (results/s3/<run>/validation_seeds.csv) and every target, the graph choice (combined or best single
relation, both locked on 2023 at seed 0) with the lower 5-seed mean 2023 MAE is kept -- the same "best on 2023" rule
the bar used. Score = protocol selection score (mean over targets of MAE / persistence MAE; mean and SD over seeds).
Writes results/s3/summary.csv (per run) and results/s3/per_target.csv (ratios next to the bar's).
Run from the repo root:  python3 src/eval/report_s3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from protocol import load, selection_score

ROOT = Path(__file__).resolve().parents[2]
S3 = ROOT / "results" / "s3"
RUNS = {  # run folder -> (model, setting)
    "gconvgru_phase5": ("GConvGRU", "A: v1, Phase 5 inputs, level"),
    "gconvgru_v2_level": ("GConvGRU", "B: v2, D1 inputs, level"),
    "gconvgru_v2_change": ("GConvGRU", "C: v2, D1 inputs, change"),
    "graphwavenet_phase5": ("GraphWaveNet", "A: v1, Phase 5 inputs, level"),
    "graphwavenet_v2_level": ("GraphWaveNet", "B: v2, D1 inputs, level"),
    "graphwavenet_v2_change": ("GraphWaveNet", "C: v2, D1 inputs, change"),
}


def main() -> None:
    targets = load()["targets"]
    bar = json.loads((ROOT / "results/s1c/bar.json").read_text())["score_2023"]
    bar_t = pd.read_csv(ROOT / "results/s1c/numbers_to_beat.csv").set_index("target").val_ratio
    summary, per_target = [], []
    for run, (model, setting) in RUNS.items():
        v = pd.read_csv(S3 / run / "validation_seeds.csv")
        mean = v.groupby(["target", "graph"]).val_mae.mean()
        pick = mean.groupby(level="target").idxmin().map(lambda ix: ix[1])          # best graph per target, 2023
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
    s = pd.DataFrame(summary)
    s.to_csv(S3 / "summary.csv", index=False)
    pt = pd.DataFrame(per_target)
    pt.to_csv(S3 / "per_target.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))
    print("\nratio to persistence on 2023 per target (bar in the last column)")
    w = pt.pivot_table(index="target", columns=["model", "setting"], values="ratio_2023").round(3)
    w[("bar", "S1c")] = bar_t.round(3)
    print(w.loc[targets].to_string())


if __name__ == "__main__":
    main()
