"""Fail-closed comparison of complete S5 Round 2 results using 2023 only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "MTGNN": "mtgnn_comparator",
    "V0": "v0_temporal_control",
    "V1_original": "v1_original_attention",
    "V1_query": "v1_query_attention",
    "V2_original": "v2_original_gate",
    "V2_positive": "v2_positive_gate",
    "V3_single": "v3_single_shared_relations",
    "V3_multi": "v3_multi_shared_relations",
}
COMPLEXITY = {"V0": 0, "V1_original": 1, "V1_query": 2, "V2_original": 3,
              "V2_positive": 4, "V3_single": 5, "V3_multi": 6}
SEEDS = [0, 1, 2, 3, 4]


def build_report(results_root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = results_root or ROOT / "results/s5_round2"
    tables, fingerprints, target_orders = {}, set(), set()
    for model, folder in RUNS.items():
        run = root / folder
        manifest_path, protocol_path, metrics_path = run / "run_manifest.json", run / "protocol.json", run / "per_seed_metrics.csv"
        if not all(p.is_file() for p in (manifest_path, protocol_path, metrics_path)):
            raise FileNotFoundError(f"incomplete Round 2 run for {model}: {run}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or protocol.get("validation_years") != [2023]:
            raise ValueError(f"run is incomplete or not 2023-only: {model}")
        if protocol.get("training_years") != [2018, 2019, 2020, 2021, 2022]:
            raise ValueError(f"unexpected training years in {model}")
        fp = manifest.get("snapshot_fingerprint")
        if fp != protocol.get("snapshot_fingerprint"):
            raise ValueError(f"manifest/protocol fingerprint mismatch for {model}")
        fingerprints.add(fp)
        frame = pd.read_csv(metrics_path)
        needed = {"target", "seed", "normalized_mae", "val_mae", "persistence_mae", "snapshot_fingerprint"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"per-seed metrics schema incomplete for {model}")
        if set(frame.seed.astype(int)) != set(SEEDS) or frame.duplicated(["target", "seed"]).any():
            raise ValueError(f"missing, duplicate, or unexpected seed rows for {model}")
        target_sets = set(frame.target.astype(str).unique())
        if len(target_sets) != 10 or (frame.groupby("target").seed.nunique() != 5).any():
            raise ValueError(f"expected exactly 10 targets x 5 seeds for {model}")
        if set(frame.snapshot_fingerprint) != {fp}:
            raise ValueError(f"row-level snapshot fingerprint mismatch for {model}")
        if not np.isfinite(frame[["normalized_mae", "val_mae", "persistence_mae"]].to_numpy()).all():
            raise ValueError(f"non-finite validation metric in {model}")
        target_orders.add(tuple(sorted(target_sets)))
        frame["model"] = model
        tables[model] = frame
    if len(fingerprints) != 1 or len(target_orders) != 1:
        raise ValueError("Round 2 runs do not share one snapshot fingerprint and target set")

    long = pd.concat(tables.values(), ignore_index=True)
    seed_score = long.groupby(["model", "seed"], as_index=False).normalized_mae.mean()
    summary = seed_score.groupby("model").normalized_mae.agg(["mean", "std"]).reset_index()
    summary = summary.rename(columns={"mean": "mean_normalized_mae", "std": "sd_over_seeds"})
    summary["fit_count"] = summary.model.map({"MTGNN": 90, "V0": 90, "V1_original": 90,
        "V1_query": 90, "V2_original": 90, "V2_positive": 90, "V3_single": 90, "V3_multi": 9})
    target_summary = long.groupby(["model", "target"], as_index=False).agg(
        mean_normalized_mae=("normalized_mae", "mean"), sd_over_seeds=("normalized_mae", "std"),
        mean_val_mae=("val_mae", "mean"), mean_persistence_mae=("persistence_mae", "mean"))
    mtgnn = long[long.model == "MTGNN"].set_index(["target", "seed"]).normalized_mae
    paired = long[long.model != "MTGNN"].copy()
    paired["delta_vs_mtgnn"] = [r.normalized_mae - mtgnn.loc[(r.target, int(r.seed))]
                                  for r in paired.itertuples()]
    paired_summary = paired.groupby(["model", "target"], as_index=False).delta_vs_mtgnn.agg(
        mean_paired_delta="mean", sd_paired_delta="std")
    proposed = summary[summary.model.isin(COMPLEXITY)].copy()
    best_score = float(proposed.mean_normalized_mae.min())
    best_row = proposed.loc[proposed.mean_normalized_mae.idxmin()]
    within_one_sd = proposed[proposed.mean_normalized_mae <= best_score + float(best_row.sd_over_seeds)]
    winner = min(within_one_sd.model.tolist(), key=lambda model: COMPLEXITY[model])
    summary["selected_proposed"] = summary.model == winner
    summary["snapshot_fingerprint"] = next(iter(fingerprints))
    return {"summary": summary.sort_values("mean_normalized_mae"),
            "per_target": target_summary, "paired_vs_mtgnn": paired_summary,
            "per_seed": seed_score, "all_rows": long}


def main() -> None:
    reports = build_report()
    root = ROOT / "results/s5_round2"
    out_paths = [root / "comparison_summary.csv", root / "comparison_per_target.csv",
                 root / "comparison_paired_vs_mtgnn.csv", root / "comparison_per_seed.csv"]
    existing = [str(p) for p in out_paths if p.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite Round 2 comparison outputs: " + ", ".join(existing))
    keys = {"summary": out_paths[0], "per_target": out_paths[1],
            "paired_vs_mtgnn": out_paths[2], "per_seed": out_paths[3]}
    for key, path in keys.items():
        reports[key].to_csv(path, index=False)
    print(reports["summary"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
