"""Train and select a neural model on the locked protocol (configs/protocol.yaml), from one experiment config.

    python src/train/run.py configs/experiments/<name>.yaml            # 2023 selection only (seed 0)
    python src/train/run.py configs/experiments/<name>.yaml --val-seeds  # + 2023 score for all 5 seeds (mean +/- SD)
    python src/train/run.py configs/experiments/<name>.yaml --test     # + refit on 2018-2023, 5 seeds, score 2024

Selection (2023): for every graph candidate (`combined` = all of a target's relations, plus each single relation if
`graphs: combined+single`), train with seed 0 and pick the epoch with the best 2023 score. `combined` and the best
single relation are locked. --test refits each locked choice on 2018-2023 for every seed and forecasts 2024.
For our own models --test is used once, in S6; for reproducing existing baselines it is allowed.

Writes results/<out>/: config.yaml, protocol.json, validation_grid.csv, selection_locked.json and, with --test,
predictions.csv (standard format + graph), per_seed.csv, comparison.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
import data as D  # noqa: E402
from families import FAMILIES  # noqa: E402
from protocol import metrics  # noqa: E402
from trainer import fit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--val-seeds", action="store_true", help="refit the locked choices on 2018-2022 for every seed, score 2023")
    ap.add_argument("--test", action="store_true", help="refit the locked choices and score 2024")
    ap.add_argument("--targets", nargs="*", help="subset of targets (smoke tests)")
    ap.add_argument("--max-epochs", type=int, help="override (smoke tests)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(cfg.get("threads", 2))
    device = cfg.get("device", "cpu")
    out = ROOT / "results" / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    panel = D.load(cfg["graph"])
    family = FAMILIES[cfg["model"]]
    targets = args.targets or cfg.get("targets") or panel.targets
    epochs = args.max_epochs or cfg["max_epochs"]
    val, test = panel.val_index, panel.val_index + 1
    (out / "protocol.json").write_text(json.dumps(
        {"config": cfg, "snapshot": str(panel.path.relative_to(ROOT)), "snapshot_sha256": panel.sha256,
         "targets": targets, "torch_version": torch.__version__, "device": device, "test_run": args.test}, indent=2))

    cache = {}

    def inputs(relations):
        key = tuple(relations)
        if key not in cache:
            cache[key] = family.prepare(panel.snapshots, relations).to(device)
        return cache[key]

    def train(target_list, relations, stop, n_epochs, seed, tune):
        j = [panel.targets.index(t) for t in target_list]
        x = inputs(relations)
        build = lambda: family.build(family.features(x), cfg["hidden"], len(j))
        return fit(build, x, panel.y[..., j].to(device), panel.y_lag1[..., j].to(device),
                   panel.eval_mask[..., j].to(device), stop, n_epochs, cfg["lr"], seed, cfg["label"], tune, device)

    # ---- selection on 2023
    groups = [[t] for t in targets] if cfg["tasks"] == "single" else [targets]
    grid, locked = [], []
    for group in groups:
        relations = panel.relations(group[0]) if len(group) == 1 else cfg["relations"]
        candidates = [("combined", relations)]
        if cfg.get("graphs") == "combined+single":
            candidates += [(r, [r]) for r in relations]
        rows = []
        for name, rel in candidates:
            f = train(group, rel, val, epochs, 0, True)
            rows.append({"targets": group, "graph": name, "relations": rel, "epochs": f.best_epoch,
                         "val_ratio": f.val_ratio, "val_mae": dict(zip(group, f.val_mae))})
        grid += rows
        locked.append(rows[0])
        if len(rows) > 1:
            locked.append(min(rows[1:], key=lambda r: r["val_ratio"]))
        print("selected:", ", ".join(group), "|", ", ".join(f"{r['graph']} ({r['epochs']} ep)" for r in locked[-2:]),
              flush=True)
    pd.DataFrame([{**{k: v for k, v in r.items() if k != "val_mae"}, "target": t, "val_mae": r["val_mae"][t]}
                  for r in grid for t in r["targets"]]).drop(columns="targets").to_csv(out / "validation_grid.csv",
                                                                                     index=False)
    (out / "selection_locked.json").write_text(json.dumps(locked, indent=2))

    # ---- 2023 for every seed (the final-model score is a mean +/- SD over seeds); never touches 2024
    if args.val_seeds:
        rows = []
        for row in locked:
            for seed in panel.split["seeds"]:
                f = train(row["targets"], row["relations"], val, row["epochs"], seed, False)
                for k, t in enumerate(row["targets"]):
                    j = panel.targets.index(t)
                    mask = panel.eval_mask[val, :, j]
                    truth = panel.y[val, mask, j].numpy()
                    rows.append({"target": t, "graph": row["graph"], "seed": seed, "n": int(mask.sum()),
                                 "val_mae": float(np.abs(f.forecast[mask.numpy(), k] - truth).mean()),
                                 "persistence_mae": float(np.abs(panel.y_lag1[val, mask, j].numpy() - truth).mean())})
        pd.DataFrame(rows).to_csv(out / "validation_seeds.csv", index=False)
    if not args.test:
        return

    # ---- 2024, once: refit each locked choice on 2018-2023 for every seed
    preds = []
    for row in locked:
        for seed in panel.split["seeds"]:
            f = train(row["targets"], row["relations"], test, row["epochs"], seed, False)
            for k, t in enumerate(row["targets"]):
                j = panel.targets.index(t)
                mask = panel.eval_mask[test, :, j].numpy()
                ids = np.array(panel.campus_ids)[mask]
                common = dict(target=t, seed=seed, campus_id=ids, year=panel.years[test], split="test",
                              y_true=panel.y[test, mask, j].numpy())
                preds.append(pd.DataFrame(dict(model=cfg["name"], graph=row["graph"], y_pred=f.forecast[mask, k], **common)))
    p = pd.concat(preds, ignore_index=True)
    p.to_csv(out / "predictions.csv", index=False)
    per_seed = p.groupby(["target", "graph", "seed"]).apply(metrics, include_groups=False).reset_index()
    per_seed.to_csv(out / "per_seed.csv", index=False)
    g = per_seed.groupby(["target", "graph"])
    g[["mae", "rmse", "r2"]].mean().assign(mae_sd=g.mae.std(ddof=0)).to_csv(out / "comparison.csv")
    print(f"test: {len(locked)} locked choices x {len(panel.split['seeds'])} seeds -> {out}")


if __name__ == "__main__":
    main()
