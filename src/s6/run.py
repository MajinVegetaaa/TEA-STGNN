"""S6 step 2: the single 2024 test run with the settings in configs/s6_lock.json (nothing is tuned here).

Parts (each can run on its own machine; outputs go to results/s6/):
  neural --model v0|v1|mtgnn   refit on label years 2019-2023 with the locked settings, 5 seeds, forecast 2024.
                               STGNN predictions also carry the shrinkage forecast (`anchor_pred`), whose
                               coefficients must equal the locked ones. Needs data/graph_v2/pyg/s6_test.
  tree                         the Phase A tree, trained on all years before 2024, forecast 2024. Needs the
                               data/graph_v2/*.csv sources.
  s1c                          copy the locked S1c bar picks' 2024 predictions (written in S1c) into a small file.
                               Needs the local results/s1c/*/predictions.csv files.

Every part checks that the 2018-2023 data equal the locked content hash first, and refuses to run again once its
output is complete. `--rehearse` replays the same code on 2023 with the validation bundle and the settings chosen
on 2022 (Step 4 / Plan A), into results/s6_rehearsal/, so the whole path is checked without any 2024 value.

    .venv-expanded/bin/python src/s6/run.py neural --model v0 [--device cuda] [--rehearse]
    .venv-expanded/bin/python src/s6/run.py tree [--rehearse]
    .venv-expanded/bin/python src/s6/run.py s1c
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/train"), str(ROOT / "src/models"), str(ROOT / "src/s6"), str(ROOT / "src/eval")]
# torch-based modules are imported inside the neural part only: torch and scikit-learn's tree load conflicting
# OpenMP runtimes on macOS, so the tree part runs without torch.
from source_digest import digest_through, file_sha256  # noqa: E402

LOCK = ROOT / "configs/s6_lock.json"
OUT = {False: ROOT / "results/s6", True: ROOT / "results/s6_rehearsal"}


def _lock() -> dict:
    return json.loads(LOCK.read_text())


def _provenance(out: Path) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps({
        "code_commit": commit or "unknown (no git)",
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest()}, indent=1))


def _refuse_if_done(out: Path) -> None:
    status = out / "status.json"
    if status.exists() and json.loads(status.read_text()).get("status") == "completed":
        raise SystemExit(f"{out} is already complete; the 2024 test is run once")


def _panel(rehearse: bool):
    from s6_data import TEST_BUNDLE, content_hash, load_test_bundle
    if rehearse:
        from data_s5_round2 import load_validation_bundle
        panel = load_validation_bundle()
    else:
        panel = load_test_bundle(TEST_BUNDLE)
        got, want = content_hash(panel), _lock()["data"]["content_hash_2018_2023"]
        if got != want:
            raise SystemExit(f"2018-2023 data differ from the locked data ({got[:12]} != {want[:12]}); stop")
    return panel


def _rehearsal_settings(model: str) -> dict:
    if model == "mtgnn":
        sel = json.loads((ROOT / "results/s5_round2/mtgnn_comparator/selected_configs.json").read_text())
        return {t: {k: v[k] for k in ("hidden", "dropout", "epochs")} for t, v in sel.items()}
    sel = json.loads((ROOT / f"results/step4_relative/{model}/origin_2023/selection.json").read_text())
    return {t: {k: v["chosen"][k] for k in ("hidden", "dropout", "epochs")} for t, v in sel.items()}


def neural(model: str, device: str, rehearse: bool) -> None:
    import run_step4_rolling as rolling
    lock = _lock()
    spec = lock["models"][model]
    year = 2023 if rehearse else lock["test_year"]
    root = OUT[rehearse] / "neural"
    _refuse_if_done(root / model / f"origin_{year}")
    panel = _panel(rehearse)
    settings = _rehearsal_settings(model) if rehearse else spec["settings"]
    _provenance(root / model)
    rolling.run(model, device, [year], spec["target_mode"], panel=panel, scaler=panel.scaler, out_root=root,
                locked=settings, use_anchor=spec["anchor"] != "persistence", model_args=spec.get("model_args"))
    if spec["anchor"] != "persistence" and not rehearse:
        got = json.loads((root / model / f"origin_{year}" / "anchor_coefficients.json").read_text())["score"]
        want = lock["models"]["shrinkage"]["betas"]
        if [(c["target"], c["tercile"]) for c in got] != [(c["target"], c["tercile"]) for c in want] or not np.allclose(
                [c["beta"] for c in got], [c["beta"] for c in want], atol=1e-6):
            raise SystemExit("shrinkage coefficients differ from the locked ones; results are not valid")


def tree(rehearse: bool) -> None:
    import phase_a_rolling as A
    from sklearn.ensemble import HistGradientBoostingRegressor
    lock = _lock()
    year = 2023 if rehearse else lock["test_year"]
    out = OUT[rehearse] / "tree"
    _refuse_if_done(out)
    want = lock["data"]["source_digests_2018_2023"]  # the tree's inputs must equal the locked 2018-2023 sources
    for name, digest in want.items():
        path = A.GRAPH / name.split(":")[0]
        got = digest_through(path, 2023) if name.endswith(":through_2023") else file_sha256(path)
        if got != digest:
            raise SystemExit(f"{name} differs from the locked source; stop")
    nodes = A.read_through(A.GRAPH / "nodes.csv", cutoff=year)
    edges = {layer: A.read_through(A.GRAPH / f"edges_{layer}.csv", cutoff=year) for layer in A.RELATIONS}
    if nodes.year.max() != year:
        raise SystemExit(f"node source does not reach {year}")
    sets = pd.read_csv(A.GRAPH / "academic_edge_sets.csv").set_index("target").edge_set
    A.B.G, A.S.INPUT_SET = A.GRAPH, "all"
    _provenance(out)
    frames = []
    for target in A.TARGETS:
        df = A.target_table(nodes, edges, target, sets.get(target))
        feats = df.attrs["features"]
        train, test = df[df.year < year], df[df.year == year]
        model = HistGradientBoostingRegressor(**lock["models"]["tree"]["params"]).fit(train[feats], train["_y_fit"])
        pred = model.predict(test[feats]) * test["_scale"].to_numpy() + test["_offset"].to_numpy()
        frames.append(pd.DataFrame({"campus_id": test.campus_id.to_numpy(), "year": year, "target": target,
                                    "y_true": test[target].to_numpy(), "persistence_pred": test["_raw_lag1"].to_numpy(),
                                    "y_pred": pred}))
        print(f"tree {year} {target} done", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    (out / "status.json").write_text(json.dumps({"status": "completed", "year": year}))


def s1c() -> None:
    """The S1c bar picks' 2024 predictions, as written in S1c (refit on 2018-2023). Run after the lock is committed."""
    lock = _lock()
    out = OUT[False] / "s1c_bar"
    _refuse_if_done(out)
    _provenance(out)
    frames = []
    for target, pick in lock["models"]["s1c_bar"]["picks"].items():
        model, _, rest = pick.partition(" | ")
        if model.startswith("persistence"):
            frames.append(pd.DataFrame({"target": [target], "model": ["persistence"]}))
            continue
        mode, input_set = rest.split(" | ")
        keep = []
        for chunk in pd.read_csv(ROOT / f"results/s1c/{input_set}/{mode}/predictions.csv", dtype={"campus_id": str},
                                 chunksize=500_000):
            keep.append(chunk[(chunk.split == "test") & (chunk.model == model) & (chunk.target == target)])
        rows = pd.concat(keep, ignore_index=True)
        if rows.empty or set(rows.year) != {lock["test_year"]}:
            raise SystemExit(f"no 2024 S1c predictions for {pick}")
        frames.append(rows.assign(model=model))
    pd.concat(frames, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    (out / "status.json").write_text(json.dumps({"status": "completed"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="part", required=True)
    n = sub.add_parser("neural")
    n.add_argument("--model", choices=["v0", "v1", "mtgnn"], required=True)
    n.add_argument("--device", default="cpu")
    n.add_argument("--rehearse", action="store_true")
    t = sub.add_parser("tree")
    t.add_argument("--rehearse", action="store_true")
    sub.add_parser("s1c")
    args = ap.parse_args()
    if not LOCK.exists():
        raise SystemExit("configs/s6_lock.json missing: run src/s6/lock.py first")
    if args.part == "neural":
        neural(args.model, args.device, args.rehearse)
    elif args.part == "tree":
        tree(args.rehearse)
    else:
        s1c()


if __name__ == "__main__":
    main()
