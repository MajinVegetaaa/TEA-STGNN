"""S7 post-test ablation: every STGNN version in both setups, forecast 2022, 2023 and 2024 with nested selection.

POST-TEST: run after the S6 2024 result was seen. It does not change the S6 final model or table; it is reported
as an ablation (DECISIONS 2026-11-26).

Setups
  r2     Round 2 design: the model predicts the change from last year's value (persistence anchor, change label).
  planA  Plan A: the model predicts the change from the shrinkage forecast, with each year's statewide mean change
         removed from the training labels (shrinkage anchor, relative-change label).
Models  v0 (no graph), v1 (V1_original), v1s4 (V1 on the four shared relations), v2 (V1 + gate), v3 (V2 as one
        multi-task model, roadmap v3), and MTGNN (r2 only, as designed).
Origins 2022 and 2023: train on label years before s-1, choose hidden/dropout/epochs on s-1 (budget B), refit on
        label years before s, forecast s (validation bundle). 2024: choose on 2023 with the validation bundle,
        refit on 2019-2023 with the test bundle, forecast 2024 (planA uses the locked S6 shrinkage, as S6 did).
Already available and not re-run: planA V0/V1 at 2022/2023 (results/step4_relative), planA V0/V1 and MTGNN at
        2024 (results/s6/neural).

    python src/s7/ablation_run.py status
    python src/s7/ablation_run.py all --device cuda          # every pending run, cheapest first; safe to re-run
    python src/s7/ablation_run.py one --setup r2 --model v2 --origin 2023 --device cuda
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/train"), str(ROOT / "src/models"), str(ROOT / "src/s6")]

OUT = ROOT / "results/s7_ablation"
LOCK = ROOT / "configs/s6_lock.json"
SETUPS = {"r2": {"target_mode": "change", "use_anchor": False},
          "planA": {"target_mode": "change_relative", "use_anchor": True}}
STGNN = ["v0", "v1", "v1s4", "v2", "v3"]
ORIGINS = [2022, 2023, 2024]
REUSED = {("planA", "v0", 2022): "results/step4_relative/v0/origin_2022",
          ("planA", "v0", 2023): "results/step4_relative/v0/origin_2023",
          ("planA", "v1", 2022): "results/step4_relative/v1/origin_2022",
          ("planA", "v1", 2023): "results/step4_relative/v1/origin_2023",
          ("planA", "v0", 2024): "results/s6/neural/v0/origin_2024",
          ("planA", "v1", 2024): "results/s6/neural/v1/origin_2024",
          ("r2", "mtgnn", 2024): "results/s6/neural/mtgnn/origin_2024"}
COST = {"v0": 0, "v3": 1, "v1": 2, "v1s4": 2, "v2": 2, "mtgnn": 3}     # rough order: cheapest first


def plan() -> list[tuple[str, str, int]]:
    runs = [(s, m, o) for s in SETUPS for m in STGNN + (["mtgnn"] if s == "r2" else []) for o in ORIGINS]
    return sorted([r for r in runs if r not in REUSED], key=lambda r: (COST[r[1]], r[2], r[0]))


def run_dir(setup: str, model: str, origin: int) -> Path:
    return ROOT / REUSED[(setup, model, origin)] if (setup, model, origin) in REUSED else OUT / setup / model / f"origin_{origin}"


def done(setup: str, model: str, origin: int) -> bool:
    status = run_dir(setup, model, origin) / "status.json"
    return status.exists() and json.loads(status.read_text()).get("status") == "completed"


def _check(panel, label: str) -> None:
    from s6_data import content_hash
    want = json.loads(LOCK.read_text())["data"]["content_hash_2018_2023"]
    if content_hash(panel) != want:
        raise SystemExit(f"{label}: 2018-2023 data differ from the locked data; stop")


def _model_args(model: str) -> dict | None:
    if model != "mtgnn":
        return None
    return json.loads(LOCK.read_text())["models"]["mtgnn"]["model_args"]


def one(setup: str, model: str, origin: int, device: str) -> None:
    import torch
    import run_step4_rolling as rolling
    from data_s5_round2 import load_validation_bundle
    if (setup, model, origin) in REUSED:
        raise SystemExit(f"{setup}/{model}/{origin} is reused from {REUSED[(setup, model, origin)]}")
    if model == "mtgnn" and setup != "r2":
        raise SystemExit("MTGNN is run in its designed setup (r2) only")
    if done(setup, model, origin):
        print(f"skip {setup}/{model}/{origin}: complete", flush=True)
        return
    spec, args = SETUPS[setup], _model_args(model)
    start = time.time()
    val = load_validation_bundle()
    _check(val, "validation bundle")
    if origin < 2024:
        rolling.run(model, device, [origin], spec["target_mode"], panel=val, scaler=val.scaler, out_root=OUT / setup,
                    use_anchor=spec["use_anchor"], model_args=args)
    else:
        tune_root = OUT / "_tune2024" / setup                # settings for 2024 are chosen on 2023
        rolling.run(model, device, [2024], spec["target_mode"], panel=val, scaler=val.scaler, out_root=tune_root,
                    tune_only=True, use_anchor=spec["use_anchor"], model_args=args)
        sel = json.loads((tune_root / model / "origin_2024" / "selection.json").read_text())
        locked = {g: {k: v["chosen"][k] for k in ("hidden", "dropout", "epochs")} for g, v in sel.items()}
        from s6_data import load_test_bundle
        test = load_test_bundle()
        _check(test, "test bundle")
        shrink = json.loads(LOCK.read_text())["models"]["shrinkage"] if spec["use_anchor"] else None
        rolling.run(model, device, [2024], spec["target_mode"], panel=test, scaler=test.scaler, out_root=OUT / setup,
                    locked=locked, use_anchor=spec["use_anchor"], model_args=args, locked_shrinkage=shrink)
        sel_out = OUT / setup / model / "origin_2024" / "selection.json"    # keep the 2023 choice next to the result
        sel_out.write_text(json.dumps({g: dict(v, tuned_on=2023) for g, v in sel.items()}, indent=1))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (run_dir(setup, model, origin) / "provenance.json").write_text(json.dumps({
        "setup": setup, "model": model, "origin": origin, "device": device, "torch": torch.__version__,
        "python": platform.python_version(), "code_commit": commit or "unknown (no git)",
        "seconds": round(time.time() - start)}, indent=1))
    print(f"finished {setup}/{model}/{origin} in {time.time() - start:.0f}s", flush=True)


def status() -> None:
    for setup, model, origin in sorted(set(plan()) | set(REUSED)):
        state = "complete" if done(setup, model, origin) else "pending"
        src = "reused" if (setup, model, origin) in REUSED else "s7"
        print(f"{setup:6s} {model:6s} {origin}  {state:8s} ({src})")
    pending = [r for r in plan() if not done(*r)]
    print(f"\n{len(pending)} of {len(plan())} runs pending")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    a = sub.add_parser("all")
    a.add_argument("--device", default="cuda")
    o = sub.add_parser("one")
    o.add_argument("--setup", choices=sorted(SETUPS), required=True)
    o.add_argument("--model", choices=STGNN + ["mtgnn"], required=True)
    o.add_argument("--origin", type=int, choices=ORIGINS, required=True)
    o.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.cmd == "status":
        status()
    elif args.cmd == "all":
        for run in plan():
            one(*run, args.device)
        status()
    else:
        one(args.setup, args.model, args.origin, args.device)


if __name__ == "__main__":
    main()
