"""S1c tabular bar at origins 2022 and 2023, chosen the same way as for 2024 (post-test, user request 2026-11-28).

The 2024 S1c bar: every tabular candidate tuned on 2023 after training on years < 2023, the pick per target made on
2023 (persistence counted as a candidate), the picked model refit on years < 2024, forecast 2024. Its recorded 2023
score (0.930) was chosen on 2023 itself. Here the same procedure is replayed one and two years earlier, so every
year's S1c score is out of sample:
  origin s: train on target years < s-1, tune each candidate on s-1, pick per target on s-1, refit on years < s,
  forecast s.
The S1c runner (run_s1c.py -> run_experiment.py) is used unchanged; only its train/val/test years are set. The pick
uses numbers_to_beat.py unchanged, pointed at this origin's folder.

    .venv-expanded/bin/python src/s7/s1c_rolling.py check                      # 2024 years must reproduce S1c
    .venv-expanded/bin/python src/s7/s1c_rolling.py fit --origin 2023          # 6 runs (2 input sets x 3 modes)
    .venv-expanded/bin/python src/s7/s1c_rolling.py select --origin 2023       # picks + forecasts of year s
Writes results/s7_s1c_rolling/origin_<s>/{existing,all}/{level,relative,zscore}/, candidates.csv,
numbers_to_beat.csv, bar.json (val = s-1) and s1c_bar/predictions.csv (format of results/s6/s1c_bar).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/baselines"), str(ROOT / "src/data"), str(ROOT / "src/eval")]
OUT = ROOT / "results/s7_s1c_rolling"
SETS, MODES = ["existing", "all"], ["level", "relative", "zscore"]
ORIGINS = [2022, 2023]


def fit_one(origin: int, input_set: str, mode: str, out: Path, targets: list[str] | None = None) -> None:
    """One S1c run with train years < origin-1, validation origin-1, test origin."""
    import run_s1c as S
    B = S.B
    if (out / "predictions.csv").exists():
        raise SystemExit(f"{out} is complete; refusing to overwrite")
    S.INPUT_SET = input_set
    B.G = S.GRAPHS["v2"]
    B.feature_table = S.d1_feature_table
    B.TRAIN, B.VAL, B.TEST = list(range(2018, origin - 1)), [origin - 1], [origin]
    sys.argv = [sys.argv[0], "--mode", mode, "--out", str(out), "--targets", *(targets or S.TARGETS)]
    B.main()                                       # RES / absolute path = the absolute path


def check(scratch: Path) -> None:
    """With the 2024 years the wrapper must reproduce results/s1c/all/level for one target (val and test rows)."""
    import numpy as np
    import pandas as pd
    out = scratch / "check_2024_all_level"
    start = time.time()
    fit_one(2024, "all", "level", out, ["act_composite"])
    new = pd.read_csv(out / "predictions.csv", dtype={"campus_id": str})
    old = pd.concat(c[c.target == "act_composite"] for c in pd.read_csv(
        ROOT / "results/s1c/all/level/predictions.csv", dtype={"campus_id": str}, chunksize=500_000))
    key = ["model", "seed", "campus_id", "year", "split"]
    m = new.merge(old, on=key, suffixes=("_new", "_old"), validate="one_to_one")
    if len(m) != len(new) or len(m) != len(old):
        raise SystemExit(f"rows differ: new {len(new)}, old {len(old)}, matched {len(m)}")
    diff = float(np.abs(m.y_pred_new - m.y_pred_old).max())
    print(f"reproduction: {len(m)} rows, max |difference| {diff:.2e}, {time.time() - start:.0f}s")
    if not np.allclose(m.y_pred_new, m.y_pred_old, rtol=0, atol=1e-8):
        raise SystemExit("the wrapper does not reproduce S1c")


def select(origin: int) -> None:
    """Pick per target on origin-1 with numbers_to_beat.py, then keep the picks' forecasts of the origin year."""
    import pandas as pd
    import numbers_to_beat as NB
    root = OUT / f"origin_{origin}"
    NB.RUNS = root
    NB.main()
    picks = json.loads((root / "bar.json").read_text())["picks"]
    out = root / "s1c_bar"
    out.mkdir(exist_ok=True)
    frames = []
    for target, pick in picks.items():
        model, _, rest = pick.partition(" | ")
        if model.startswith("persistence"):
            frames.append(pd.DataFrame({"target": [target], "model": ["persistence"]}))
            continue
        mode, input_set = rest.split(" | ")
        p = pd.read_csv(root / input_set / mode / "predictions.csv", dtype={"campus_id": str})
        rows = p[(p.split == "test") & (p.model == model) & (p.target == target)]
        if rows.empty or set(rows.year) != {origin}:
            raise SystemExit(f"no {origin} forecasts for {pick}")
        frames.append(rows.assign(model=model))
    pd.concat(frames, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    (out / "picks.json").write_text(json.dumps({"origin": origin, "chosen_on": origin - 1, "picks": picks}, indent=1))
    print(f"\norigin {origin}: picks chosen on {origin - 1} -> {out / 'predictions.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--scratch", type=Path, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--origin", type=int, choices=ORIGINS, required=True)
    f.add_argument("--input-set", choices=SETS)
    f.add_argument("--mode", choices=MODES)
    s = sub.add_parser("select")
    s.add_argument("--origin", type=int, choices=ORIGINS, required=True)
    args = ap.parse_args()
    if args.cmd == "check":
        check(args.scratch)
    elif args.cmd == "select":
        select(args.origin)
    else:
        runs = [(i, m) for i in SETS for m in MODES
                if args.input_set in (None, i) and args.mode in (None, m)]
        for i, m in runs:
            out = OUT / f"origin_{args.origin}" / i / m
            if (out / "predictions.csv").exists():
                print(f"skip {out}: done")
                continue
            start = time.time()
            fit_one(args.origin, i, m, out)
            print(f"finished origin {args.origin} {i}/{m} in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
