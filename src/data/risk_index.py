"""Academic Risk Index (0-100), a corrected version of last year's Eq. 1.

Last year's code (phase2code_ST_LLM_WGHTS_NODROP.py) is followed where it is sound: within-year percentile
ranks, a decline term, a weighted sum scaled to 0-100. Changes, each fixing a problem found in that code:
  1. Direction: risk = percentile for components where higher is worse (dropout, grade retention) and
     1 - percentile where lower is worse (scores, participation, college-ready). The old code used 1 - percentile
     for everything, so higher dropout meant LOWER risk.
  2. Missing stays missing: weights are re-normalised over the components a campus-year has, and the index is
     left empty when they carry < min_weight_coverage of the total. The old code filled missing with 0 risk.
  3. No circular weight tuning: fixed weights from configs/risk_index.yaml (the old code's defaults). The old
     code picked weights to correlate with dropout and retention, which are themselves components.
  4. Decline uses earlier years only: mean of up to 3 PRIOR years minus this year, worst across the academic
     scores, ranked within year. The old rolling mean included the current year.
  5. "Location" is dropped: it was a percentile of a campus-type code, which has no order.
The index summarises year t's outcomes, so it is a target for year t, never an input for year t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "risk_index.yaml").read_text())
WEIGHTS: dict[str, float] = {k: float(v) for k, v in CFG["weights"].items()}
MIN_COVERAGE = float(CFG["min_weight_coverage"])

HIGHER_IS_WORSE = ["dropout_9_12", "retention_rate"]
LOWER_IS_WORSE = ["sat_erw", "sat_math", "act_english", "act_math", "sat_act_above_criterion",
                  "sat_act_participation"]
DECLINE_FROM = ["sat_erw", "sat_math", "act_english", "act_math", "sat_act_above_criterion"]


def _pct(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("year")[col].rank(pct=True, method="average")


def decline_raw(m: pd.DataFrame, years: list[int]) -> pd.Series:
    """Largest drop below the school's own prior average (up to 3 earlier years), in percentile units.

    Each score is first turned into a within-year percentile so SAT and ACT drops are comparable.
    """
    grid = pd.MultiIndex.from_product([sorted(m.campus_id.unique()), years], names=["campus_id", "year"])
    drops = []
    for col in DECLINE_FROM:
        p = m.assign(_p=_pct(m, col)).set_index(["campus_id", "year"])["_p"].reindex(grid).unstack("year")
        prior = pd.concat([p.shift(k, axis=1) for k in (1, 2, 3)]).groupby(level=0).mean()   # mean of t-1..t-3
        drops.append((prior - p).clip(lower=0).stack(future_stack=True))
    d = pd.concat(drops, axis=1).max(axis=1, skipna=True)        # worst drop across scores
    d.index.names = ["campus_id", "year"]
    return d.reindex(pd.MultiIndex.from_frame(m[["campus_id", "year"]])).to_numpy()


def compute(m: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """-> DataFrame aligned with m: risk_index, its components (0-1, higher = riskier) and weight coverage."""
    comp = pd.DataFrame(index=m.index)
    for c in HIGHER_IS_WORSE:
        comp[c] = _pct(m, c)
    for c in LOWER_IS_WORSE:
        comp[c] = 1 - _pct(m, c)
    tmp = m[["campus_id", "year"]].copy()
    tmp["_d"] = decline_raw(m, years)
    comp["decline"] = tmp.groupby("year")["_d"].rank(pct=True, method="average")

    w = pd.Series(WEIGHTS)
    present = comp[w.index].notna()
    coverage = (present * w).sum(axis=1) / w.sum()
    score = (comp[w.index].fillna(0) * w).sum(axis=1) / (present * w).sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame({"risk_index": (100 * score).where(coverage >= MIN_COVERAGE),
                        "risk_index_coverage": coverage.round(4)}, index=m.index)
    for c in w.index:
        out[f"risk_c_{c}"] = comp[c]
    return out
