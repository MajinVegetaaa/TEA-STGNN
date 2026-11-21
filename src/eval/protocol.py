"""The evaluation protocol in code (configs/protocol.yaml). Import this; do not re-implement metrics elsewhere.

    from protocol import load, scored_rows, from_change, selection_score, choose, metrics, paired_bootstrap

Metrics and the paired bootstrap are the ones that produced the Phase 3 numbers (src/eval/evaluate.py), re-exported
so every step scores the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate import PERSIST, metrics, paired_bootstrap   # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parents[2]


def load() -> dict:
    """The protocol, with the split (years, seeds) merged in from configs/split.yaml."""
    cfg = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    cfg["split"] = yaml.safe_load((ROOT / cfg["split"]).read_text())
    return cfg


def scored_rows(y: pd.Series, y_last_year: pd.Series) -> pd.Series:
    """Rows every model is scored on: the target and its previous-year value are both observed."""
    return y.notna() & y_last_year.notna()


def to_change(y: pd.Series | np.ndarray, y_last_year: pd.Series | np.ndarray):
    """Training label for neural models (D6): this year's value minus last year's."""
    return y - y_last_year


def from_change(change: pd.Series | np.ndarray, y_last_year: pd.Series | np.ndarray):
    """Forecast from a predicted change: last year's value plus the change. A zero change is persistence."""
    return y_last_year + change


def selection_score(model_mae: pd.DataFrame, persistence_mae: pd.Series) -> tuple[float, float]:
    """Final-model score on the selection year.

    model_mae: rows = seeds, columns = targets (validation MAE). persistence_mae: per target (deterministic).
    Returns (mean, sd over seeds) of the per-seed average of MAE / persistence MAE. Below 1 beats persistence.
    """
    missing = set(model_mae.columns) - set(persistence_mae.index)
    if missing:
        raise ValueError(f"No persistence MAE for {sorted(missing)}")
    per_seed = (model_mae / persistence_mae[model_mae.columns]).mean(axis=1)
    return float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0


def choose(candidates: pd.DataFrame) -> str:
    """Pick the final model. candidates: index = name, columns score, sd, complexity (lower = simpler).

    Everything within one seed SD of the best score counts as tied; among those the simplest wins, then the score.
    """
    best = candidates.score.idxmin()
    tied = candidates[candidates.score <= candidates.at[best, "score"] + candidates.at[best, "sd"]]
    return str(tied.sort_values(["complexity", "score"]).index[0])
