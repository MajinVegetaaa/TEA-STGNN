"""Feature encoding fitted on training rows only."""
from __future__ import annotations

import pandas as pd


def train_only_onehot(frame: pd.DataFrame, fit_mask, columns: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot encode training-seen categories; unseen later values map to all-zero columns."""
    encoded = pd.DataFrame(index=frame.index)
    vocabulary = {}
    for column in columns:
        values = frame[column].astype(str)
        categories = sorted(values.loc[fit_mask].unique().tolist())
        vocabulary[column] = categories
        for category in categories:
            encoded[f"{column}_{category}"] = values.eq(category).astype(float)
    return encoded, vocabulary
