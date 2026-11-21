"""Edge weights and edge features for graph v2 (configs/graph_v2.yaml). numpy/pandas only, so it is testable
without PyTorch Geometric; build_graph_data.py, validate_graph.py and to_pyg.py all call these functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def local_scales(distance_matrix: np.ndarray, k: int, floor_km: float) -> np.ndarray:
    """Self-tuning scale per school: distance to its k-th nearest other school, floored. Diagonal must be inf."""
    return np.maximum(np.sort(distance_matrix, axis=1)[:, k - 1], floor_km)


def self_tuning_weight(distance: np.ndarray, scale_src: np.ndarray, scale_dst: np.ndarray) -> np.ndarray:
    """exp(-d^2 / (s_i * s_j)) (Zelnik-Manor & Perona 2004). A neighbour at the typical local distance gets ~0.37."""
    return np.exp(-(np.asarray(distance) ** 2) / (np.asarray(scale_src) * np.asarray(scale_dst)))


def training_sigma(edges: pd.DataFrame, train_years: list[int]) -> float:
    """One kernel width for all years: the median edge distance over the training-year graphs."""
    d = edges.loc[edges.year.isin(train_years), "distance"]
    if d.empty:
        raise ValueError("No training-year edges to estimate sigma from")
    return float(np.median(d))


def gaussian_weight(distance: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(np.asarray(distance) ** 2) / (2 * sigma ** 2))


def distance_stats(edges: pd.DataFrame, train_years: list[int]) -> tuple[float, float]:
    """Mean and SD of the raw distance on training years, for standardising the edge feature."""
    d = edges.loc[edges.year.isin(train_years), "distance"].dropna()
    if d.empty:
        return 0.0, 1.0                     # relation without distances (district, region, temporal)
    sd = float(d.std(ddof=0))
    return float(d.mean()), sd if sd > 0 else 1.0


def edge_attributes(edges: pd.DataFrame, stats: tuple[float, float]) -> np.ndarray:
    """[weight, distance_z] per edge; distance_z = 0 where a relation has no distance."""
    mean, sd = stats
    z = ((edges.distance - mean) / sd).fillna(0.0).to_numpy(dtype=float)
    return np.column_stack([edges.weight.to_numpy(dtype=float), z])
