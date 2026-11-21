"""Which graph a script works on, from the environment (one place, so every script agrees).

    TEA_GRAPH=v2        -> data/graph_v2/        expanded inputs, v2 weights (configs/graph_v2.yaml)   [current]
    TEA_GRAPH=expanded  -> data/graph_expanded/  expanded inputs, v1 weights (the Phase 5 graph; do not rebuild)
    (unset)             -> data/graph/           original graph (frozen)
TEA_EXPANDED=1 is the older spelling of TEA_GRAPH=expanded and still works.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {"original": "graph", "expanded": "graph_expanded", "v2": "graph_v2"}


def name() -> str:
    graph = os.environ.get("TEA_GRAPH") or ("expanded" if os.environ.get("TEA_EXPANDED") == "1" else "original")
    if graph not in FOLDERS:
        raise ValueError(f"TEA_GRAPH must be one of {sorted(FOLDERS)}, not {graph!r}")
    return graph


def expanded() -> bool:
    """Uses the expanded inputs and ten targets (everything except the original frozen graph)."""
    return name() != "original"


def folder() -> Path:
    return ROOT / "data" / FOLDERS[name()]


def features_file() -> Path:
    """History features for this graph's nodes: original, expanded (publication-filtered record) or D1 (v2)."""
    files = {"original": "temporal_features.csv", "expanded": "expanded_temporal_features.csv",
             "v2": "d1_temporal_features.csv"}
    return ROOT / "data" / "processed" / files[name()]
