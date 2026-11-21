"""Model families the harness can train. Each family turns the yearly snapshots into model input once
(prepare) and builds a fresh model (build). New models (MTGNN in S4, the proposed model in S5) are added here.

Every model maps its input to [years, schools, outputs]: output at year t is the forecast for year t from inputs that
only use years before t.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from phase5_cells import GConvGRU, GraphWaveNet, GraphWindow  # noqa: E402  (Phase 5 cells, reused unchanged)
from phase5_gcn_lstm import graph_inputs  # noqa: E402  (Phase 5 graph normalisation, reused unchanged)


class GCNLSTM(nn.Module):
    """Phase 5 GCN + LSTM with `outputs` heads (1 = single target; identical to the Phase 5 model)."""

    def __init__(self, features: int, hidden: int, outputs: int):
        super().__init__()
        self.gcn = nn.Linear(features, hidden)       # applied to A_norm X, i.e. a GCN layer
        self.lstm = nn.LSTM(hidden, hidden)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, aggregated: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(self.gcn(aggregated).relu())
        return self.head(sequence)


class Keep3D(nn.Module):
    """Phase 5 cells squeeze a single output to [years, schools]; the harness always uses [years, schools, outputs]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.unsqueeze(-1) if out.dim() == 2 else out


def windows(family: str):
    """Snapshots -> GraphWindow (inputs + one propagation matrix per year), as in Phase 5."""
    def prepare(snaps, rel):
        values, supports = zip(*[graph_inputs(s, rel, family) for s in snaps])
        return GraphWindow(torch.stack(values), list(supports))
    return prepare


@dataclass
class Family:
    prepare: Callable[[list, list[str]], torch.Tensor]       # (snapshots, relations) -> model input
    build: Callable[[int, int, int], nn.Module]              # (input features, hidden, outputs) -> model
    features: Callable[[torch.Tensor], int] = lambda x: x.shape[-1]


FAMILIES = {
    "gcn_lstm": Family(prepare=lambda snaps, rel: torch.stack([graph_inputs(s, rel, "gcn_lstm") for s in snaps]),
                       build=GCNLSTM),
    # GConvGRU's window holds [X, -A X]; its cell takes the original feature count
    "gconvgru": Family(prepare=windows("gconvgru"), build=lambda f, h, o: Keep3D(GConvGRU(f // 2, h, o))),
    "graphwavenet": Family(prepare=windows("graphwavenet"), build=lambda f, h, o: Keep3D(GraphWaveNet(f, h, o))),
}
