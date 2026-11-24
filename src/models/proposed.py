"""Proposed annual spatio-temporal GNN (S5), with additive architectural versions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


@dataclass
class ProposedWindow:
    x: torch.Tensor                         # [years, schools, features]
    edges: list[list[torch.Tensor]]         # [year][relation], [2, edges]
    edge_attrs: list[list[torch.Tensor]]    # [year][relation], [edges, 2]

    @property
    def shape(self):
        return self.x.shape

    def __getitem__(self, index):
        return ProposedWindow(self.x[index], self.edges[index], self.edge_attrs[index])

    def to(self, device):
        return ProposedWindow(self.x.to(device),
                              [[edge.to(device) for edge in year] for year in self.edges],
                              [[attr.to(device) for attr in year] for year in self.edge_attrs])


class ProposedSTGNN(nn.Module):
    """GRU temporal encoder with optional typed-neighbour attention.

    The GAT is applied independently to each relation and year. A learned softmax
    over relations combines messages before the temporal GRU. Version 2+ predicts
    a gated standardized correction; alpha=0 initially yields zero standardized
    change (the training-set mean change after de-normalization).
    """

    def __init__(self, features: int, hidden: int, outputs: int, dropout: float = 0.0,
                 version: int = 0, relations: int = 5):
        super().__init__()
        if version not in (0, 1, 2, 3):
            raise ValueError(f"unsupported proposed model version: {version}")
        if version >= 1 and relations < 1:
            raise ValueError("graph versions require at least one relation")
        self.version = version
        self.relations = relations
        self.input = nn.Linear(features, hidden)
        if version >= 1:
            self.relation_gat = nn.ModuleList(
                GATv2Conv(hidden, hidden, heads=1, concat=False, edge_dim=2,
                          add_self_loops=False, dropout=dropout)
                for _ in range(relations)
            )
            self.relation_score = nn.Parameter(torch.zeros(relations, hidden))
        self.gru = nn.GRU(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        # One independent scalar head per target; the encoder remains shared in v3.
        self.heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(outputs))
        if version >= 2:
            self.alpha = nn.Parameter(torch.zeros(outputs))

    def forward(self, window: ProposedWindow) -> torch.Tensor:
        if window.x.ndim != 3:
            raise ValueError("window.x must have shape [years, schools, features]")
        if not self.heads:
            raise ValueError("model must have at least one output head")
        if len(window.edges) != window.x.shape[0] or len(window.edge_attrs) != window.x.shape[0]:
            raise ValueError("one edge list and edge-attribute list are required per year")
        encoded = self.input(window.x).relu()
        if self.version >= 1:
            frames = []
            for t, frame in enumerate(encoded):
                if len(window.edges[t]) != self.relations or len(window.edge_attrs[t]) != self.relations:
                    raise ValueError(f"year index {t} has a relation count different from model setup")
                messages = []
                available = []
                for conv, edge, attr in zip(self.relation_gat, window.edges[t], window.edge_attrs[t]):
                    if edge.ndim != 2 or edge.shape[0] != 2 or attr.ndim != 2 or attr.shape != (edge.shape[1], 2):
                        raise ValueError("edge_index must be [2, E] and v2 edge_attr must be [E, 2]")
                    messages.append(conv(frame, edge, attr))
                    # Relations with no incoming edge for a node contribute no message.
                    available.append(torch.bincount(edge[1], minlength=frame.shape[0]) > 0)
                available = torch.stack(available, dim=0)  # [relations, schools]
                stack = torch.stack(messages, dim=0).masked_fill(~available.unsqueeze(-1), 0.0)
                scores = torch.einsum("rnh,rh->rn", stack.tanh(), self.relation_score)
                has_relation = available.any(dim=0)
                scores = scores.masked_fill(~available, float("-inf"))
                # Avoid softmax(-inf,...) when a school has no neighbours in any relation.
                scores = torch.where(has_relation.unsqueeze(0), scores, torch.zeros_like(scores))
                weights = scores.softmax(dim=0)
                # Residual path preserves each school's own encoded features.
                frames.append(frame + (stack * weights.unsqueeze(-1)).sum(dim=0))
            encoded = torch.stack(frames)
        states, _ = self.gru(self.dropout(encoded))
        states = self.dropout(states)
        prediction = torch.cat([head(states) for head in self.heads], dim=-1)
        if self.version >= 2:
            prediction = self.alpha * prediction
        return prediction


def prepare_proposed(snaps, relations: list[str]) -> ProposedWindow:
    values, edges, attrs = [], [], []
    for snap in snaps:
        values.append(snap["campus"].x)
        year_edges, year_attrs = [], []
        for relation in relations:
            store = snap["campus", relation, "campus"]
            year_edges.append(store.edge_index)
            year_attrs.append(store.edge_attr)
        edges.append(year_edges)
        attrs.append(year_attrs)
    return ProposedWindow(torch.stack(values), edges, attrs)
