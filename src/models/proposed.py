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
        self.version = version
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
        self.head = nn.Linear(hidden, outputs)
        if version >= 2:
            self.alpha = nn.Parameter(torch.zeros(outputs))

    def forward(self, window: ProposedWindow) -> torch.Tensor:
        encoded = self.input(window.x).relu()
        if self.version >= 1:
            frames = []
            for t, frame in enumerate(encoded):
                messages = []
                for k, (conv, edge, attr) in enumerate(zip(self.relation_gat, window.edges[t], window.edge_attrs[t])):
                    messages.append(conv(frame, edge, attr))
                stack = torch.stack(messages, dim=0)  # [relations, schools, hidden]
                scores = torch.einsum("rnh,rh->rn", stack.tanh(), self.relation_score)
                weights = scores.softmax(dim=0)
                frames.append((stack * weights.unsqueeze(-1)).sum(dim=0))
            encoded = torch.stack(frames)
        states, _ = self.gru(self.dropout(encoded))
        prediction = self.head(self.dropout(states))
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
