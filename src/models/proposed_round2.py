"""Isolated S5 Round 2 variants; the first-round models remain unchanged."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from proposed import ProposedSTGNN, ProposedWindow


class QueryAwareRelationSTGNN(ProposedSTGNN):
    """V1 with query-aware pooling over relation-specific GATv2 messages.

    The original V1's GAT, residual path, GRU and output heads are retained.
    Only the final relation pooling logit changes to depend on both the target
    school's encoded state and its message from each relation.
    """

    def __init__(self, *args, **kwargs):
        kwargs["version"] = 1
        super().__init__(*args, **kwargs)
        hidden = self.input.out_features
        self.query_proj = nn.Linear(hidden, hidden, bias=False)
        self.message_proj = nn.Linear(hidden, hidden, bias=True)
        self.relation_bias = nn.Parameter(torch.zeros(self.relations, hidden))
        self.relation_vector = nn.Parameter(torch.empty(hidden))
        nn.init.xavier_uniform_(self.relation_vector.unsqueeze(0))

    def _spatial_encode(self, window: ProposedWindow, return_attention: bool = False):
        encoded = self.input(window.x).relu()
        frame_outputs, attention_outputs = [], []
        for t, frame in enumerate(encoded):
            if len(window.edges[t]) != self.relations or len(window.edge_attrs[t]) != self.relations:
                raise ValueError(f"year index {t} has a relation count different from model setup")
            messages, available = [], []
            for conv, edge, attr in zip(self.relation_gat, window.edges[t], window.edge_attrs[t]):
                if (edge.ndim != 2 or edge.shape[0] != 2 or attr.ndim != 2
                        or attr.shape != (edge.shape[1], 2)):
                    raise ValueError("edge_index must be [2, E] and v2 edge_attr must be [E, 2]")
                messages.append(conv(frame, edge, attr))
                available.append(torch.bincount(edge[1], minlength=frame.shape[0]) > 0)
            available_t = torch.stack(available, dim=0)
            message_stack = torch.stack(messages, dim=0)
            message_stack = message_stack.masked_fill(~available_t.unsqueeze(-1), 0.0)
            query = self.query_proj(frame).unsqueeze(0)
            keys = self.message_proj(message_stack) + self.relation_bias[:, None, :]
            logits = torch.einsum("rnh,h->rn", torch.tanh(query + keys), self.relation_vector)
            logits = logits.masked_fill(~available_t, float("-inf"))
            has_relation = available_t.any(dim=0)
            logits = torch.where(has_relation.unsqueeze(0), logits, torch.zeros_like(logits))
            weights = logits.softmax(dim=0)
            frame_outputs.append(frame + (message_stack * weights.unsqueeze(-1)).sum(dim=0))
            attention_outputs.append(weights.masked_fill(~available_t, 0.0))
        return torch.stack(frame_outputs), torch.stack(attention_outputs)

    def forward(self, window: ProposedWindow) -> torch.Tensor:
        if window.x.ndim != 3:
            raise ValueError("window.x must have shape [years, schools, features]")
        if len(window.edges) != window.x.shape[0] or len(window.edge_attrs) != window.x.shape[0]:
            raise ValueError("one edge list and edge-attribute list are required per year")
        encoded, _ = self._spatial_encode(window)
        states, _ = self.gru(self.dropout(encoded))
        states = self.dropout(states)
        return torch.cat([head(states) for head in self.heads], dim=-1)

    def forward_with_attention(self, window: ProposedWindow):
        """Return predictions and [years, relations, schools] normalized weights."""
        encoded, weights = self._spatial_encode(window, return_attention=True)
        states, _ = self.gru(self.dropout(encoded))
        states = self.dropout(states)
        return torch.cat([head(states) for head in self.heads], dim=-1), weights


class PositiveGateSTGNN(nn.Module):
    """V2 correction with a positive, nonzero softplus scale at initialization."""

    def __init__(self, features: int, hidden: int, outputs: int, dropout: float = 0.0,
                 relations: int = 5, initial_scale: float = 0.1):
        super().__init__()
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")
        self.backbone = ProposedSTGNN(features, hidden, outputs, dropout=dropout, version=1,
                                      relations=relations)
        initial_logit = math.log(math.expm1(initial_scale))
        self.gate_logit = nn.Parameter(torch.full((outputs,), initial_logit))

    @property
    def gate(self) -> torch.Tensor:
        return F.softplus(self.gate_logit)

    def forward(self, window: ProposedWindow) -> torch.Tensor:
        return self.backbone(window) * self.gate


def prediction_parameter_grad_norm(model: nn.Module) -> float:
    """L2 norm of gradients outside the explicit V2 gate parameter."""
    params = [p.grad.detach().float().norm(2) for name, p in model.named_parameters()
              if name != "gate_logit" and p.grad is not None]
    if not params:
        return 0.0
    return float(torch.stack(params).norm(2).item())
