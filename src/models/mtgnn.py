"""MTGNN (Wu et al., KDD 2020, "Connecting the Dots"), adapted to yearly school snapshots. Plain PyTorch, following
the official code (github.com/nnzhan/MTGNN).

Input: a GraphWindow (x [years, schools, features], supports = one sparse row-normalised (A + I) per year built from
our v2 relations). Output: [years, schools, outputs]; the output at year t uses only years <= t (causal in time), so
it is the forecast for year t (inputs at year t are history from years before t).

Adaptations (stated in docs/MODELS.md): 7 yearly steps, so temporal kernels {1, 2, 3} with dilations 1 and 2
(receptive field 7 years) instead of {2, 3, 6, 7}; the learned graph is used together with the yearly v2 relations
(the paper's option when a graph is known); layer normalisation over channels per school and year (schools appear
and disappear between years); the official random jitter in the top-k link choice is applied only while
training, so evaluation is deterministic.
"""
from __future__ import annotations

import torch
from torch import nn


class GraphLearner(nn.Module):
    """Learned uni-directional adjacency from two school embeddings, top-k per school (MTGNN graph_constructor)."""

    def __init__(self, nodes: int, dim: int = 40, alpha: float = 3.0, k: int = 20):
        super().__init__()
        self.emb1, self.emb2 = nn.Embedding(nodes, dim), nn.Embedding(nodes, dim)
        self.lin1, self.lin2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.alpha, self.k = alpha, k

    def forward(self) -> torch.Tensor:
        v1 = torch.tanh(self.alpha * self.lin1(self.emb1.weight))
        v2 = torch.tanh(self.alpha * self.lin2(self.emb2.weight))
        adj = torch.relu(torch.tanh(self.alpha * (v1 @ v2.T - v2 @ v1.T)))
        k = min(self.k, adj.shape[0])
        # the official code adds a random jitter (x0.01) before choosing the top k per row; here only while training,
        # so a scored forecast never depends on the random state (deterministic top k when evaluating)
        jitter = torch.rand_like(adj) * 0.01 if self.training else 0.0
        top = (adj + jitter).topk(k, dim=1).indices
        mask = torch.zeros_like(adj).scatter_(1, top, 1.0)
        return adj * mask


def row_normalise(adj: torch.Tensor) -> torch.Tensor:
    """MTGNN's (A + I) / row sum, for a dense learned adjacency."""
    a = adj + torch.eye(adj.shape[0], device=adj.device)
    return a / a.sum(1, keepdim=True)


class MixProp(nn.Module):
    """Mix-hop propagation: h_k = beta * x + (1 - beta) * A h_{k-1}, k = 1..depth; output = W [h_0, ..., h_depth]."""

    def __init__(self, c_in: int, c_out: int, depth: int = 2, beta: float = 0.05):
        super().__init__()
        self.depth, self.beta = depth, beta
        self.out = nn.Linear((depth + 1) * c_in, c_out)

    def forward(self, x: torch.Tensor, adjacency) -> torch.Tensor:
        """x [years, schools, channels]; adjacency: one dense matrix for every year, or a list of one per year."""
        per_year = adjacency if isinstance(adjacency, (list, tuple)) else [adjacency] * x.shape[0]
        outs = []
        for xt, a in zip(x, per_year):
            hops, h = [xt], xt
            for _ in range(self.depth):
                h = self.beta * xt + (1 - self.beta) * (a @ h)
                hops.append(h)
            outs.append(self.out(torch.cat(hops, -1)))
        return torch.stack(outs)


class DilatedInception(nn.Module):
    """Parallel causal temporal convolutions of several widths at one dilation, concatenated and projected."""

    def __init__(self, c_in: int, c_out: int, kernels=(1, 2, 3), dilation: int = 1):
        super().__init__()
        self.kernels, self.dilation = kernels, dilation
        self.branches = nn.ModuleList(nn.Linear(k * c_in, c_out) for k in kernels)
        self.project = nn.Linear(len(kernels) * c_out, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for k, branch in zip(self.kernels, self.branches):
            taps = [shift(x, j * self.dilation) for j in range(k)]       # x_t, x_{t-d}, ..., zero-padded in the past
            outs.append(branch(torch.cat(taps, -1)))
        return self.project(torch.cat(outs, -1))


def shift(x: torch.Tensor, steps: int) -> torch.Tensor:
    """x shifted `steps` years into the future along the first axis, zeros before the start (causal)."""
    if steps == 0:
        return x
    return torch.cat([torch.zeros_like(x[:steps]), x[:-steps]], 0)


class MTGNN(nn.Module):
    def __init__(self, features: int, hidden: int, outputs: int, nodes: int, dropout: float = 0.0,
                 embedding: int = 40, alpha: float = 3.0, top_k: int = 20, beta: float = 0.05, depth: int = 2,
                 kernels=(1, 2, 3), dilations=(1, 2)):
        super().__init__()
        self.start = nn.Linear(features, hidden)
        self.learner = GraphLearner(nodes, embedding, alpha, top_k)
        self.filters = nn.ModuleList(DilatedInception(hidden, hidden, kernels, d) for d in dilations)
        self.gates = nn.ModuleList(DilatedInception(hidden, hidden, kernels, d) for d in dilations)
        self.skips = nn.ModuleList(nn.Linear(hidden, hidden) for _ in dilations)
        # per layer: learned graph in both directions (as in MTGNN) + the yearly v2 relations
        self.learned_fwd = nn.ModuleList(MixProp(hidden, hidden, depth, beta) for _ in dilations)
        self.learned_bwd = nn.ModuleList(MixProp(hidden, hidden, depth, beta) for _ in dilations)
        self.given = nn.ModuleList(MixProp(hidden, hidden, depth, beta) for _ in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in dilations)
        self.skip_end = nn.Linear(hidden, hidden)
        self.end = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, outputs))
        self.dropout = nn.Dropout(dropout)

    def forward(self, window) -> torch.Tensor:
        h = self.start(window.x)
        adj = self.learner()
        fwd, bwd = row_normalise(adj), row_normalise(adj.T)
        skip = 0
        for i in range(len(self.filters)):
            residual = h
            h = torch.tanh(self.filters[i](h)) * torch.sigmoid(self.gates[i](h))
            h = self.dropout(h)
            skip = skip + self.skips[i](h)
            h = self.learned_fwd[i](h, fwd) + self.learned_bwd[i](h, bwd) + self.given[i](h, window.supports)
            h = self.norms[i](h + residual)
        return self.end(skip + self.skip_end(h))


def yearly_supports(snapshots, relations: list[str]) -> list[torch.Tensor]:
    """One sparse row-normalised (A + I) per year: the relations' weights summed (the v2 graph MTGNN may use)."""
    supports = []
    for s in snapshots:
        n = s["campus"].x.shape[0]
        stores = [s["campus", r, "campus"] for r in relations]
        index = torch.cat([st.edge_index for st in stores] + [torch.arange(n).repeat(2, 1)], 1)
        weight = torch.cat([st.edge_weight for st in stores] + [torch.ones(n)])
        dense_rows = torch.zeros(n).index_add_(0, index[0], weight)
        a = torch.sparse_coo_tensor(index, weight / dense_rows[index[0]], (n, n)).coalesce()
        supports.append(a.to_sparse_csr())
    return supports
