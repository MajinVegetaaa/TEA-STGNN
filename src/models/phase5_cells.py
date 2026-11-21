"""Temporal graph baselines for annual, weighted campus snapshots."""
import torch
from torch import nn


class TGCN(nn.Module):
    def __init__(self, features, hidden, attention=False):
        super().__init__()
        self.hidden = hidden
        self.project = nn.Linear(features, 3 * hidden)
        self.gates = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(3))
        self.attention = nn.Parameter(torch.zeros(3)) if attention else None
        self.readout = nn.Linear(hidden, 1)

    def forward(self, aggregated):
        projected = self.project(aggregated)
        state = projected.new_zeros(projected.shape[1], self.hidden)
        states = []
        for frame in projected:
            z, r, c = frame.chunk(3, dim=-1)
            prior = state if self.attention is None else torch.zeros_like(state)
            update = self.gates[0](torch.cat((z, prior), -1)).sigmoid()
            reset = self.gates[1](torch.cat((r, prior), -1)).sigmoid()
            proposal = self.gates[2](torch.cat((c, reset * prior), -1)).tanh()
            state = update * prior + (1 - update) * proposal
            states.append(state)
        sequence = torch.stack(states)
        if self.attention is not None:
            # Prefix-only, sliding three-period A3TGCN window.
            outputs = []
            for end in range(1, len(states) + 1):
                window = sequence[max(0, end-3):end]
                weights = self.attention[-len(window):].softmax(0)
                outputs.append((window * weights[:, None, None]).sum(0))
            sequence = torch.stack(outputs)
        return self.readout(sequence).squeeze(-1)


class GraphWindow:
    def __init__(self, x, supports):
        self.x = x
        self.supports = supports
        self.shape = x.shape

    def __getitem__(self, key):
        return GraphWindow(self.x[key], self.supports[key])

    def to(self, device):
        return GraphWindow(self.x.to(device), [a.to(device) for a in self.supports])


class GConvGRU(nn.Module):
    """Two-term Chebyshev filters for inputs and hidden states (lambda_max=2)."""
    def __init__(self, features, hidden, outputs=1):
        super().__init__()
        self.hidden = hidden
        self.input = nn.Linear(2 * features, 3 * hidden)
        self.gates = nn.Linear(2 * hidden, 2 * hidden)
        self.candidate = nn.Linear(2 * hidden, hidden)
        self.readout = nn.Linear(hidden, outputs)

    def forward(self, window):
        state = window.x.new_zeros(window.x.size(1), self.hidden)
        outputs = []
        for x, adjacency in zip(window.x, window.supports):
            z, r, c = self.input(x).chunk(3, -1)
            hz, hr = self.gates(torch.cat((state, -adjacency @ state), -1)).chunk(2, -1)
            update, reset = (z+hz).sigmoid(), (r+hr).sigmoid()
            gated = reset * state
            candidate = (c + self.candidate(torch.cat((gated, -adjacency @ gated), -1))).tanh()
            state = update * state + (1-update) * candidate
            outputs.append(self.readout(state).squeeze(-1))
        return torch.stack(outputs)


class GraphWaveNet(nn.Module):
    """Causal dilated gated blocks with residual, skip, and graph diffusion."""
    def __init__(self, features, hidden, outputs=1):
        super().__init__()
        self.start = nn.Linear(features, hidden)
        self.filters = nn.ModuleList(nn.Linear(2*hidden, 2*hidden) for _ in range(2))
        self.diffuse = nn.ModuleList(nn.Linear(3*hidden, hidden) for _ in range(2))
        self.skip = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
        self.end = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))

    def forward(self, window):
        sequence = self.start(window.x)
        skip = 0
        for dilation, gate, diffusion, project in zip((1, 2), self.filters, self.diffuse, self.skip):
            delayed = torch.cat((torch.zeros_like(sequence[:dilation]), sequence[:-dilation]), 0)
            f, g = gate(torch.cat((sequence, delayed), -1)).chunk(2, -1)
            gated = f.tanh() * g.sigmoid()
            skip = skip + project(gated)
            first = torch.stack([a @ x for a, x in zip(window.supports, gated)])
            second = torch.stack([a @ x for a, x in zip(window.supports, first)])
            sequence = sequence + diffusion(torch.cat((gated, first, second), -1))
        # Include final diffusion to retain both spatial blocks in the readout.
        return self.end(skip + sequence).squeeze(-1)
