"""Phase 4 smoke test: one small spatio-temporal GNN trains end to end on the PyG snapshots.

Model: per year, one SAGEConv per relation (HeteroConv, mean over relations) -> GRU cell across years (node i is the
same campus in every snapshot) -> linear head. One target (default ACT composite), level version, scaled by the
training-years mean/sd. Trained on target years 2019-2022 (2018 is history only), validated on 2023 on exactly the
Phase 3 rows (eval_mask), compared with persistence and with predicting last year's statewide mean.
Not a tuned model: it only proves the data, masks and training loop work, and times one run.
Runs on the GPU when one is available (--device to override). Writes predictions (hand-off format) to --out. Run from the repo root:
  python3 src/models/smoke_test.py --target act_composite --out <folder>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data" / "graph" / "pyg" / "snapshots.pt"
LAYERS = ["district", "geographic", "demographic", "region"]


class STGNN(nn.Module):
    def __init__(self, n_in: int, relations: list[str], hidden: int = 64):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.conv = HeteroConv({("campus", r, "campus"): SAGEConv(hidden, hidden) for r in relations}, aggr="mean")
        self.gru = nn.GRUCell(2 * hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, snaps, relations):
        h = None
        outs = []
        for s in snaps:
            x = torch.relu(self.inp(s["campus"].x))
            ei = {("campus", r, "campus"): s["campus", r, "campus"].edge_index for r in relations}
            z = self.conv({"campus": x}, ei)["campus"]
            z = torch.cat([x, torch.relu(z)], dim=1)       # own features + neighbours
            h = self.gru(z, h)
            outs.append(self.head(h).squeeze(1))
        return outs                                        # one prediction vector per snapshot year


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="act_composite")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    D = torch.load(SNAP, weights_only=False)
    dev = torch.device(a.device)
    snaps, years, j = [s.to(dev) for s in D["snapshots"]], D["years"], D["targets"].index(a.target)
    rel = LAYERS + ([D["academic_relation"][a.target]] if a.target in D["academic_relation"] else [])
    train_y = [y for y in D["split"]["train_years"] if y != years[0]]
    val_y = D["split"]["val_years"][0]
    upto = years.index(val_y) + 1                          # run the sequence through the validation year only

    tr_vals = torch.cat([s["campus"].y[s["campus"].y_mask[:, j], j] for s in snaps if s.year in train_y])
    mu, sd = tr_vals.mean(), tr_vals.std()
    model = STGNN(snaps[0]["campus"].x.shape[1], rel).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        opt.zero_grad()
        outs = model(snaps[:upto], rel)
        loss = 0
        for s, o in zip(snaps[:upto], outs):
            if s.year in train_y:
                m = s["campus"].y_mask[:, j]
                loss = loss + ((o[m] - (s["campus"].y[m, j] - mu) / sd) ** 2).mean()
        loss = loss / len(train_y)
        loss.backward()
        opt.step()
        if ep % 25 == 0 or ep == a.epochs - 1:
            model.eval()
            with torch.no_grad():
                s = snaps[upto - 1]
                m = s["campus"].eval_mask[:, j]
                p = model(snaps[:upto], rel)[-1] * sd + mu
                print(f"epoch {ep:3d}  train loss {loss.item():.4f}  val MAE {(p[m] - s['campus'].y[m, j]).abs().mean():.4f}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0

    model.eval()
    with torch.no_grad():
        s = snaps[upto - 1]
        m = s["campus"].eval_mask[:, j]
        p = (model(snaps[:upto], rel)[-1] * sd + mu)[m]
    y, lag1 = s["campus"].y[m, j], s["campus"].y_lag1[m, j]
    prev_mean = D["year_stats"][a.target]["mean"][val_y - 1]
    print(f"\n{a.target}, validation {val_y}, n={int(m.sum())}:  GNN MAE {(p - y).abs().mean():.4f}   "
          f"persistence {(lag1 - y).abs().mean():.4f}   last year's statewide mean {(prev_mean - y).abs().mean():.4f}")
    print(f"training time {secs:.1f}s for {a.epochs} epochs on {dev} ({secs / a.epochs:.2f}s/epoch)")

    p, y = p.cpu(), y.cpu()
    ids = [D["campus_ids"][i] for i in m.nonzero().flatten().tolist()]
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model": "smoke_stgnn", "seed": 0, "campus_id": ids, "year": val_y, "target": a.target,
                  "y_true": y.numpy(), "y_pred": p.numpy(), "split": "val"}).to_csv(a.out / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
