"""Phase 5 GCN + LSTM benchmark on the verified expanded snapshot."""
import argparse
import hashlib
import json
import io
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data/graph_expanded/pyg/snapshots.pt"
BASE = ROOT / "results/expanded"
LAYERS = ["district", "geographic", "demographic", "region"]
EPOCHS = 30
HIDDEN = 32
LR = 0.003


class Model(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gcn = nn.Linear(features, HIDDEN)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, aggregated):
        encoded = self.gcn(aggregated).relu()
        sequence, _ = self.lstm(encoded)
        return self.head(sequence).squeeze(-1)


def graph_inputs(snapshot, layers, family="gcn_lstm"):
    # Sum overlapping relation weights before adding one self-loop per campus.
    stores = [snapshot["campus", layer, "campus"] for layer in layers]
    x = snapshot["campus"].x
    edges = torch.cat([store.edge_index for store in stores], dim=1)
    weights = torch.cat([store.edge_weight for store in stores])
    if family == "graphwavenet":
        degree = weights.new_zeros(x.size(0)).index_add_(0, edges[0], weights)
        weights = weights / degree[edges[0]].clamp_min(1e-12)
    else:
        edges, weights = gcn_norm(edges, weights, x.size(0), add_self_loops=family != "gconvgru")
    adjacency = torch.sparse_coo_tensor(edges.flip(0), weights, (len(x), len(x))).coalesce()
    # A_norm X is independent of trainable weights. Projecting it in Model
    # computes the same convolution as A_norm X W + bias at every epoch.
    support = adjacency.to_sparse_csr()
    if family == "gconvgru":
        return torch.cat((x, -(support @ x)), -1), support
    if family == "graphwavenet":
        return x, support
    return support @ x


def fit(x, labels, eligible, stop, seed, device, epochs, tune, family="gcn_lstm"):
    torch.manual_seed(seed)
    from phase5_cells import TGCN, GConvGRU, GraphWaveNet
    constructors = {"gcn_lstm": Model, "tgcn": lambda f: TGCN(f, HIDDEN),
                    "a3tgcn": lambda f: TGCN(f, HIDDEN, attention=True),
                    "gconvgru": lambda f: GConvGRU(f//2, HIDDEN),
                    "graphwavenet": lambda f: GraphWaveNet(f, HIDDEN)}
    model = constructors[family](x.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = labels[1:stop]
    center, scale = history[history.isfinite()].mean(), history[history.isfinite()].std()
    observed = history.isfinite()
    best = (float("inf"), 0)
    for epoch in range(1, epochs + 1):
        pred = model(x[:stop])[1:]
        loss = ((pred[observed] - (history[observed] - center) / scale) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if tune:
            with torch.no_grad():
                forecast = model(x[:stop + 1])[-1] * scale + center
                error = (forecast[eligible[stop]] - labels[stop, eligible[stop]]).abs().mean().item()
                best = min(best, (error, epoch))
    with torch.no_grad():
        prediction = model(x[:stop + 1])[-1] * scale + center
    return best, prediction.cpu().numpy(), model, center.item(), scale.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", choices=["gcn_lstm","tgcn","gconvgru","a3tgcn","graphwavenet"], default="gcn_lstm")
    args = parser.parse_args()
    model_name = {"tgcn":"T-GCN","gconvgru":"GConvGRU","a3tgcn":"A3TGCN","graphwavenet":"GraphWaveNet (fixed supports)","gcn_lstm":"GCN + LSTM"}[args.model]
    torch.set_num_threads(2)
    out = args.out / ("smoke" if args.smoke else "benchmark")
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(SNAP, weights_only=False, map_location="cpu")
    snaps = data["snapshots"]
    split = data["split"]
    assert data["years"] == split["train_years"] + split["val_years"] + split["test_years"]
    val = len(split["train_years"])
    targets = ["graduation_4yr"] if args.smoke else data["targets"]
    epochs = 2 if args.smoke else EPOCHS
    seeds = [0] if args.smoke else split["seeds"]
    protocol = dict(model=model_name, hidden=HIDDEN, learning_rate=LR, epochs=epochs,
                    validation_seed=0, split=split, device=args.device, torch_version=torch.__version__,
                    snapshot_sha256=hashlib.sha256(SNAP.read_bytes()).hexdigest(), targets=targets)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    cache = {}

    def features(layers):
        key = tuple(layers)
        if key not in cache:
            frames = [graph_inputs(s, layers, args.model) for s in snaps]
            if args.model in ("graphwavenet","gconvgru"):
                from phase5_cells import GraphWindow
                values, supports = zip(*frames)
                cache[key] = GraphWindow(torch.stack(values), list(supports))
            else:
                cache[key] = torch.stack(frames)
        return cache[key].to(args.device)

    def outcomes(target):
        j = data["targets"].index(target)
        return [torch.stack([s["campus"][key][:, j] for s in snaps]).to(args.device)
                for key in ("y", "eval_mask")]

    choices = []
    tuning = []
    for target in targets:
        layers = LAYERS.copy()
        if target in data["academic_relation"]:
            layers.append(data["academic_relation"][target])
        labels, masks = outcomes(target)
        candidates = []
        for name, relations in [("combined", layers)] + [(layer, [layer]) for layer in layers]:
            best, *_ = fit(features(relations), labels, masks, val, 0, args.device, epochs, True, args.model)
            row = dict(target=target, graph=name, relations=relations, val_mae=best[0], epochs=best[1])
            assert np.isfinite(best[0]) and best[1] > 0
            candidates.append(row)
        choices.extend([candidates[0], min(candidates[1:], key=lambda r: r["val_mae"])])
        tuning.extend(candidates)
        print("Validation complete:", target, flush=True)
    pd.DataFrame(tuning).to_csv(out / "validation_grid.csv", index=False)
    (out / "selection_locked.json").write_text(json.dumps(choices, indent=2))
    if args.smoke:
        print("Smoke validation passed", flush=True)
        return
    reference = pd.read_csv(BASE / "selected_test_predictions.csv", dtype={"campus_id": str})
    locked = pd.read_csv(BASE / "feature_selection_locked.csv")[["target", "input_set"]]
    reference = reference.merge(locked, on=["target", "input_set"], validate="many_to_one")
    predictions, scores = [], []
    archive_path = out / "checkpoints.partial.zip"
    with ZipFile(archive_path, "w"):
        pass
    for row in choices:
        target = row["target"]
        j = data["targets"].index(target)
        labels, masks = outcomes(target)
        mask = masks[-1].cpu().numpy()
        ids = np.array(data["campus_ids"])[mask]
        for seed in seeds:
            _, pred, model, center, scale = fit(features(row["relations"]), labels, masks, val+1, seed, args.device, row["epochs"], False, args.model)
            truth = labels[-1].cpu().numpy()[mask]
            ref = reference[(reference.target == target) & (reference.seed == seed)].set_index("campus_id")
            assert ref.index.is_unique and set(ref.index) == set(ids)
            ref = ref.loc[ids]
            np.testing.assert_allclose(ref.y_true, truth, atol=1e-5)
            assert np.isfinite(pred[mask]).all()
            err = pred[mask] - ref.y_true.to_numpy()
            mae = np.abs(err).mean()
            persist = snaps[-1]["campus"].y_lag1[:, j].numpy()[mask]
            scores.append(dict(target=target, graph=row["graph"], seed=seed, n=len(ids),
                               coverage_pct=100*len(ids)/int(snaps[-1]["campus"].present.sum()),
                               mae=mae, rmse=np.sqrt(np.mean(err**2)),
                               r2=1-np.sum(err**2)/np.sum((ref.y_true-ref.y_true.mean())**2),
                               delta_vs_phase3=mae-(ref.y_pred-ref.y_true).abs().mean(),
                               delta_vs_persistence=mae-np.abs(persist-ref.y_true).mean()))
            predictions.append(pd.DataFrame(dict(model=model_name, graph=row["graph"], target=target, seed=seed,
                campus_id=ids, year=data["years"][-1], split="test", y_true=ref.y_true.to_numpy(), y_pred=pred[mask])))
            buffer = io.BytesIO()
            torch.save(dict(state_dict=model.state_dict(), center=center, scale=scale, selection=row), buffer)
            with ZipFile(archive_path, "a") as archive:
                archive.writestr(f"{target}_{row['graph']}_{seed}.pt", buffer.getvalue())
    archive_path.replace(out / "checkpoints.zip")
    pd.concat(predictions).to_csv(out/"predictions.csv", index=False)
    metrics = pd.DataFrame(scores)
    metrics.to_csv(out/"per_seed.csv", index=False)
    grouped = metrics.groupby(["target","graph"])
    grouped.mean(numeric_only=True).drop(columns="seed").assign(mae_sd=grouped.mae.std(ddof=0)).to_csv(out/"comparison.csv")
    print("Completed", len(choices), "fixed target/graph comparisons", flush=True)


if __name__ == "__main__":
    main()
