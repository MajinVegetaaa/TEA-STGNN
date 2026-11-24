"""Non-training Kaggle checks for S5 Round 2 and a tiny synthetic forward/backward smoke test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch_geometric
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/train"))
sys.path.insert(0, str(ROOT / "src/models"))
from data_s5_round2 import load_validation_bundle  # noqa: E402
from proposed import ProposedSTGNN, ProposedWindow  # noqa: E402
from proposed_round2 import PositiveGateSTGNN, QueryAwareRelationSTGNN  # noqa: E402


def synthetic_smoke(device: str) -> None:
    gen = torch.Generator().manual_seed(2718)
    years, nodes, features, relations, hidden = 4, 6, 5, 2, 8
    x = torch.randn(years, nodes, features, generator=gen, device=device)
    edges, attrs = [], []
    for _ in range(years):
        year_edges, year_attrs = [], []
        for _ in range(relations):
            edge = torch.randint(nodes, (2, 12), generator=gen, device=device)
            year_edges.append(edge)
            year_attrs.append(torch.randn(edge.shape[1], 2, generator=gen, device=device))
        edges.append(year_edges)
        attrs.append(year_attrs)
    window = ProposedWindow(x, edges, attrs)
    models = [ProposedSTGNN(features, hidden, 1, version=0),
              ProposedSTGNN(features, hidden, 1, version=1, relations=relations),
              QueryAwareRelationSTGNN(features, hidden, 1, relations=relations),
              PositiveGateSTGNN(features, hidden, 1, relations=relations),
              ProposedSTGNN(features, hidden, 10, version=1, relations=relations)]
    for i, model in enumerate(models):
        model = model.to(device)
        output = model(window)
        expected = 10 if i == len(models) - 1 else 1
        if tuple(output.shape) != (years, nodes, expected) or not torch.isfinite(output).all():
            raise RuntimeError(f"synthetic smoke shape/numerics failed for {type(model).__name__}")
        output.square().mean().backward()
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
            raise RuntimeError(f"synthetic smoke gradient failed for {type(model).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "data/graph_v2/pyg/s5_validation")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if not torch.__version__.startswith("2.12.1"):
        raise RuntimeError(f"expected PyTorch 2.12.1, found {torch.__version__}")
    if torch_geometric.__version__ != "2.8.1":
        raise RuntimeError(f"expected PyG 2.8.1, found {torch_geometric.__version__}")
    import torch_scatter
    import torch_sparse
    import torch_geometric_temporal
    from importlib.metadata import version
    temporal_version = version("torch-geometric-temporal")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Kaggle GPU accelerator and restart the session")
    panel = load_validation_bundle(args.bundle)
    if panel.years != list(range(2018, 2024)) or panel.val_index != 5:
        raise RuntimeError("validation bundle has an unexpected temporal window")
    synthetic_smoke(args.device)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}; CUDA runtime: {torch.version.cuda}")
    print(f"PyG: {torch_geometric.__version__}; torch_scatter: {torch_scatter.__version__}; "
          f"torch_sparse: {torch_sparse.__version__}; PyG Temporal: {temporal_version}")
    print(f"Device: {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}")
    print(f"Validation years: {panel.years}; fingerprint: {panel.fingerprint}")
    print("Synthetic forward/backward smoke: PASS")


if __name__ == "__main__":
    main()
