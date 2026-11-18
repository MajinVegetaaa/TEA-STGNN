"""Collect independently verified Phase 5 metrics and document adaptations."""
import argparse
from pathlib import Path

import pandas as pd

from check_phase5 import ROOT, check

FAMILIES = ["gcn_lstm", "tgcn", "gconvgru", "a3tgcn", "graphwavenet"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/phase5")
    args = parser.parse_args()
    summaries = []
    for family in FAMILIES:
        folder = args.results / family / "benchmark"
        check(folder, ROOT/"data/graph_expanded/pyg/snapshots.pt", ROOT/"results/expanded")
        summaries.append(pd.read_csv(folder/"comparison.csv").assign(family=family))
    metrics = pd.concat(summaries, ignore_index=True)
    gpu_path = args.results/"gcn_lstm/comparison.csv"
    gpu_note = "GPU summary has not been supplied."
    if gpu_path.exists():
        gpu = pd.read_csv(gpu_path)
        cpu = metrics[metrics.family.eq("gcn_lstm")]
        keys = ["target", "graph"]
        matched = gpu.merge(cpu, on=keys, suffixes=("_gpu","_cpu"), validate="one_to_one")
        assert len(matched) == len(cpu) == len(gpu) == 20
        assert matched.n_gpu.eq(matched.n_cpu).all()
        baseline = pd.read_csv(ROOT/"results/expanded/final_selection_report.csv").set_index("target")
        import numpy as np
        assert np.isfinite(gpu.select_dtypes("number")).all().all()
        for col, delta in [("mae","delta_vs_phase3"),("persistence_mae","delta_vs_persistence")]:
            np.testing.assert_allclose(gpu.mae-gpu[delta], gpu.target.map(baseline[col]), atol=1e-5)
        max_gap = (matched.mae_gpu - matched.mae_cpu).abs().max()
        gpu_note = (f"The [Colab GCN + LSTM scores](gcn_lstm/comparison.csv) contain all 20 expected rows. "
                    f"Graph choices and campus counts match the CPU run throughout. The largest difference "
                    f"in reported MAE was {max_gap:.5f}. Both GPU graph variants had higher error than the "
                    f"Phase 3 baseline for all ten targets. This check used the supplied summary CSV.")
    display = pd.read_csv(ROOT/"results/expanded/final_selection_report.csv").set_index("target")[["mae"]]
    display.columns = ["Phase 3"]
    for family in FAMILIES:
        lock = pd.read_json(args.results/family/"benchmark/selection_locked.json")
        winner = lock.loc[lock.groupby("target").val_mae.idxmin(), ["target","graph"]]
        rows = metrics[metrics.family.eq(family)].merge(winner, validate="one_to_one")
        display[family] = rows.set_index("target").mae
    display = display.rename(columns={"gcn_lstm":"GCN + LSTM","tgcn":"T-GCN","gconvgru":"GConvGRU",
                                     "a3tgcn":"A3TGCN","graphwavenet":"GraphWaveNet*"})
    simple = ["| Target | " + " | ".join(display.columns) + " |", "|---|" + "---:|"*len(display.columns)]
    for target, values in display.iterrows():
        simple.append("| " + target.replace("_"," ") + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    table = ["| Model | Graph | Lower MAE than Phase 3 | Lower MAE than persistence |",
             "|---|---|---:|---:|"]
    for (model, combined), group in metrics.groupby(["family",metrics.graph.eq("combined")]):
        table.append(f"| {model} | {'Combined' if combined else 'Selected single relation'} | "
                     f"{(group.delta_vs_phase3 < 0).sum()}/10 | {(group.delta_vs_persistence < 0).sum()}/10 |")
    (args.results/"README.md").write_text("""# Phase 5 results

**Your GPU result** is [GCN + LSTM on Colab](gcn_lstm/comparison.csv).
The table below summarizes all five local CPU benchmarks. For the other four
models, the optional [Colab notebook](Phase5_Colab.ipynb) runs the same comparisons on a GPU.

Each number is the average 2024 prediction error
for a target. Lower is better. Compare models across the same row. Each family
uses the graph selected by its 2023 validation error. Phase 3 is the earlier
benchmark. The asterisk marks the fixed-support GraphWaveNet variant.

""" + "\n".join(simple) + """

## Your Colab run

""" + gpu_note + """

## Audit details

All five model families used the same expanded snapshot, ten targets, eligible
campuses and five test seeds. The independent checker reproduced per-seed and
mean MAE, RMSE, R², coverage and differences from the two benchmarks, including
the standard deviation of MAE across seeds. The table counts targets where the
mean 2024 MAE was lower. Units differ by target, so errors are reported
individually above. Full metrics are in each model folder under benchmark/comparison.csv.

""" + "\n".join(table) + """

## Protocol and model changes

Each target uses its eligible weighted graph relations. Selection compares all
single relations and tunes the training duration over 1–30 epochs on 2023 at
seed 0. Combined and best single-relation selections are locked before each
family's 2024 refit and scoring. Final refits use the training and validation
years. All families use width 32 and Adam at 0.003. The Phase 3 and earlier
GCN + LSTM test results were known before these four additional experiments.

- T-GCN uses graph convolution on the annual inputs with separate update, reset
  and candidate transformations. A3TGCN uses learned period weights over up to
  three annual T-GCN outputs. Each period uses its own year's graph and a zero
  initial state, with the softmax restricted to available history.
- GConvGRU uses two-term Chebyshev filters on both the inputs and recurrent
  states. With symmetric normalization and a spectral bound of two, the first
  nonconstant term is the negative normalized adjacency. Input propagation is
  cached and hidden-state propagation runs inside each cell.
- GraphWaveNet is a **fixed-support adaptation**. It uses annual weighted
  diffusion through two hops, gated causal convolutions at dilations 1 and 2,
  residual and skip paths, and a readout that includes the final diffusion.
  The short receptive field fits the seven annual snapshots. This run has no
  learned adaptive adjacency. Scores apply to this configured variant.
- The implementations use PyTorch and PyG with the existing runner and cached
  graph transforms. The architecture references are the
  [T-GCN](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/temporalgcn.py),
  [GConvGRU](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/gconv_gru.py),
  [A3TGCN](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/attentiontemporalgcn.py)
  and [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py)
  implementations. A prefix-invariance test checks that adding future annual
  inputs does not change earlier predictions and that gradients are finite.

## Pending and reproduction

MTGNN is pending the professor's choice of architecture. CPU reference outputs
are stored here. The user supplied return code 0 for the full GCN + LSTM CUDA command on
Colab. Its summary CSV is checked above. Individual GPU predictions are available in Drive. For the GPU audit
and remaining CUDA runs, run Phase5_Colab.ipynb in Colab. It reuses the existing Phase 4 and 5 inputs
and writes one summary to My Drive/phase5_scores.csv. A failed or interrupted family can be rerun.

Local command (replace FAMILY with tgcn, gconvgru, a3tgcn or graphwavenet):

```sh
.venv-expanded/bin/python src/models/phase5_gcn_lstm.py --model FAMILY --device cpu --out results/phase5/FAMILY
.venv-expanded/bin/python src/eval/report_phase5.py
```
""")


if __name__ == "__main__":
    main()
