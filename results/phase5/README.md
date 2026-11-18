# Phase 5 results

**Your GPU result** is [GCN + LSTM on Colab](gcn_lstm/comparison.csv).
The table below summarizes all five local CPU benchmarks. For the other four
models, the optional [Colab notebook](Phase5_Colab.ipynb) runs the same comparisons on a GPU.

Each number is the average 2024 prediction error
for a target. Lower is better. Compare models across the same row. Each family
uses the graph selected by its 2023 validation error. Phase 3 is the earlier
benchmark. The asterisk marks the fixed-support GraphWaveNet variant.

| Target | Phase 3 | GCN + LSTM | T-GCN | GConvGRU | A3TGCN | GraphWaveNet* |
|---|---:|---:|---:|---:|---:|---:|
| attendance rate | 0.728 | 1.643 | 1.710 | 1.654 | 1.785 | 2.006 |
| dropout 9 12 | 0.449 | 0.612 | 0.626 | 0.647 | 0.640 | 0.586 |
| graduation 4yr | 2.215 | 2.416 | 2.385 | 2.778 | 2.647 | 2.639 |
| retention rate | 1.071 | 1.850 | 1.863 | 1.617 | 1.857 | 1.420 |
| sat act participation | 8.667 | 12.557 | 13.006 | 11.757 | 13.060 | 11.804 |
| act composite | 1.148 | 1.366 | 1.322 | 1.263 | 1.480 | 1.194 |
| sat act above criterion | 5.889 | 6.906 | 6.759 | 6.742 | 8.489 | 6.710 |
| risk index | 6.714 | 8.246 | 9.189 | 7.340 | 9.729 | 7.489 |
| sat total | 30.284 | 37.714 | 37.150 | 36.823 | 44.448 | 32.864 |
| tsi both | 14.333 | 16.215 | 16.292 | 15.159 | 16.297 | 15.323 |

## Your Colab run

The [Colab GCN + LSTM scores](gcn_lstm/comparison.csv) contain all 20 expected rows. Graph choices and campus counts match the CPU run throughout. The largest difference in reported MAE was 0.00126. Both GPU graph variants had higher error than the Phase 3 baseline for all ten targets. This check used the supplied summary CSV.

## Audit details

All five model families used the same expanded snapshot, ten targets, eligible
campuses and five test seeds. The independent checker reproduced per-seed and
mean MAE, RMSE, R², coverage and differences from the two benchmarks, including
the standard deviation of MAE across seeds. The table counts targets where the
mean 2024 MAE was lower. Units differ by target, so errors are reported
individually above. Full metrics are in each model folder under benchmark/comparison.csv.

| Model | Graph | Lower MAE than Phase 3 | Lower MAE than persistence |
|---|---|---:|---:|
| a3tgcn | Selected single relation | 0/10 | 0/10 |
| a3tgcn | Combined | 0/10 | 0/10 |
| gcn_lstm | Selected single relation | 0/10 | 0/10 |
| gcn_lstm | Combined | 0/10 | 0/10 |
| gconvgru | Selected single relation | 0/10 | 3/10 |
| gconvgru | Combined | 0/10 | 2/10 |
| graphwavenet | Selected single relation | 0/10 | 3/10 |
| graphwavenet | Combined | 0/10 | 3/10 |
| tgcn | Selected single relation | 0/10 | 0/10 |
| tgcn | Combined | 0/10 | 0/10 |

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
