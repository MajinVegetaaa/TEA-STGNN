# Phase 5 milestone 1 — GCN + LSTM

The model uses the 430 expanded node inputs and fixed campus order. It performs
one symmetric degree-normalized graph convolution with self-loops, followed by
a 32-unit LSTM over calendar years and a linear prediction head. The combined
graph sums parallel relation weights. Each target uses its eligible academic
relation. Risk Index uses the four eligible non-academic relations.

Implementation choices for this first run include hidden width 32, Adam learning
rate 0.003, and 30 candidate epochs. Only graph and epoch are selected using
2023 validation. The model trains on observed labels in 2019–2022 using 2018 as
initial history. Target standardization uses the fitting window. Validation
and test scoring require both the observed target and the observed preceding
year. Final models refit through 2023, with graph and epoch locked, across seeds
0–4. Node input preprocessing comes from the frozen expanded snapshot.

## Colab T4

Upload `phase5_colab.zip` to My Drive alongside the existing
`tea_phase4_colab.zip`. Choose a GPU runtime and run this cell. Outputs persist
under My Drive. The CPU benchmark uses the same runner with `--device cpu`.

```python
from google.colab import drive
from pathlib import Path
from zipfile import ZipFile
import subprocess, sys, torch

drive.mount("/content/drive")
drive_root = Path("/content/drive/MyDrive")
work = Path("/content/tea_phase5")
assert torch.cuda.is_available(), "Choose a GPU runtime"
for archive in ["tea_phase4_colab.zip", "phase5_colab.zip"]:
    with ZipFile(drive_root / archive) as z:
        z.extractall(work)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(work / "requirements-colab.txt")], check=True)
out = drive_root / "tea_phase5_gcn_lstm"
subprocess.run([sys.executable, str(work / "check_phase4.py"), "--verify-only",
                "--out", str(out)], check=True)
cmd = [sys.executable, "-u", str(work / "src/models/phase5_gcn_lstm.py"),
       "--device", "cuda", "--out", str(out)]
subprocess.run(cmd + ["--smoke"], check=True)
subprocess.run(cmd, check=True)
```

The two-epoch smoke exercises all eligible graduation graph variants. The full
run saves the validation grid, selection lock, five checkpoints per target/graph
choice, campus predictions, per-seed metrics and comparison.csv. Each target
has two rows in the comparison: combined and best single relation. Negative MAE
differences favor GCN + LSTM. Both are prespecified comparisons.

## Changes and outstanding work

The first benchmark runs on the local CPU as a reference. Colab execution uses
the user's GPU runtime. Phase 4's T4 verification is complete. The full Phase 5 GCN + LSTM CUDA command returned code 0 in Colab, as
reported by the user. The user supplied [the GPU comparison table](comparison.csv). All 20 rows
match the CPU graph choices and campus counts. The maximum absolute MAE
difference between the two runs was 0.00127 or less. This milestone covers GCN + LSTM.
T-GCN, GConvGRU, A3TGCN and GraphWaveNet follow. MTGNN's intended implementation
still needs clarification. Phase 3's 2024 results were already known at kickoff.

## Completed local CPU reference

The run produced 20 comparisons and 100 checkpoints. Each test prediction was checked against the Phase 3 campus IDs and outcomes.

- Validation-selected single relation had lower mean test MAE than the locked Phase 3 model on 0/10 targets and lower MAE than persistence on 0/10.
- Combined graph had lower mean test MAE than the locked Phase 3 model on 0/10 targets and lower MAE than persistence on 0/10.

[Full per-target results](benchmark/comparison.csv)
