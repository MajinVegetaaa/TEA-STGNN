# Kaggle execution: S5 Round 2

This guide packages the already prepared Round 2 code for a Kaggle Tesla T4 session. It does not authorize training by itself. The corrected validation bundle is `data/graph_v2/pyg/s5_validation/`; its manifest fixes the data window at 2018–2023 (training 2018–2022, model selection 2023). The original monolithic snapshots and first-round result directories are not inputs to the Round 2 runner.

## Prepare a writable Kaggle copy

Attach the project dataset containing the repository folder `TEA-STGNN/` and a separate persistent dataset holding any prior Round 2 backup. Turn on the Tesla T4 accelerator and internet only if dependency installation is needed. Kaggle inputs under `/kaggle/input` are read-only; all experiment output must remain under `/kaggle/working`.

```bash
cd /kaggle/working
cp -a /kaggle/input/<project-dataset-slug>/TEA-STGNN ./TEA-STGNN
cd /kaggle/working/TEA-STGNN
```

Do not copy the project into `/kaggle/input`, and do not run any data, feature, graph, or snapshot rebuild command. The project dataset must include the source/configuration files and the full `data/graph_v2/pyg/s5_validation/` bundle (manifest, metadata, and six annual shards). The builder is not needed in the Kaggle run.

## Environment and data preflight

First check the installed runtime without changing packages:

```bash
python --version
python -c "import torch, torch_geometric, yaml; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available()); print('PyG', torch_geometric.__version__); print('PyYAML', yaml.__version__)"
python src/train/preflight_s5_round2.py --device cuda
```

Proceed only if the preflight reports PyTorch 2.12.1, PyG 2.8.1, `torch_scatter`, `torch_sparse`, PyG Temporal 0.56.2, an available CUDA device and fingerprint `09f87650e47552392326034004e15ff985c1bf16fa981dd8eb5dcb8440fcd4d9`. The preflight loads only 2018–2023 shards and does synthetic forward/backward checks; it does not make predictions or score the 2023 outcomes.

If and only if a required dependency/version is missing, install the pinned CUDA 13.0 build in the Kaggle environment, then restart the session and rerun preflight:

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt -r requirements-gpu.txt -f https://data.pyg.org/whl/torch-2.12.1+cu130.html
```

Do not force-reinstall a matching environment. If Kaggle's driver cannot support the CUDA 13.0 wheel, stop and use a compatible supported runtime documented in `docs/MODELS.md`; do not proceed with a failed preflight.

## Run order

Run one command at a time from the repository root. Each runner writes into its distinct `results/s5_round2/<run>/` folder, logs protocol and provenance, and refuses to reuse a non-empty output unless the matching incomplete run is explicitly resumed.

```bash
python src/train/run_s5_round2.py configs/experiments/s5_round2_mtgnn_comparator.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v0.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v1_original.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v1_query_attention.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v2_original.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v2_positive_gate.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v3_single_shared.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v3_multi_shared.yaml --device cuda
python src/train/run_s5_round2.py configs/experiments/s5_round2_v3_planned.yaml --device cuda
python src/eval/report_s5_round2.py
```

The first run is the corrected-snapshot MTGNN comparator. Do not use the historical S4 result with the earlier snapshot fingerprint as the Round 2 comparator. The comparison reporter requires all nine completed runs, the same snapshot fingerprint, all 10 targets and seeds 0–4, and validation year 2023 before writing the comparison tables.

Each single-target run (MTGNN, V0, both V1s, both V2s and V3 single-task) has 4 grid candidates per target selected at seed 0 plus 5 fixed-epoch seed fits per target: 90 fit calls per run. The two multi-target runs (shared-relation multi-task and planned V3) use the same 4-point grid and five scoring seeds but share one fit across all targets: 9 fit calls each. Thus the grid is shared as a hyperparameter policy, but total compute is not identical: seven single-target runs account for 630 fit calls and the two multi-task runs for 18, 648 nominal fit calls total.

**What the v3-named runs are.** `s5_round2_v3_single_shared` and `s5_round2_v3_multi_shared` build the V1 model (no gate) on the four shared relations only; the report labels them `V1_shared4_single` / `V1_shared4_multi` (relation ablations). The roadmap's v3 — V2 with its zero-initialised gate, one multi-task model, the shared four plus every target's academic relation — is `s5_round2_v3_planned` (report label `V3_planned`). Only the planned run needs training if the other eight are complete; the reporter refuses to compare until it exists. These counts are not wall-clock estimates or optimizer-step counts.

## Resume an interrupted run

After restoring its backup (below), rerun the exact same config with `--resume`:

```bash
python src/train/run_s5_round2.py configs/experiments/s5_round2_v1_query_attention.yaml --device cuda --resume
```

Replace the config with the interrupted run's config. The runner checks the saved config hash and only resumes a matching incomplete run. Do not use `--resume` for a completed run or to change the data, model, device or configuration. When no backup has been restored and the current `/kaggle/working` directory is intact, use the same command directly.

## Back up results between runs

Kaggle session storage is temporary. After each completed model (and before ending a session), create a zip in `/kaggle/working`. This archives only generated Round 2 results, not the repository or source data:

```bash
python - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
from shutil import make_archive
root = Path('/kaggle/working/TEA-STGNN')
results = root / 'results' / 's5_round2'
assert results.is_dir()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive = make_archive(f'/kaggle/working/s5_round2_backup_{stamp}', 'zip',
                       root_dir=root, base_dir='results/s5_round2')
print(archive)
PY
```

Use Kaggle **Save Version** to preserve the notebook output containing `s5_round2_backup.zip`; download it and upload it as a private Kaggle dataset for the next session, then attach that dataset as an input. Restore the archive only into a fresh project copy with no `results/s5_round2/` directory. This prevents extraction from replacing existing experiment files:

```bash
test ! -e /kaggle/working/TEA-STGNN/results/s5_round2
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
archive = Path('/kaggle/input/<round2-backup-dataset-slug>/<timestamped-backup-zip>')
destination = Path('/kaggle/working/TEA-STGNN')
with ZipFile(archive) as zf:
    names = zf.namelist()
    assert names and all(n.startswith('results/s5_round2/') for n in names)
    zf.extractall(destination)
print('Restored Round 2 outputs; use --resume only for the interrupted matching config.')
PY
```

Never put generated results into the uploaded project dataset or commit them to the source repository. Preserve the backup until the complete comparison report has been saved.

## Outputs and boundary

Each run directory contains its protocol and manifest, per-seed metrics/predictions, epoch log, event ledger, and seed checkpoints. The final reporter writes four CSVs in `results/s5_round2/`. The reporter is the first comparison/model-selection step; it reads only those produced validation results and does not load a model or create new predictions. No command in this guide evaluates 2024. S6 remains a separate, not-yet-authorized step.
