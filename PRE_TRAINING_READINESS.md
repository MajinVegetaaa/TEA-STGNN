# S5 Pre-Training Readiness

**Status: NOT READY FOR GPU TRAINING in the current checkout.** The repository code, data paths, snapshot, S5 configs,
leakage guards, and future run order are prepared. The active `.venv-expanded` is CPU-only (`torch 2.12.1+cpu`,
`torch.version.cuda is None`, `torch.cuda.is_available() == False`), so a supported CUDA runtime is still required.
No model was trained, fitted, scored, or used to generate predictions in this audit.

## Checks completed

- Branch: `s5-proposed-stgnn`, based on committed audit state `7f72802` before this continuation. The required raw,
  processed, Graph V2, and PyG snapshot paths exist. The corrected snapshot fingerprint recorded by the previous
  preparation is `ed128378b06d1156a4435f0da24e50150678882ea174ff62720b3bcf73154bab`; this audit did not read or
  re-hash its contents. All 82 source-file paths listed in `configs/data_sources.yaml` exist. Existing S4 results
  and all data/snapshot files were left untouched.
- Environment: repository venv Python 3.12.10; PyTorch 2.12.1+cpu; PyG 2.8.1; `torch_scatter` 2.1.2+pt212cpu;
  `torch_sparse` 0.6.18+pt212cpu; PyG Temporal 0.56.2; PyYAML 6.0.3. These match the pinned CPU dependencies.
  GNU Make is not available on the current PowerShell PATH; the commands below invoke Python directly, so Make is
  not required for the planned execution.
- Static inspection confirms every S5 config selects Graph V2, the change target, `through_year: 2023`,
  `validation_only: true`, and the shared budget. v0–v2 are single-target runs; v3 is one multi-target run with
  equal standardized target losses and separate output heads.
- Training windows: the harness selects `val_index = len(train_years) = 5`. The training loss uses `y[1:5]`,
  corresponding to forecast years 2019–2022. Training model inputs end at year index 4. Validation forecasts the
  2023 index using that year's features, whose temporal inputs are built from prior years. The validation truth and
  persistence reference are read only at index 5. Target-label means and standard deviations are fit from those
  training labels only. 2023 controls grid/epoch selection; 2024 is not a training or validation index.
- Graph/input preprocessing inspection: `input_columns()` excludes catalogued same-year observations. D1 temporal
  features are histories; academic and demographic relations for year *t* use year *t−1*. Numeric imputation and
  scaling, one-hot category vocabularies, academic/demographic kernel widths, and edge-distance scaling use training
  years only. Later unseen categories map to all-zero one-hot columns. The fixed Directory/NCES metadata vintage
  remains the previously documented D1 caveat.
- The snapshot is a monolithic file containing all years, and `torch.load` deserializes it before the 2023 cutoff is
  applied. The loader then exposes only snapshots/target tensors through 2023, removes `test_years` from the returned
  panel split, and drops the full deserialized container before returning. The S5 code does not read the snapshot's
  all-year `year_stats` diagnostics. If “no 2024 access” is interpreted to prohibit even transient deserialization
  of the serialized test tensors, this existing snapshot format cannot satisfy that stronger interpretation without
  creating a test-free snapshot; this continuation did not and could not regenerate one under the instruction.
- Training/test guards now follow every `model: proposed` config regardless of output-folder name, require the
  validation-only 2023 Graph V2 change protocol and `--val-seeds`, and reject test mode for any 2023-capped run.
  Experiment and comparison-report writers refuse non-empty/existing output paths rather than overwrite artifacts.
  The Makefile S5 target now places the corrected-snapshot MTGNN run before v0–v3 and the report.
- S5 comparability remains guarded by matching snapshot fingerprint, targets, shared budget/grid, 2023 cutoff,
  device, PyTorch/PyG versions, and exact target/graph/seed row coverage. The old CPU S4 result is not accepted as
  the comparator; the separate corrected-snapshot run is required before reporting a comparison.

## Tests and checks

Passed: **41 focused synthetic/config tests** across `test_data_window`, `test_s5_config`, `test_feature_encoding`,
`test_graph_v2`, `test_neural_audit`, `test_proposed`, and `test_frozen_gate`. Python compilation of changed runtime
and test modules passed; `git diff --check` passed. These tests use synthetic fixtures or configuration files, not
the school panel or snapshot. The trainer-fitting test module was intentionally not run because it executes fits.

The full `make check` and real-data validators were not run: GNU Make is unavailable, and the broad validators/tests
read all-year school data or snapshots, which would violate this audit's no-2024-value-access restriction. Prior
documented independent checks remain the evidence for the current artifacts: processed validation was 69/70 solely
because the authentic frozen reference is absent; the 2019–2023 graph diagnostic found no stored/rebuilt topology
mismatch. No model or result validation was performed here.

## Historical verification and remaining graph note

The authentic `results/frozen/pre_expansion/` reference is unavailable and was not searched for again. Its
fail-closed integrity gate remains intact. The missing archive prevents historical byte-for-byte verification of
the pre-expansion files and blocks a claim that the complete historical master-data gate passed. It does not block
S5 on the already-built, fingerprinted Graph V2 snapshot: independent current-data checks and the saved snapshot
provenance are sufficient for this step. Do not rebuild the data or weaken the gate to make the historical check
green.

The previously diagnosed 2024 `excl_criterion` near-tie discrepancy is unchanged. S5's returned panel ends in 2023,
so it does not use that year or edge set. Preserve the current snapshot fingerprint. Carry the discrepancy into S6
for explicit review before the one permitted 2024 evaluation; do not regenerate the graph as part of S5.

## Outputs, checkpoints, and overwrite protection

Each model run writes under its configured `results/<out>/` directory: `config.yaml`, `protocol.json`,
`validation_grid.csv`, `selection_locked.json`, and `validation_seeds.csv`. The training code does not save model
checkpoints or a dedicated log file; progress is printed to the console. A run interrupted after creating its output
directory is not resumable in place. The new guard refuses a rerun into any non-empty directory; archive/review a
partial run and choose a new output path rather than deleting or overwriting it. The report writer similarly
refuses to replace an existing `summary.csv` or `per_target.csv`.

Current output locations for the future run are unused: `results/s4/mtgnn_corrected_snapshot/`,
`results/s5/proposed_v0/` through `results/s5/proposed_v3/`, and `results/s5/summary.csv` plus
`results/s5/per_target.csv`.

## Future training order — not executed

Run in the supported CUDA environment from the repository root. Confirm the GPU runtime first:

```bash
python -c "import torch, torch_geometric; assert torch.__version__.startswith('2.12.1'); assert torch_geometric.__version__ == '2.8.1'; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch_geometric.__version__)"
```

Then execute only after authorization, in this order:

```bash
python src/train/run.py configs/experiments/s4_mtgnn_comparable.yaml --val-seeds --device cuda
python src/train/run.py configs/experiments/s5_proposed_v0.yaml --val-seeds --device cuda
python src/train/run.py configs/experiments/s5_proposed_v1.yaml --val-seeds --device cuda
python src/train/run.py configs/experiments/s5_proposed_v2.yaml --val-seeds --device cuda
python src/train/run.py configs/experiments/s5_proposed_v3.yaml --val-seeds --device cuda
python src/eval/report_neural.py --step s5
```

The corrected MTGNN command writes to its separate output directory, preserving the old result. Each S5 model writes
to its own configured directory. The final report checks comparator metadata/rows before calculating the 2023
comparison; do not run it until all five runs completed.

The shared *per-candidate* grid and epoch cap do not equal total compute. MTGNN and v0–v2 perform 10 target-specific
groups × (4 seed-0 grid fits + 5 seed validation fits) = 90 fits each. v3 performs one shared multi-target group ×
(4 grid fits + 5 seed fits) = 9 fits, with all targets and a union of their relations in each fit. Model sizes and
per-fit costs also differ. Report the common hyperparameter budget accurately, not as identical total FLOPs/runtime.

For a new Colab/Lightning CUDA environment, the repository-documented install order is:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt -r requirements-gpu.txt -f https://data.pyg.org/whl/torch-2.12.1+cu130.html
```

Use the documented `cu126` wheel/index variant only if the GPU driver requires it. No environment was installed or
changed in this audit.

## Remaining blocker

The current environment is CPU-only, with no CUDA runtime/device. Switch to an available Colab/Lightning GPU
environment, verify the command above reports CUDA available with the pinned versions, then start with the corrected
MTGNN comparator. GNU Make is optional for this direct-command sequence. The absent frozen archive remains a
historical verification limitation, not an S5 blocker.
