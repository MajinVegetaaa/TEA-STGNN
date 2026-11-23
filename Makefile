# One entry point for the pipeline. Run from the repo root. See ROADMAP.md.
# Graphs: data/graph_v2/ is current (S1b). data/graph_expanded/ (v1, the Phase 5 graph) and data/graph/ (original) are
# frozen and never rebuilt here; `check` still validates v1.
PY ?= python3
NN ?= .venv-expanded/bin/python   # environment with torch + PyG (requirements-gpu.txt) for snapshots and neural models
TESTS = tests.test_new_sources tests.test_expanded_features tests.test_finance tests.test_phase5_models tests.test_protocol tests.test_graph_v2 tests.test_d1_features tests.test_harness tests.test_mtgnn tests.test_proposed

.PHONY: help check data features graph snapshot test protocol s1c s2 s5

help:            ## list targets
	@grep -E '^[a-z0-9]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

check:           ## validate the current files and run tests (no rebuild)
	$(PY) src/data/validate_processed.py
	TEA_GRAPH=v2 $(PY) src/graphs/validate_graph.py
	TEA_GRAPH=expanded $(PY) src/graphs/validate_graph.py
	$(PY) -m unittest $(TESTS)
	$(PY) src/eval/check_protocol.py

data:            ## audit raw files, rebuild the master table, validate it (~12 min)
	$(PY) src/data/audit_raw.py
	$(PY) src/data/build_master.py
	$(PY) src/data/validate_processed.py

features:        ## rebuild D1 history features for graph v2 (needs a passing `make data`)
	$(PY) src/data/prepare_d1_features.py

graph:           ## rebuild and validate graph v2 (data/graph_v2/) and compare it with v1
	TEA_GRAPH=v2 $(PY) src/graphs/build_graph_data.py
	TEA_GRAPH=v2 $(PY) src/graphs/validate_graph.py
	$(PY) src/graphs/compare_v1_v2.py

snapshot:        ## graph v2 -> PyG snapshot with edge features (runs with $(NN))
	TEA_GRAPH=v2 $(NN) src/graphs/to_pyg.py

test:            ## unit tests
	$(PY) -m unittest $(TESTS)

protocol:        ## check the locked protocol reproduces the stored baseline numbers
	$(PY) src/eval/check_protocol.py

s1c:             ## numbers to beat under D1 on graph v2 (~2.5 h on the laptop: 60 runs, 3 in parallel)
	mkdir -p results/s1c/logs
	for mode in level relative zscore; do (for set in existing all; do \
	  $(PY) src/baselines/run_s1c.py --input-set $$set --mode $$mode > results/s1c/logs/$${set}_$${mode}.log 2>&1; done) & done; wait
	$(PY) src/eval/numbers_to_beat.py

s2:              ## harness reproduces the Phase 5 GCN + LSTM benchmark (~1 min on the laptop CPU)
	$(NN) src/train/run.py configs/experiments/s2_gcn_lstm_phase5.yaml --test
	$(NN) src/train/check_reproduction.py

s5:              ## proposed STGNN v0-v3: shared budget B, 2023 selection and 5 validation seeds only
	$(NN) src/train/run.py configs/experiments/s5_proposed_v0.yaml --val-seeds
	$(NN) src/train/run.py configs/experiments/s5_proposed_v1.yaml --val-seeds
	$(NN) src/train/run.py configs/experiments/s5_proposed_v2.yaml --val-seeds
	$(NN) src/train/run.py configs/experiments/s5_proposed_v3.yaml --val-seeds
	$(NN) src/eval/report_neural.py --step s5
