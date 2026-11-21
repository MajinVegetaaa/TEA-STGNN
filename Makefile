# One entry point for the pipeline. Run from the repo root. See ROADMAP.md.
# Graphs: data/graph_v2/ is current (S1b). data/graph_expanded/ (v1, the Phase 5 graph) and data/graph/ (original) are
# frozen and never rebuilt here; `check` still validates v1.
PY ?= python3
TESTS = tests.test_new_sources tests.test_expanded_features tests.test_finance tests.test_phase5_models tests.test_protocol tests.test_graph_v2

.PHONY: help check data features graph snapshot test protocol

help:            ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

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

features:        ## rebuild publication-filtered features (needs a passing `make data`)
	$(PY) src/data/prepare_expanded_features.py

graph:           ## rebuild and validate graph v2 (data/graph_v2/) and compare it with v1
	TEA_GRAPH=v2 $(PY) src/graphs/build_graph_data.py
	TEA_GRAPH=v2 $(PY) src/graphs/validate_graph.py
	$(PY) src/graphs/compare_v1_v2.py

snapshot:        ## graph v2 -> PyG snapshot with edge features (needs requirements-gpu.txt; run in the GPU environment)
	TEA_GRAPH=v2 $(PY) src/graphs/to_pyg.py

test:            ## unit tests
	$(PY) -m unittest $(TESTS)

protocol:        ## check the locked protocol reproduces the stored baseline numbers
	$(PY) src/eval/check_protocol.py
