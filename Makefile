# One entry point for the pipeline. Run from the repo root. See ROADMAP.md.
# The frozen original graph (data/graph/) is never rebuilt here: everything graph-related uses data/graph_expanded/.
PY ?= python3
TESTS = tests.test_new_sources tests.test_expanded_features tests.test_finance tests.test_phase5_models tests.test_protocol

.PHONY: help check data features graph test protocol

help:            ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

check:           ## validate the current files and run tests (no rebuild)
	$(PY) src/data/validate_processed.py
	TEA_EXPANDED=1 $(PY) src/graphs/validate_graph.py
	$(PY) -m unittest $(TESTS)
	$(PY) src/eval/check_protocol.py

data:            ## audit raw files, rebuild the master table, validate it (~12 min)
	$(PY) src/data/audit_raw.py
	$(PY) src/data/build_master.py
	$(PY) src/data/validate_processed.py

features:        ## rebuild publication-filtered features (needs a passing `make data`)
	$(PY) src/data/prepare_expanded_features.py

graph:           ## rebuild and validate the expanded graph (data/graph_expanded/)
	TEA_EXPANDED=1 $(PY) src/graphs/build_graph_data.py
	TEA_EXPANDED=1 $(PY) src/graphs/validate_graph.py

test:            ## unit tests
	$(PY) -m unittest $(TESTS)

protocol:        ## check the locked protocol reproduces the stored baseline numbers
	$(PY) src/eval/check_protocol.py
