# data/graph_v2 — current graph (S1b)

Built by `make graph` (`TEA_GRAPH=v2 python3 src/graphs/build_graph_data.py`), not in git. Same nodes and edges as
`data/graph_expanded/` (v1, frozen); re-weighted as described in `docs/GRAPH_DATA_DOCUMENTATION.md` ("Graph v2").
`graph_v2_parameters.json` records the training-year kernel widths. The PyG snapshot (`pyg/snapshots.pt`, with
`edge_attr`) is built by `make snapshot` in the GPU environment.
