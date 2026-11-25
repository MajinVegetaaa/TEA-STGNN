"""Build a separate Graph V2 snapshot bundle capped at the 2023 validation year.

This builder intentionally reads the sorted Graph V2 CSV sources and never opens
the monolithic all-years ``snapshots.pt``. It stops at the first row after the
cutoff, before converting that row or any later row into pandas/torch values.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

if os.environ.get("TEA_GRAPH", "v2") != "v2":
    raise RuntimeError("S5 validation snapshot building requires TEA_GRAPH=v2")
os.environ["TEA_GRAPH"] = "v2"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "graphs"))
import to_pyg as converter  # noqa: E402
from edge_features import distance_stats, edge_attributes  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

SOURCE = ROOT / "data" / "graph_v2"
OUTPUT = SOURCE / "pyg" / "s5_validation"
SPLIT_PATH = ROOT / "configs" / "split.yaml"
SPLIT = yaml.safe_load(SPLIT_PATH.read_text(encoding="utf-8"))
CUTOFF = max(SPLIT["train_years"] + SPLIT["val_years"])
ID_COLUMNS = {"campus_id", "src_campus_id", "dst_campus_id", "district_id", "esc_region"}
LAYERS = ["district", "geographic", "demographic", "region", "academic"]
COMMON_RELATIONS = ["district", "geographic", "demographic", "region"]


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sorted_year_csv(path: Path, cutoff: int = CUTOFF) -> tuple[pd.DataFrame, str]:
    """Read rows through ``cutoff`` from a CSV sorted ascending by ``year``.

    The year column is checked before retaining a row. At the first later-year
    record the reader stops; no later values are converted or incorporated.
    The Graph V2 node writer sorts by (year, campus_id), and edge generation
    appends yearly groups in ascending year order.
    """
    rows: list[list[str]] = []
    digest = hashlib.sha256()
    previous_year = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc
        if "year" not in header:
            raise ValueError(f"CSV has no year column: {path}")
        year_col = header.index("year")
        digest.update(json.dumps(header, ensure_ascii=False).encode("utf-8"))
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"malformed CSV row in {path} before/at cutoff")
            try:
                year = int(row[year_col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid year field in {path}") from exc
            if previous_year is not None and year < previous_year:
                raise ValueError(f"{path} is not sorted by year; refusing unsafe cutoff")
            previous_year = year
            if year > cutoff:
                break
            rows.append(row)
            digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    frame = pd.DataFrame(rows, columns=header)
    if "year" in frame:
        frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    for column in ID_COLUMNS & set(frame.columns):
        frame[column] = frame[column].astype(str)
    text_columns = ID_COLUMNS | {"edge_set", "edge_type", "method", "urbanicity"}
    for column in frame.columns:
        if column not in text_columns and column != "year":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, digest.hexdigest()


def _source_files() -> dict[str, Path]:
    return {"nodes": SOURCE / "nodes.csv", **{layer: SOURCE / f"edges_{layer}.csv" for layer in LAYERS}}


def build_bundle(source: Path = SOURCE, output: Path = OUTPUT, cutoff: int = CUTOFF) -> dict:
    """Build and atomically publish the bundle through ``cutoff``; refuse to overwrite.

    The default (2023) is the S5 validation bundle. S6 passes cutoff=2024; preprocessing (medians, scaling,
    category vocabularies, edge-distance statistics) is still fitted on the training years 2018-2022 only, so the
    2018-2023 part of the test bundle carries the same values as the validation bundle.
    """
    test = cutoff > CUTOFF
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing validation bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    input_paths = {"nodes": source / "nodes.csv", **{layer: source / f"edges_{layer}.csv" for layer in LAYERS}}
    missing = [str(p) for p in input_paths.values() if not p.is_file()]
    for extra in (source / "node_feature_catalog.csv", source / "academic_edge_sets.csv"):
        if not extra.is_file():
            missing.append(str(extra))
    if missing:
        raise FileNotFoundError("missing Graph V2 input files: " + ", ".join(missing))

    temporary = Path(tempfile.mkdtemp(prefix=".s5_validation_", dir=output.parent))
    try:
        nodes, node_digest = read_sorted_year_csv(input_paths["nodes"], cutoff)
        expected_years = list(range(min(SPLIT["train_years"]), cutoff + 1))
        years = sorted(nodes.year.unique().tolist())
        if years != expected_years:
            raise ValueError(f"permitted-year node rows do not cover the expected years: {years}")
        if nodes.empty or nodes.year.max() != cutoff:
            raise ValueError(f"node source did not yield the required cutoff year {cutoff}")

        catalog = pd.read_csv(source / "node_feature_catalog.csv")
        edge_sets = pd.read_csv(source / "academic_edge_sets.csv")
        if not {"target", "edge_set"}.issubset(edge_sets.columns):
            raise ValueError("academic_edge_sets.csv is missing target/edge_set columns")
        target_to_set = dict(zip(edge_sets.target.astype(str), edge_sets.edge_set.astype(str)))
        expected_targets = list(converter.TARGETS)

        edges: dict[str, pd.DataFrame] = {}
        source_hashes = {"nodes.csv:through_cutoff": node_digest}
        for name in ("node_feature_catalog.csv", "academic_edge_sets.csv", "graph_v2_parameters.json"):
            source_hashes[name] = _file_hash(source / name)
        source_hashes["configs/split.yaml"] = _file_hash(SPLIT_PATH)
        for layer in LAYERS:
            frame, digest = read_sorted_year_csv(input_paths[layer], cutoff)
            if not frame.empty and int(frame.year.max()) > cutoff:
                raise AssertionError("future-year edge rows passed the cutoff reader")
            edges[layer] = frame
            source_hashes[f"edges_{layer}.csv:through_cutoff"] = digest

        feature_frame, scaler = converter.build_features(nodes, converter.input_columns(catalog))
        feature_names = feature_frame.columns.tolist()
        campus_ids = sorted(nodes.campus_id.astype(str).unique().tolist())
        campus_pos = {campus_id: i for i, campus_id in enumerate(campus_ids)}
        n = len(campus_ids)
        if n == 0:
            raise ValueError("no campuses in permitted-year nodes")

        distance_stats_by_relation = {}
        for layer, frame in edges.items():
            groups = frame.groupby("edge_set", sort=True) if layer == "academic" and not frame.empty else (
                [(None, frame)] if layer != "academic" else []
            )
            for edge_set, group in groups:
                relation = f"academic_{str(edge_set).removeprefix('excl_')}" if edge_set else layer
                distance_stats_by_relation[relation] = distance_stats(group, SPLIT["train_years"])

        # The graph-builder map is intentionally partial: derived targets such
        # as risk_index have no target-specific academic edge set. Include all
        # graph relations from the edge source, while metadata maps only the
        # targets that actually have a designated relation.
        academic_relations = sorted({
            f"academic_{str(es).removeprefix('excl_')}"
            for es in edges["academic"].edge_set.dropna().astype(str).unique()
        })
        relations = COMMON_RELATIONS + academic_relations
        snapshots = []
        for year in years:
            year_rows = nodes.year == year
            year_nodes = nodes.loc[year_rows]
            indices = torch.tensor(year_nodes.campus_id.astype(str).map(campus_pos).to_numpy(), dtype=torch.long)
            data = HeteroData()
            x = torch.zeros((n, len(feature_names)), dtype=torch.float32)
            x[indices] = torch.as_tensor(feature_frame.loc[year_rows].to_numpy(dtype=np.float32))
            present = torch.zeros(n, dtype=torch.bool)
            present[indices] = True
            y = torch.full((n, len(expected_targets)), float("nan"), dtype=torch.float32)
            y[indices] = torch.as_tensor(year_nodes[expected_targets].to_numpy(dtype=np.float32))
            lag_columns = [f"{target}_lag1" for target in expected_targets]
            missing_lags = set(lag_columns) - set(nodes.columns)
            if missing_lags:
                raise ValueError(f"node source lacks lag columns: {sorted(missing_lags)}")
            y_lag1 = torch.full_like(y, float("nan"))
            y_lag1[indices] = torch.as_tensor(year_nodes[lag_columns].to_numpy(dtype=np.float32))
            data["campus"].x = x
            data["campus"].present = present
            data["campus"].y = y
            data["campus"].y_lag1 = y_lag1
            data["campus"].y_mask = torch.isfinite(y) & present[:, None]
            data["campus"].eval_mask = data["campus"].y_mask & torch.isfinite(y_lag1)
            data.year = int(year)
            for layer, frame in edges.items():
                year_edges = frame.loc[frame.year == year]
                grouped = year_edges.groupby("edge_set", sort=True) if layer == "academic" else [(None, year_edges)]
                for edge_set, group in grouped:
                    relation = f"academic_{str(edge_set).removeprefix('excl_')}" if edge_set else layer
                    src = group.src_campus_id.astype(str).map(campus_pos)
                    dst = group.dst_campus_id.astype(str).map(campus_pos)
                    if src.isna().any() or dst.isna().any():
                        raise ValueError(f"{year}/{relation}: graph edge endpoint absent from permitted-year campuses")
                    s = torch.as_tensor(src.to_numpy(dtype=np.int64))
                    t = torch.as_tensor(dst.to_numpy(dtype=np.int64))
                    weight = torch.as_tensor(group.weight.to_numpy(dtype=np.float32))
                    store = data["campus", relation, "campus"]
                    store.edge_index = torch.stack([torch.cat([s, t]), torch.cat([t, s])])
                    store.edge_weight = torch.cat([weight, weight])
                    stats = distance_stats_by_relation[relation]
                    attr = torch.as_tensor(edge_attributes(group, stats), dtype=torch.float32)
                    store.edge_attr = torch.cat([attr, attr])
            for relation in relations:
                edge_type = ("campus", relation, "campus")
                if edge_type not in data.edge_types:
                    store = data[edge_type]
                    store.edge_index = torch.zeros((2, 0), dtype=torch.long)
                    store.edge_weight = torch.zeros(0, dtype=torch.float32)
                    store.edge_attr = torch.zeros((0, 2), dtype=torch.float32)
            snapshots.append(data)

        metadata = {
            "format_version": 1,
            "graph": "v2",
            "years": years,
            "campus_ids": campus_ids,
            "feature_names": feature_names,
            "targets": expected_targets,
            "academic_relation": {t: f"academic_{target_to_set[t].removeprefix('excl_')}"
                                  for t in expected_targets if t in target_to_set},
            "split": {"train_years": SPLIT["train_years"], "val_years": SPLIT["val_years"],
                      "seeds": SPLIT["seeds"], **({"test_years": SPLIT["test_years"]} if test else {})},
            "scaler": scaler,
            "edge_attr": ["weight", "distance_z"],
            "edge_distance_stats": {k: list(v) for k, v in distance_stats_by_relation.items()},
            "source_hashes_through_cutoff": source_hashes,
        }
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shard_hashes = {}
        for snap in snapshots:
            year = int(snap.year)
            shard_path = temporary / f"year_{year}.pt"
            torch.save({"year": year, "snapshot": snap}, shard_path)
            shard_hashes[shard_path.name] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        payload = {"metadata_sha256": metadata_hash, "shards": shard_hashes}
        manifest = {
            "format_version": 1,
            "graph": "v2",
            "cutoff_year": cutoff,
            "years": years,
            "train_years": SPLIT["train_years"],
            "validation_years": SPLIT["val_years"],
            "test_years_included": test,
            "metadata_sha256": metadata_hash,
            "shards": shard_hashes,
            "fingerprint": _canonical_hash(payload),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build_bundle(output=args.output)
    print(f"created {args.output} through {CUTOFF}; fingerprint={result['fingerprint']}")


if __name__ == "__main__":
    main()
