"""Loader for the year-sharded, 2023-capped S5 Round 2 Graph V2 bundle."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "data/graph_v2/pyg/s5_validation"
CUTOFF = 2023
BASE_RELATIONS = ["district", "geographic", "demographic", "region"]


@dataclass
class ValidationPanel:
    years: list[int]
    campus_ids: list[str]
    targets: list[str]
    feature_names: list[str]
    snapshots: list
    split: dict
    academic_relation: dict[str, str]
    scaler: dict
    edge_distance_stats: dict
    fingerprint: str
    path: Path

    @property
    def val_index(self) -> int:
        return len(self.split["train_years"])

    def relations(self, target: str) -> list[str]:
        return BASE_RELATIONS + ([self.academic_relation[target]] if target in self.academic_relation else [])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_bundle(path: Path = DEFAULT_BUNDLE) -> ValidationPanel:
    """Load only manifest-listed annual shards and reject any post-2023 year."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    metadata_path = path / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing validation bundle manifest/metadata under {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    years = manifest.get("years")
    if (manifest.get("format_version") != 1 or manifest.get("graph") != "v2"
            or manifest.get("cutoff_year") != CUTOFF or manifest.get("test_years_included") is not False):
        raise ValueError("unsupported or non-validation-only S5 Round 2 manifest")
    if not isinstance(years, list) or not years or any(not isinstance(y, int) or y > CUTOFF for y in years):
        raise ValueError("validation bundle contains an invalid or post-cutoff year")
    if years != list(range(2018, CUTOFF + 1)):
        raise ValueError("validation bundle must contain each year from 2018 through 2023")
    shard_map = manifest.get("shards")
    expected_names = {f"year_{year}.pt" for year in years}
    if not isinstance(shard_map, dict) or set(shard_map) != expected_names:
        raise ValueError("manifest shard list must exactly match the permitted years")
    if _sha256(metadata_path) != manifest.get("metadata_sha256"):
        raise ValueError("validation bundle metadata fingerprint mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("years") != years or metadata.get("graph") != "v2":
        raise ValueError("validation metadata does not match the manifest")
    if metadata.get("scaler", {}).get("fit_years") != manifest.get("train_years"):
        raise ValueError("feature scaler was not fit on the declared training years")
    if metadata.get("scaler", {}).get("categories_fit_years") != manifest.get("train_years"):
        raise ValueError("category vocabulary was not fit on the declared training years")
    if metadata.get("scaler", {}).get("unknown_category") != "all_zero":
        raise ValueError("unexpected handling for unseen validation categories")
    if metadata.get("edge_attr") != ["weight", "distance_z"]:
        raise ValueError("unexpected Graph V2 edge attribute schema")

    snapshots = []
    for year in years:
        name = f"year_{year}.pt"
        shard_path = path / name
        if _sha256(shard_path) != shard_map[name]:
            raise ValueError(f"snapshot shard fingerprint mismatch: {name}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        if payload.get("year") != year or getattr(payload.get("snapshot"), "year", None) != year:
            raise ValueError(f"shard year does not match its manifest entry: {name}")
        snapshots.append(payload["snapshot"])
        del payload

    identity = {"metadata_sha256": manifest["metadata_sha256"], "shards": shard_map}
    expected_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected_fingerprint != manifest.get("fingerprint"):
        raise ValueError("validation bundle aggregate fingerprint mismatch")
    return ValidationPanel(years=years, campus_ids=metadata["campus_ids"], targets=metadata["targets"],
                           feature_names=metadata["feature_names"], snapshots=snapshots,
                           split={"train_years": manifest["train_years"],
                                  "val_years": manifest["validation_years"],
                                  "seeds": metadata["split"]["seeds"]},
                           academic_relation=metadata["academic_relation"], scaler=metadata["scaler"],
                           edge_distance_stats=metadata["edge_distance_stats"],
                           fingerprint=manifest["fingerprint"], path=path)
