"""S6 data: build and load the 2018-2024 test bundle, and a content hash that ties it to the locked 2018-2023 data.

The test bundle is built by the same builder as the validation bundle with cutoff=2024; preprocessing is still fitted
on 2018-2022, so every 2018-2023 value equals the validation bundle's. `content_hash` hashes those values by campus id
(not by row position, which shifts if a campus first appears in 2024), so a bundle built on another machine (Kaggle)
can be checked against the hash locked from the validation bundle before any model is fitted.

    .venv-expanded/bin/python src/s6/s6_data.py build            # writes data/graph_v2/pyg/s6_test (refuses to overwrite)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/train"), str(ROOT / "src/graphs")]
from data_s5_round2 import ValidationPanel, _sha256  # noqa: E402

TEST_BUNDLE = ROOT / "data/graph_v2/pyg/s6_test"
TEST_YEAR = 2024
LOCKED_YEARS = list(range(2018, TEST_YEAR))          # 2018-2023: the part that must match the validation bundle


def build_test_bundle(output: Path = TEST_BUNDLE) -> dict:
    import build_s5_validation_snapshot as builder   # imported here: it requires torch_geometric
    return builder.build_bundle(output=output, cutoff=TEST_YEAR)


def load_test_bundle(path: Path = TEST_BUNDLE) -> ValidationPanel:
    """Load the 2018-2024 bundle with the same integrity checks as the validation loader."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    years = manifest.get("years")
    if (manifest.get("format_version") != 1 or manifest.get("graph") != "v2" or manifest.get("cutoff_year") != TEST_YEAR
            or manifest.get("test_years_included") is not True or years != list(range(2018, TEST_YEAR + 1))):
        raise ValueError("not an S6 2018-2024 test bundle")
    if _sha256(path / "metadata.json") != manifest.get("metadata_sha256"):
        raise ValueError("test bundle metadata fingerprint mismatch")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    scaler = metadata["scaler"]
    if scaler.get("fit_years") != [2018, 2019, 2020, 2021, 2022] or scaler.get("categories_fit_years") != scaler["fit_years"]:
        raise ValueError("test bundle preprocessing must be fitted on 2018-2022, as in validation")
    snapshots = []
    for year in years:
        name = f"year_{year}.pt"
        if _sha256(path / name) != manifest["shards"][name]:
            raise ValueError(f"test bundle shard fingerprint mismatch: {name}")
        payload = torch.load(path / name, map_location="cpu", weights_only=False)
        if payload.get("year") != year:
            raise ValueError(f"shard year mismatch: {name}")
        snapshots.append(payload["snapshot"])
    return ValidationPanel(years=years, campus_ids=metadata["campus_ids"], targets=metadata["targets"],
                           feature_names=metadata["feature_names"], snapshots=snapshots,
                           split={"train_years": [y for y in years if y < TEST_YEAR], "val_years": [TEST_YEAR],
                                  "seeds": metadata["split"]["seeds"]},
                           academic_relation=metadata["academic_relation"], scaler=scaler,
                           edge_distance_stats=metadata["edge_distance_stats"],
                           fingerprint=manifest["fingerprint"], path=path)


def content_hash(panel, years: list[int] = LOCKED_YEARS) -> str:
    """Hash of every value the models see for `years`, keyed by campus id and relation (row-order independent)."""
    digest = hashlib.sha256()
    digest.update(json.dumps([list(panel.feature_names), list(panel.targets)]).encode())
    ids = np.asarray(panel.campus_ids)
    for year in years:
        snap = panel.snapshots[panel.years.index(year)]
        campus = snap["campus"]
        present = campus.present.numpy()
        order = np.argsort(ids[present], kind="stable")
        digest.update(f"year {year}".encode())
        digest.update("\n".join(ids[present][order]).encode())
        for name in ("x", "y", "y_lag1", "eval_mask"):
            digest.update(np.ascontiguousarray(getattr(campus, name).numpy()[present][order]).tobytes())
        for edge_type in sorted(snap.edge_types):
            store = snap[edge_type]
            if store.edge_index.numel() == 0:
                continue
            src, dst = ids[store.edge_index[0].numpy()], ids[store.edge_index[1].numpy()]
            rows = np.column_stack([store.edge_weight.numpy(), store.edge_attr.numpy()]).astype(np.float32)
            key = np.lexsort((dst, src))
            digest.update(edge_type[1].encode())
            digest.update("\n".join(src[key] + ">" + dst[key]).encode())
            digest.update(np.ascontiguousarray(rows[key]).tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        result = build_test_bundle()
        print(f"created {TEST_BUNDLE}; fingerprint={result['fingerprint']}")
    else:
        raise SystemExit("usage: s6_data.py build")
