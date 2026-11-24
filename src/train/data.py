"""Load a PyG snapshot file into one panel of tensors that every neural model uses.

Years run along the first axis, schools along the second (the same school order in every year), targets along the last.
Year index 0 (2018) has no history, so it is never a training label.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = {"v1": ROOT / "data/graph_expanded/pyg/snapshots.pt",     # Phase 5 graph (frozen)
             "v2": ROOT / "data/graph_v2/pyg/snapshots.pt"}           # current graph (S1b)
BASE_RELATIONS = ["district", "geographic", "demographic", "region"]   # + the target's own academic relation


@dataclass
class Panel:
    years: list[int]
    campus_ids: list[str]
    targets: list[str]
    snapshots: list            # HeteroData per year (graph structure, edge weights / attributes)
    y: torch.Tensor            # [years, schools, targets] raw outcome (NaN = missing)
    y_lag1: torch.Tensor       # [years, schools, targets] last year's value
    eval_mask: torch.Tensor    # [years, schools, targets] target and last year's value observed (scored rows)
    present: torch.Tensor      # [years, schools]
    split: dict
    academic_relation: dict    # target -> its academic edge set
    sha256: str
    path: Path

    @property
    def val_index(self) -> int:
        """Index of the validation year (2023); training years are 1..val_index-1."""
        return len(self.split["train_years"])

    def relations(self, target: str) -> list[str]:
        """Relations a single-target model may use: the shared ones plus the target's own academic set."""
        rel = BASE_RELATIONS.copy()
        if target in self.academic_relation:
            rel.append(self.academic_relation[target])
        return rel


def load(graph: str, through_year: int | None = None,
         require_train_only_categories: bool = False) -> Panel:
    path = SNAPSHOTS[graph]
    data = torch.load(path, weights_only=False, map_location="cpu")
    split = data["split"]
    assert data["years"] == split["train_years"] + split["val_years"] + split["test_years"], "years out of order"
    if require_train_only_categories:
        scaler = data.get("scaler", {})
        if (scaler.get("categories_fit_years") != split["train_years"]
                or scaler.get("fit_years") != split["train_years"]
                or not scaler.get("category_vocabulary")
                or scaler.get("unknown_category") != "all_zero"
                or data.get("edge_attr") != ["weight", "distance_z"]):
            raise ValueError("S5 requires a snapshot whose categorical vocabulary was fit on training years only; rebuild it")
    years = data["years"]
    snaps = data["snapshots"]
    panel_split = dict(split)
    if through_year is not None:
        validation_cutoff = max(split["train_years"] + split["val_years"])
        if through_year > validation_cutoff:
            raise ValueError(f"through_year={through_year} exceeds the validation cutoff {validation_cutoff}")
        keep = [i for i, year in enumerate(years) if year <= through_year]
        if not keep or years[keep[-1]] != through_year:
            raise ValueError(f"snapshot does not contain requested through_year={through_year}")
        years = [years[i] for i in keep]
        snaps = [snaps[i] for i in keep]
        # A validation panel must not carry a test split year that downstream
        # code could use to infer an index, even though the source snapshot does.
        panel_split["test_years"] = []
    stack = lambda key: torch.stack([s["campus"][key] for s in snaps])
    campus_ids, targets = data["campus_ids"], data["targets"]
    academic_relation = data["academic_relation"]
    # Drop the deserialized full-snapshot container before exposing the panel;
    # for a 2023 cutoff, only the selected snapshots remain referenced below.
    del data
    return Panel(years=years, campus_ids=campus_ids, targets=targets, snapshots=snaps,
                 y=stack("y"), y_lag1=stack("y_lag1"), eval_mask=stack("eval_mask"), present=stack("present"),
                 split=panel_split, academic_relation=academic_relation,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), path=path)
