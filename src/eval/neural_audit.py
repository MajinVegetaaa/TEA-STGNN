"""Protocol comparability checks for neural validation reports."""


def validate_s5_comparators(records: dict[str, dict], expected_seeds: list[int]) -> None:
    """Require S4 MTGNN and every proposed run to share data and tuning budget."""
    required = {"s4/mtgnn", "s5/proposed_v0", "s5/proposed_v1", "s5/proposed_v2", "s5/proposed_v3"}
    missing = required - records.keys()
    if missing:
        raise ValueError(f"S5 comparison missing protocol records: {sorted(missing)}")
    expected_models = {"s4/mtgnn": "mtgnn", **{f"s5/proposed_v{i}": "proposed" for i in range(4)}}
    reference = records["s4/mtgnn"]
    for run in sorted(required):
        record = records[run]
        cfg = record.get("config", {})
        if cfg.get("model") != expected_models[run]:
            raise ValueError(f"{run} protocol does not identify its expected model family")
        if record.get("snapshot_sha256") != reference.get("snapshot_sha256"):
            raise ValueError(f"{run} does not use the same snapshot as S4 MTGNN")
        if record.get("budget") != reference.get("budget") or record.get("grid_points") != reference.get("grid_points"):
            raise ValueError(f"{run} does not use the S4 shared training budget")
        if record.get("targets") != reference.get("targets"):
            raise ValueError(f"{run} does not score the same targets as S4 MTGNN")
        if record.get("test_run") is not False or cfg.get("label") != "change" or cfg.get("graph") != "v2":
            raise ValueError(f"{run} is not a 2023-only graph-v2 change-label run")
        if record.get("config", {}).get("validation_only") and cfg.get("through_year") != 2023:
            raise ValueError(f"{run} does not cap its panel at 2023")
    for run in sorted(required - {"s4/mtgnn"}):
        if not records[run]["config"].get("validation_only"):
            raise ValueError(f"{run} is missing the validation-only guard")
    if not expected_seeds:
        raise ValueError("expected validation seed list is empty")
