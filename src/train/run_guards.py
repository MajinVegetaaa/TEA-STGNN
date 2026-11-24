"""Safety guards shared by neural experiment entry points and tests."""


def validate_run_request(cfg: dict, test_requested: bool, val_seeds: bool = False) -> None:
    """Keep validation-only experiments from reaching the reserved test-year path."""
    # Treat every proposed-model run as S5 even if someone copies a config and
    # changes its output folder; the holdout guard must follow the model, not its name.
    is_s5 = cfg.get("model") == "proposed" or str(cfg.get("out", "")).startswith("s5/")
    if is_s5 and (cfg.get("validation_only") is not True or cfg.get("through_year") != 2023):
        raise ValueError("S5 configs must be validation-only and cap the panel at 2023")
    if is_s5 and (cfg.get("graph") != "v2" or cfg.get("label") != "change"):
        raise ValueError("S5 runs must use Graph V2 and the change-from-last-year target")
    if test_requested and (cfg.get("validation_only", False) or is_s5 or cfg.get("through_year") == 2023):
        raise ValueError("this experiment is validation-only; the 2024 test year is reserved for S6")
    if cfg.get("validation_only") is True and not val_seeds:
        raise ValueError("validation-only runs must include --val-seeds to produce the required five-seed protocol")


def require_fresh_output(path) -> None:
    """Refuse to overwrite artifacts from an earlier run."""
    from pathlib import Path

    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty experiment output: {path}")
