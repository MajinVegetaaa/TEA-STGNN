"""Safety guards shared by neural experiment entry points and tests."""


def validate_run_request(cfg: dict, test_requested: bool) -> None:
    """Keep validation-only experiments from reaching the reserved test-year path."""
    is_s5 = cfg.get("model") == "proposed" and str(cfg.get("out", "")).startswith("s5/")
    if is_s5 and (cfg.get("validation_only") is not True or cfg.get("through_year") != 2023):
        raise ValueError("S5 configs must be validation-only and cap the panel at 2023")
    if test_requested and (cfg.get("validation_only", False) or is_s5):
        raise ValueError("this experiment is validation-only; the 2024 test year is reserved for S6")
