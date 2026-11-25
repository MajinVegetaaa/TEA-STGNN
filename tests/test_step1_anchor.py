"""Synthetic checks for Step 1 (shrinkage anchor for the STGNN); no school data is read."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/models"), str(ROOT / "src/train")]
import run_s5_round2 as runner  # noqa: E402
from proposed import ProposedWindow  # noqa: E402

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def panel_with_history(n=60, beta=0.5, seed=0):
    """y = lag1 + beta*(ma3 - lag1) + noise from 2020 on; ma3 flagged missing for 2018/2019."""
    g = torch.Generator().manual_seed(seed)
    level = torch.randn(n, generator=g) * 3 + 50
    hist = [level + torch.randn(n, generator=g) for _ in range(3)]          # 2015-2017
    snaps, ys, lags = [], [], []
    for i, year in enumerate(YEARS):
        lag1 = hist[-1]
        ma3 = torch.stack(hist[-3:]).mean(0)
        y = lag1 + beta * (ma3 - lag1) + torch.randn(n, generator=g) * 0.1
        missing = float(year < 2020)
        x = torch.stack([torch.randn(n, generator=g), torch.zeros(n),
                         (ma3 - 50) / 2 if not missing else torch.zeros(n), torch.full((n,), missing)], dim=1)
        snaps.append({"campus": SimpleNamespace(x=x, y=y.view(n, 1).clone(), y_lag1=lag1.view(n, 1).clone())})
        ys.append(y.view(n, 1))
        lags.append(lag1.view(n, 1))
        hist.append(y)
    panel = SimpleNamespace(years=YEARS, snapshots=snaps, targets=["t"],
                            feature_names=["enrollment_lag1", "enrollment_lag1_missing", "t_ma3", "t_ma3_missing"],
                            split={"train_years": YEARS[:-1]})
    scaler = {"sd": {"t_ma3": 2.0}, "mean": {"t_ma3": 50.0}}
    return panel, torch.stack(ys), torch.stack(lags), scaler


class Step1AnchorTests(unittest.TestCase):
    def test_recovers_beta_and_falls_back_to_lag_where_ma3_missing(self):
        panel, y, lag, scaler = panel_with_history(n=600, beta=0.5)
        anchor, coef = runner.shrinkage_anchor(panel, y, lag, scaler)
        for c in coef:
            self.assertAlmostEqual(c["beta"], 0.5, delta=0.05)
        torch.testing.assert_close(anchor[:2], lag[:2])                 # 2018/2019: ma3 missing
        self.assertLess(float((anchor[5] - y[5]).abs().mean()), float((lag[5] - y[5]).abs().mean()))

    def test_validation_year_outcomes_do_not_change_the_anchor(self):
        panel, y, lag, scaler = panel_with_history()
        before, coef = runner.shrinkage_anchor(panel, y, lag, scaler)
        y2 = y.clone()
        y2[5] += 100.0                                                   # 2023 outcomes
        after, coef2 = runner.shrinkage_anchor(panel, y2, lag, scaler)
        torch.testing.assert_close(after, before)
        self.assertEqual(coef, coef2)

    def test_fit_trial_anchor_equal_to_lag_matches_default_and_anchor_moves_forecast(self):
        t, n = 5, 20
        g = torch.Generator().manual_seed(3)
        lag = torch.randn(t, n, 1, generator=g) + 50
        y = lag + torch.randn(t, n, 1, generator=g)
        x = torch.randn(t, n, 3, generator=g)
        panel = SimpleNamespace(val_index=4, campus_ids=[str(i) for i in range(n)])
        runner.stop_year, runner.campus_ids = 2023, panel.campus_ids
        common = dict(variant="v0_temporal", x=ProposedWindow(x, [[] for _ in range(t)], [[] for _ in range(t)]),
                      y=y, lag=lag, eval_mask=torch.isfinite(y), target_indices=[0], target_names=["t"],
                      panel=panel, relations=[], hidden=4, dropout=0.0, epochs=3, lr=0.01, weight_decay=0.0,
                      patience=None, seed=0, device="cpu", tune=False, gradient_clip_norm=None, config_hash="h")
        default = runner.fit_trial(**common)
        same = runner.fit_trial(**common, anchor=lag.clone())
        torch.testing.assert_close(default["forecast"], same["forecast"], rtol=0, atol=0)
        moved = runner.fit_trial(**common, anchor=lag + 5.0)              # constant shift of the anchor
        # labels shift by -5 and the centre absorbs it, so the forecast is unchanged up to float error
        torch.testing.assert_close(moved["forecast"], default["forecast"], rtol=0, atol=1e-3)
        row = moved["prediction_rows"][0]
        self.assertAlmostEqual(row["anchor_pred"] - row["persistence_pred"], 5.0, places=4)
        varied = runner.fit_trial(**common, anchor=lag + torch.randn(t, n, 1, generator=g))
        self.assertFalse(torch.allclose(varied["forecast"], default["forecast"]))

    def test_config_guards(self):
        base = {"variant": "v1_original", "through_year": 2023, "validation_only": True, "graph": "v2",
                "label": "change", "out": "s5_round2/step1/x", "tasks": "single", "relation_policy": "target_specific"}
        runner.validate_config(dict(base, anchor="shrink_own_3yr"))
        for bad in (dict(base, anchor="district"), dict(base, anchor="shrink_own_3yr", target_mode="change_relative"),
                    dict(base, anchor="shrink_own_3yr", variant="mtgnn", relation_policy="shared_four")):
            with self.assertRaises(ValueError):
                runner.validate_config(bad)


if __name__ == "__main__":
    unittest.main()
