"""Synthetic checks for Step 2 (relative-change target, dist_2019 input); no school data is read."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/models"), str(ROOT / "src/train")]
import run_s5_round2 as runner  # noqa: E402
from proposed import ProposedWindow  # noqa: E402


def synthetic(t=5, n=30, shock=(0.0, 3.0, -4.0, 2.0, 1.0), seed=0):
    g = torch.Generator().manual_seed(seed)
    lag = torch.randn(t, n, 1, generator=g) + 50
    shocks = torch.tensor(shock[:t]).view(t, 1, 1)
    y = lag + shocks + torch.randn(t, n, 1, generator=g) * 0.5
    y[1, 0, 0] = float("nan")                                   # a missing label
    return y, lag


class Step2TargetTests(unittest.TestCase):
    def test_change_mode_is_the_old_label(self):
        y, lag = synthetic()
        torch.testing.assert_close(runner.training_labels(y, lag, 4, "change"), y[1:4] - lag[1:4],
                                   rtol=0, atol=0, equal_nan=True)

    def test_relative_mode_removes_each_years_statewide_shift(self):
        y, lag = synthetic()
        rel = runner.training_labels(y, lag, 4, "change_relative")
        self.assertTrue(torch.isnan(rel[0, 0, 0]))                  # missing stays missing
        for year in range(rel.shape[0]):
            v = rel[year, :, 0]
            self.assertAlmostEqual(float(v[torch.isfinite(v)].mean()), 0.0, places=5)
        with self.assertRaises(ValueError):
            runner.training_labels(y, lag, 4, "zscore")

    def test_fit_trial_default_equals_explicit_change_mode(self):
        t, n = 5, 20
        y, lag = synthetic(t, n)
        x = torch.randn(t, n, 3, generator=torch.Generator().manual_seed(4))
        mask = torch.isfinite(y)
        panel = SimpleNamespace(val_index=4, campus_ids=[str(i) for i in range(n)])
        runner.stop_year, runner.campus_ids = 2023, panel.campus_ids
        window = ProposedWindow(x, [[] for _ in range(t)], [[] for _ in range(t)])
        common = dict(variant="v0_temporal", x=window, y=y, lag=lag, eval_mask=mask, target_indices=[0],
                      target_names=["t"], panel=panel, relations=[], hidden=4, dropout=0.0, epochs=3, lr=0.01,
                      weight_decay=0.0, patience=None, seed=0, device="cpu", tune=False,
                      gradient_clip_norm=None, config_hash="h")
        default = runner.fit_trial(**common)
        explicit = runner.fit_trial(**common, target_mode="change")
        torch.testing.assert_close(default["forecast"], explicit["forecast"], rtol=0, atol=0)
        relative = runner.fit_trial(**common, target_mode="change_relative")
        self.assertFalse(torch.equal(default["forecast"], relative["forecast"]))

    def _panel(self):
        years = [2018, 2019, 2020, 2021, 2022, 2023]
        n = 6
        g = torch.Generator().manual_seed(1)
        y = torch.randn(len(years), n, 2, generator=g) + 10
        lag = torch.cat([torch.randn(1, n, 2, generator=g) + 10, y[:-1]])  # lag of year t = y of t-1
        snaps = [{"campus": SimpleNamespace(x=torch.zeros(n, 3), y=y[i].clone(), y_lag1=lag[i].clone())}
                 for i in range(len(years))]
        return SimpleNamespace(years=years, snapshots=snaps, targets=["a", "b"], feature_names=["f1", "f2", "f3"],
                               split={"train_years": [2018, 2019, 2020, 2021, 2022]})

    def test_dist_2019_is_zero_before_2020_and_does_not_use_same_year_outcomes(self):
        panel = self._panel()
        runner.add_dist_2019(panel)
        self.assertEqual(panel.feature_names[-2:], ["dist_2019_a", "dist_2019_b"])
        for i in (0, 1):                                            # 2018, 2019 snapshots
            self.assertTrue(torch.equal(panel.snapshots[i]["campus"].x[:, 3:], torch.zeros(6, 2)))
        before = [s["campus"].x.clone() for s in panel.snapshots]
        # perturb every year's own outcome from 2020 on: the inputs of that year must not move
        perturbed = self._panel()
        for i in range(2, 6):
            perturbed.snapshots[i]["campus"].y += 100.0
        runner.add_dist_2019(perturbed)
        for i in range(6):
            torch.testing.assert_close(perturbed.snapshots[i]["campus"].x, before[i])

    def test_config_guards(self):
        base = {"variant": "v1_original", "through_year": 2023, "validation_only": True, "graph": "v2",
                "label": "change", "out": "s5_round2/step2/x", "tasks": "single", "relation_policy": "target_specific"}
        runner.validate_config(dict(base, target_mode="change_relative", extra_inputs=["dist_2019"]))
        with self.assertRaises(ValueError):
            runner.validate_config(dict(base, target_mode="zscore"))
        with self.assertRaises(ValueError):
            runner.validate_config(dict(base, extra_inputs=["future"]))


if __name__ == "__main__":
    unittest.main()
