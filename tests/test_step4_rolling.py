"""Synthetic checks for Step 4 rolling origins; no school data is read."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "src/train"), str(ROOT / "src/models")]
import run_step4_rolling as step4  # noqa: E402
from test_step1_anchor import panel_with_history  # noqa: E402


class Step4Tests(unittest.TestCase):
    def test_view_moves_the_validation_year_and_training_years(self):
        panel, *_ = panel_with_history()
        panel.campus_ids, panel.fingerprint, panel.relations = [], "f", lambda t: []
        v = step4.view(panel, 2022)
        self.assertEqual(v.val_index, panel.years.index(2022))
        self.assertEqual(v.split["train_years"], [2018, 2019, 2020, 2021])
        self.assertEqual(panel.split["train_years"], [2018, 2019, 2020, 2021, 2022])   # original untouched

    def test_anchor_for_an_origin_ignores_that_year_and_later_outcomes(self):
        panel, y, lag, scaler = panel_with_history(n=300)
        panel.campus_ids, panel.fingerprint, panel.relations = [], "f", lambda t: []
        before, coef = step4.anchor_for(panel, y, lag, scaler, 2022)
        y2 = y.clone()
        y2[4:] += 50.0                                     # 2022 and 2023 outcomes
        after, coef2 = step4.anchor_for(panel, y2, lag, scaler, 2022)
        torch.testing.assert_close(after, before)
        self.assertEqual(coef, coef2)
        self.assertTrue(all(c["n_fit"] < 300 * 3 for c in coef))   # only 2020 and 2021 label years can be used


if __name__ == "__main__":
    unittest.main()
