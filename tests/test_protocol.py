"""Tests for the locked evaluation protocol (configs/protocol.yaml, src/eval/protocol.py)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/eval"))
import protocol as pr


class ProtocolTests(unittest.TestCase):
    def test_config_is_complete_and_never_tests_on_selection_years(self):
        cfg = pr.load()
        split = cfg["split"]
        self.assertEqual(len(cfg["targets"]), 10)
        self.assertEqual(cfg["selection"]["year"], split["val_years"][0])
        self.assertFalse(set(split["test_years"]) & set(split["train_years"] + split["val_years"]))
        self.assertEqual(cfg["forecast_rule"]["name"], "retrospective")
        self.assertEqual(cfg["neural_target"], "change_from_last_year")

    def test_scored_rows_need_target_and_last_year(self):
        y = pd.Series([1.0, np.nan, 3.0, 4.0])
        last = pd.Series([1.0, 2.0, np.nan, 4.0])
        self.assertEqual(pr.scored_rows(y, last).tolist(), [True, False, False, True])

    def test_change_round_trip_and_zero_change_is_persistence(self):
        y, last = np.array([10.0, 12.5]), np.array([9.0, 13.0])
        self.assertTrue(np.allclose(pr.from_change(pr.to_change(y, last), last), y))
        self.assertTrue(np.allclose(pr.from_change(np.zeros(2), last), last))

    def test_score_is_mean_ratio_to_persistence(self):
        model = pd.DataFrame({"a": [1.0, 1.2], "b": [4.0, 4.0]})     # two seeds
        persistence = pd.Series({"a": 2.0, "b": 4.0})
        score, sd = pr.selection_score(model, persistence)
        self.assertAlmostEqual(score, np.mean([(0.5 + 1.0) / 2, (0.6 + 1.0) / 2]))
        self.assertAlmostEqual(sd, np.std([0.75, 0.8], ddof=1))
        persistence_only = pr.selection_score(pd.DataFrame({"a": [2.0]}), persistence)
        self.assertEqual(persistence_only, (1.0, 0.0))

    def test_score_requires_persistence_for_every_target(self):
        with self.assertRaises(ValueError):
            pr.selection_score(pd.DataFrame({"a": [1.0], "z": [1.0]}), pd.Series({"a": 2.0}))

    def test_tie_goes_to_the_simpler_model(self):
        c = pd.DataFrame({"score": [0.900, 0.905, 0.95], "sd": [0.01, 0.01, 0.01], "complexity": [3, 1, 0]},
                         index=["proposed", "gru_only", "persistence_like"])
        self.assertEqual(pr.choose(c), "gru_only")        # within one SD of the best, simpler
        c.loc["gru_only", "score"] = 0.95
        self.assertEqual(pr.choose(c), "proposed")       # outside one SD: best score wins


if __name__ == "__main__":
    unittest.main()
