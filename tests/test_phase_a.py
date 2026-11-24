"""Synthetic checks for the Phase A diagnostics; no school data is read."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/eval"))
from phase_a_rolling import fit_score, morans_i, neighbour_mean, read_through  # noqa: E402
from phase_a_shift import decompose  # noqa: E402


def edges(pairs, w=1.0):
    rows = [(a, b, w) for a, b in pairs] + [(b, a, w) for a, b in pairs]
    return pd.DataFrame(rows, columns=["src_campus_id", "dst_campus_id", "weight"])


class PhaseATests(unittest.TestCase):
    def test_reader_stops_at_cutoff_and_refuses_unsorted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("campus_id,year,v\n001,2022,1\n001,2023,2\n001,2024,999\n001,2025,not-a-number\n")
            df = read_through(path, cutoff=2023)
            self.assertEqual(df.year.tolist(), [2022, 2023])
            self.assertEqual(df.campus_id.tolist(), ["001", "001"])      # ids stay strings
            path.write_text("campus_id,year,v\n001,2023,1\n001,2022,2\n")
            with self.assertRaisesRegex(ValueError, "not sorted"):
                read_through(path, cutoff=2023)

    def test_neighbour_mean_is_weighted_excludes_self_and_skips_missing(self):
        el = pd.DataFrame({"src_campus_id": ["a", "a", "a", "a"], "dst_campus_id": ["a", "b", "c", "d"],
                           "weight": [5.0, 1.0, 3.0, 1.0]})
        values = pd.Series({"a": 100.0, "b": 2.0, "c": 6.0, "d": np.nan})
        self.assertAlmostEqual(neighbour_mean(el, values)["a"], (2 * 1 + 6 * 3) / 4)

    def test_morans_i_sign_on_clustered_and_alternating_values(self):
        rng = np.random.default_rng(0)
        two_cliques = edges([("a", "b"), ("a", "c"), ("b", "c"), ("d", "e"), ("d", "f"), ("e", "f")])
        clustered = pd.Series({"a": 1.0, "b": 1.1, "c": 0.9, "d": -1.0, "e": -1.1, "f": -0.9})
        i, p, m = morans_i(two_cliques, clustered, perms=99, rng=rng)
        self.assertGreater(i, 0.8)
        self.assertEqual(m, 6)
        path = edges([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")])
        alternating = pd.Series({"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0, "e": 1.0, "f": -1.0})
        self.assertLess(morans_i(path, alternating, perms=9, rng=rng)[0], -0.9)

    def test_correction_ignores_year_level_shift_and_uses_per_school_signal(self):
        rng = np.random.default_rng(1)
        frames = []
        for year, shift in ((2021, 5.0), (2022, -5.0), (2023, 3.0)):
            nbr = rng.normal(size=400)
            frames.append(pd.DataFrame({"year": year, "own": rng.normal(size=400), "nbr": nbr,
                                        "e": shift + 0.8 * nbr + rng.normal(scale=0.3, size=400)}))
        design = pd.concat(frames, ignore_index=True)
        train, test = design[design.year < 2023], design[design.year == 2023]
        tree_alone = fit_score(train, test, [])
        with_nbr = fit_score(train, test, ["own", "nbr"])
        self.assertLess(with_nbr, tree_alone)
        # the past years' opposite shifts must not be carried forward as an intercept
        self.assertLess(abs((test.e - 0.8 * (test.nbr - test.nbr.mean())).mean() - 3.0), 0.2)

    def test_decompose_separates_shift_from_spread(self):
        y = np.array([10.0, 12.0, 14.0, 16.0])
        lag = y - 2.0                                     # every school rose by 2
        d = decompose(y, lag, lag + 2.0)                  # predicting the common shift exactly
        self.assertAlmostEqual(d["ratio"], 0.0)
        self.assertAlmostEqual(d["pers_bias"], -2.0)
        self.assertAlmostEqual(d["oracle_shift"], 0.0)
        noisy = decompose(y, lag, y + np.array([1.0, -1.0, 1.0, -1.0]))
        self.assertAlmostEqual(noisy["bias"], 0.0)
        self.assertAlmostEqual(noisy["spread_mae"], 1.0)
        with self.assertRaises(ValueError):
            decompose(y, y, y)                            # zero persistence error is rejected


if __name__ == "__main__":
    unittest.main()
