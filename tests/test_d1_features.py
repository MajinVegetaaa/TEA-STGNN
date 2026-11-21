"""D1 features never use year-t information: scrambling every added source within year t leaves year-t rows unchanged."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/data"))
P = ROOT / "data/processed"


@unittest.skipUnless((P / "d1_temporal_features.csv").exists(), "run make features first")
class D1LeakageTest(unittest.TestCase):
    def test_scrambling_year_t_leaves_year_t_features_unchanged(self):
        from build_master import temporal_features
        from prepare_d1_features import SOURCES
        m = pd.read_csv(P / "master_school_year.csv", dtype={"campus_id": str})
        d1 = pd.read_csv(P / "d1_temporal_features.csv", dtype={"campus_id": str})
        rng = np.random.default_rng(0)
        for year in (2023, 2024):
            scrambled = m.copy()
            rows = scrambled.year == year
            for v in SOURCES:
                scrambled.loc[rows, v] = rng.permutation(scrambled.loc[rows, v].to_numpy())
            rebuilt = temporal_features(scrambled, SOURCES)
            cols = [c for c in d1.columns if any(c.startswith(v + "_") for v in SOURCES)]
            a = rebuilt[rebuilt.year == year].set_index("campus_id")[cols].sort_index()
            b = d1[d1.year == year].set_index("campus_id")[cols].sort_index()
            pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-12)
            # and the scramble did change something, so the test has teeth
            later = rebuilt[rebuilt.year == year + 1] if year < 2024 else None
            if later is not None:
                c = d1[d1.year == year + 1].set_index("campus_id")[cols].sort_index()
                self.assertFalse(later.set_index("campus_id")[cols].sort_index().equals(c))


if __name__ == "__main__":
    unittest.main()
