"""Categorical vocabularies must not be learned from validation or test rows."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/graphs"))
from feature_encoding import train_only_onehot  # noqa: E402


class TrainOnlyCategoryTests(unittest.TestCase):
    def test_unseen_later_category_does_not_create_a_feature(self):
        frame = pd.DataFrame({"urbanicity": ["Town", "City", "FutureClass"],
                              "region": ["01", "02", "99"]})
        train = [True, True, False]
        encoded, vocabulary = train_only_onehot(frame, train, ["urbanicity", "region"])
        self.assertEqual(vocabulary, {"urbanicity": ["City", "Town"], "region": ["01", "02"]})
        self.assertNotIn("urbanicity_FutureClass", encoded.columns)
        self.assertNotIn("region_99", encoded.columns)
        self.assertEqual(float(encoded.iloc[2].sum()), 0.0)

    def test_known_categories_match_expected_onehot_values(self):
        frame = pd.DataFrame({"kind": ["A", "B", "A"]})
        encoded, _ = train_only_onehot(frame, [True, True, False], ["kind"])
        self.assertEqual(encoded.columns.tolist(), ["kind_A", "kind_B"])
        self.assertEqual(encoded.iloc[2].tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
