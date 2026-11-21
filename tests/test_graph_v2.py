"""Tests for graph v2 weights and edge features (src/graphs/edge_features.py, graph_version.py)."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/graphs"))
import edge_features as ef
import graph_version as gv


class EdgeFeatureTests(unittest.TestCase):
    def test_local_scale_is_kth_neighbour_with_floor(self):
        d = np.array([[np.inf, 0.0, 3.0], [0.0, np.inf, 4.0], [3.0, 4.0, np.inf]])
        self.assertEqual(ef.local_scales(d, 1, 1.0).tolist(), [1.0, 1.0, 3.0])   # 0 km floored to 1 km
        self.assertEqual(ef.local_scales(d, 2, 1.0).tolist(), [3.0, 4.0, 4.0])

    def test_neighbour_at_local_scale_gets_same_weight_in_city_and_country(self):
        city = ef.self_tuning_weight(np.array([6.0]), np.array([6.0]), np.array([6.0]))
        rural = ef.self_tuning_weight(np.array([28.0]), np.array([28.0]), np.array([28.0]))
        self.assertAlmostEqual(city[0], np.exp(-1))
        self.assertAlmostEqual(rural[0], city[0])

    def test_sigma_uses_training_years_only(self):
        e = pd.DataFrame({"year": [2019, 2020, 2024], "distance": [1.0, 3.0, 100.0]})
        self.assertEqual(ef.training_sigma(e, [2019, 2020]), 2.0)
        with self.assertRaises(ValueError):
            ef.training_sigma(e, [2030])

    def test_same_distance_same_weight_in_every_year(self):
        w = ef.gaussian_weight(np.array([0.5, 0.5]), 0.5)
        self.assertEqual(w[0], w[1])
        self.assertAlmostEqual(w[0], np.exp(-0.5))

    def test_edge_attributes_standardise_on_training_years_and_zero_missing(self):
        e = pd.DataFrame({"year": [2019, 2020, 2024], "distance": [10.0, 20.0, 40.0], "weight": [1.0, 0.5, 0.1]})
        stats = ef.distance_stats(e, [2019, 2020])
        self.assertEqual(stats, (15.0, 5.0))
        self.assertEqual(ef.edge_attributes(e, stats).tolist(), [[1.0, -1.0], [0.5, 1.0], [0.1, 5.0]])
        no_distance = pd.DataFrame({"year": [2019], "distance": [np.nan], "weight": [1.0]})
        self.assertEqual(ef.edge_attributes(no_distance, ef.distance_stats(no_distance, [2019])).tolist(), [[1.0, 0.0]])


class GraphVersionTests(unittest.TestCase):
    def test_folders_and_old_flag(self):
        with patch.dict(os.environ, {"TEA_GRAPH": "v2"}, clear=False):
            self.assertEqual(gv.folder().name, "graph_v2")
            self.assertTrue(gv.expanded())
        with patch.dict(os.environ, {"TEA_EXPANDED": "1"}, clear=True):
            self.assertEqual(gv.folder().name, "graph_expanded")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gv.folder().name, "graph")
            self.assertFalse(gv.expanded())
        with patch.dict(os.environ, {"TEA_GRAPH": "v3"}, clear=True):
            with self.assertRaises(ValueError):
                gv.name()


if __name__ == "__main__":
    unittest.main()
