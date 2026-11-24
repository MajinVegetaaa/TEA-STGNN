"""Year-cutoff checks for the neural panel loader."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/train"))
import data as panel_data  # noqa: E402
from run_guards import require_fresh_output, validate_run_request  # noqa: E402


class PanelWindowTests(unittest.TestCase):
    def setUp(self):
        self.years = list(range(2018, 2025))
        self.snapshot = {
            "years": self.years,
            "split": {"train_years": [2018, 2019, 2020, 2021, 2022],
                      "val_years": [2023], "test_years": [2024], "seeds": [0, 1, 2, 3, 4]},
            "campus_ids": ["000000001"],
            "targets": ["target"],
            "edge_attr": ["weight", "distance_z"],
            "academic_relation": {},
            "scaler": {"fit_years": [2018, 2019, 2020, 2021, 2022],
                       "categories_fit_years": [2018, 2019, 2020, 2021, 2022],
                       "category_vocabulary": {"urbanicity": ["Town"]}, "unknown_category": "all_zero"},
            "snapshots": [
                {"campus": {"y": torch.tensor([[float(year)]]),
                            "y_lag1": torch.tensor([[float(year - 1)]]),
                            "eval_mask": torch.ones(1, 1, dtype=torch.bool),
                            "present": torch.ones(1, dtype=torch.bool)}}
                for year in self.years
            ],
        }

    def load_with_cutoff(self, cutoff):
        with patch.dict(panel_data.SNAPSHOTS, {"v2": Path("unused-snapshot.pt")}), \
             patch.object(panel_data.torch, "load", return_value=self.snapshot), \
             patch.object(Path, "read_bytes", return_value=b"snapshot"):
            return panel_data.load("v2", through_year=cutoff)

    def test_validation_panel_excludes_test_year(self):
        panel = self.load_with_cutoff(2023)
        self.assertEqual(panel.years, self.years[:-1])
        self.assertEqual(panel.y.shape[0], 6)
        self.assertEqual(panel.val_index, 5)
        self.assertNotIn(2024, panel.years)
        self.assertEqual(panel.split["test_years"], [])
        self.assertEqual(float(panel.y[-1, 0, 0]), 2023.0)

    def test_cutoff_cannot_include_test_year(self):
        with self.assertRaisesRegex(ValueError, "exceeds the validation cutoff"):
            self.load_with_cutoff(2024)

    def test_validation_only_run_rejects_test_mode(self):
        with self.assertRaisesRegex(ValueError, "reserved for S6"):
            validate_run_request({"model": "proposed", "out": "s5/proposed_v0",
                                  "through_year": 2023, "validation_only": True, "graph": "v2",
                                  "label": "change"}, test_requested=True, val_seeds=True)
        with self.assertRaisesRegex(ValueError, "--val-seeds"):
            validate_run_request({"validation_only": True}, test_requested=False)
        validate_run_request({"validation_only": True}, test_requested=False, val_seeds=True)

    def test_s5_cannot_drop_its_2023_cutoff_guard(self):
        with self.assertRaisesRegex(ValueError, "cap the panel at 2023"):
            validate_run_request({"model": "proposed", "out": "s5/proposed_v0"}, test_requested=False)

    def test_proposed_model_cannot_bypass_guard_by_renaming_output(self):
        with self.assertRaisesRegex(ValueError, "cap the panel at 2023"):
            validate_run_request({"model": "proposed", "out": "custom/copied_run",
                                  "validation_only": True}, test_requested=False)

    def test_s5_requires_the_locked_graph_label_and_five_seed_mode(self):
        cfg = {"model": "proposed", "out": "s5/proposed_v0", "through_year": 2023,
               "validation_only": True, "graph": "v2", "label": "change"}
        with self.assertRaisesRegex(ValueError, "--val-seeds"):
            validate_run_request(cfg, test_requested=False)
        validate_run_request(cfg, test_requested=False, val_seeds=True)
        wrong_graph = {**cfg, "graph": "v1"}
        with self.assertRaisesRegex(ValueError, "Graph V2"):
            validate_run_request(wrong_graph, test_requested=False, val_seeds=True)

    def test_validation_cutoff_blocks_test_mode_for_any_model(self):
        with self.assertRaisesRegex(ValueError, "reserved for S6"):
            validate_run_request({"model": "mtgnn", "through_year": 2023}, test_requested=True)

    def test_nonempty_experiment_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            require_fresh_output(out)
            out.mkdir()
            (out / "protocol.json").write_text("preserve")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                require_fresh_output(out)

    def test_s5_loader_rejects_snapshot_without_train_only_category_audit(self):
        with patch.dict(panel_data.SNAPSHOTS, {"v2": Path("unused-snapshot.pt")}), \
             patch.object(panel_data.torch, "load", return_value={**self.snapshot,
                 "scaler": {"fit_years": [2018, 2019, 2020, 2021, 2022],
                            "categories_fit_years": [2018, 2019, 2020, 2021, 2022, 2023],
                            "category_vocabulary": {"urbanicity": ["Town"]}, "unknown_category": "all_zero"}}), \
             patch.object(Path, "read_bytes", return_value=b"snapshot"):
            with self.assertRaisesRegex(ValueError, "categorical vocabulary was fit on training years only"):
                panel_data.load("v2", through_year=2023, require_train_only_categories=True)

    def test_s5_loader_accepts_train_only_snapshot_provenance(self):
        with patch.dict(panel_data.SNAPSHOTS, {"v2": Path("unused-snapshot.pt")}), \
             patch.object(panel_data.torch, "load", return_value=self.snapshot), \
             patch.object(Path, "read_bytes", return_value=b"snapshot"):
            panel = panel_data.load("v2", through_year=2023, require_train_only_categories=True)
        self.assertEqual(panel.years[-1], 2023)


if __name__ == "__main__":
    unittest.main()
