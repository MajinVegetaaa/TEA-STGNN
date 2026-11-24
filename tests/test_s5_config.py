"""Static checks for locked S5 and corrected-comparator run configurations."""
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/train"))
from run_guards import validate_run_request  # noqa: E402


class S5ConfigTests(unittest.TestCase):
    def test_split_and_shared_budget_are_locked(self):
        split = yaml.safe_load((ROOT / "configs/split.yaml").read_text())
        budget = yaml.safe_load((ROOT / "configs/budget.yaml").read_text())
        self.assertEqual(split["train_years"], [2018, 2019, 2020, 2021, 2022])
        self.assertEqual(split["val_years"], [2023])
        self.assertEqual(split["test_years"], [2024])
        self.assertEqual(split["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(budget, {"grid": {"hidden": [32, 64], "dropout": [0.0, 0.2]},
                                  "lr": 0.003, "weight_decay": 0.0001, "max_epochs": 100, "patience": 20})

    def test_all_versions_are_validation_only_and_use_the_locked_protocol(self):
        for version in range(4):
            with self.subTest(version=version):
                path = ROOT / f"configs/experiments/s5_proposed_v{version}.yaml"
                cfg = yaml.safe_load(path.read_text())
                self.assertEqual(cfg["model"], "proposed")
                self.assertEqual(cfg["model_args"]["version"], version)
                self.assertEqual(cfg["graph"], "v2")
                self.assertEqual(cfg["through_year"], 2023)
                self.assertIs(cfg["validation_only"], True)
                self.assertEqual(cfg["label"], "change")
                self.assertEqual(cfg["budget"], "configs/budget.yaml")
                self.assertEqual(cfg["tasks"], "multi" if version == 3 else "single")
                validate_run_request(cfg, test_requested=False, val_seeds=True)
                with self.assertRaisesRegex(ValueError, "reserved for S6"):
                    validate_run_request(cfg, test_requested=True, val_seeds=True)

    def test_comparator_uses_a_separate_validation_only_output(self):
        cfg = yaml.safe_load((ROOT / "configs/experiments/s4_mtgnn_comparable.yaml").read_text())
        self.assertEqual(cfg["through_year"], 2023)
        self.assertIs(cfg["validation_only"], True)
        self.assertEqual(cfg["out"], "s4/mtgnn_corrected_snapshot")
        validate_run_request(cfg, test_requested=False, val_seeds=True)

    def test_make_s5_sequence_runs_comparator_before_proposed_versions(self):
        recipe = (ROOT / "Makefile").read_text().split("s5:", 1)[1]
        steps = ["s4_mtgnn_comparable.yaml", *(f"s5_proposed_v{i}.yaml" for i in range(4)),
                 "report_neural.py --step s5"]
        positions = [recipe.index(step) for step in steps]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
