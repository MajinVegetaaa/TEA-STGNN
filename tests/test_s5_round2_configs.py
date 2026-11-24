"""Static protocol and output-isolation checks for every Round 2 config."""
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/train"))
from run_s5_round2 import validate_config  # noqa: E402


class Round2ConfigTests(unittest.TestCase):
    def test_all_round2_configs_share_the_locked_budget_and_cutoff(self):
        budget = yaml.safe_load((ROOT / "configs/budget.yaml").read_text())
        configs = sorted((ROOT / "configs/experiments").glob("s5_round2_*.yaml"))
        self.assertEqual(len(configs), 8)
        outputs = set()
        for path in configs:
            cfg = yaml.safe_load(path.read_text())
            validate_config(cfg)
            self.assertEqual(cfg["budget"], "configs/budget.yaml")
            self.assertEqual(cfg["through_year"], 2023)
            self.assertIs(cfg["validation_only"], True)
            self.assertEqual(cfg["graph"], "v2")
            self.assertEqual(cfg["label"], "change")
            self.assertEqual(cfg["snapshot_bundle"], "data/graph_v2/pyg/s5_validation")
            self.assertEqual(cfg.get("gradient_clip_norm"), None)
            self.assertEqual(cfg["out"].split("/")[0], "s5_round2")
            outputs.add(cfg["out"])
        self.assertEqual(len(outputs), 8)
        self.assertEqual(budget["grid"], {"hidden": [32, 64], "dropout": [0.0, 0.2]})

    def test_v3_pair_has_same_relations_but_different_task_sharing(self):
        single = yaml.safe_load((ROOT / "configs/experiments/s5_round2_v3_single_shared.yaml").read_text())
        multi = yaml.safe_load((ROOT / "configs/experiments/s5_round2_v3_multi_shared.yaml").read_text())
        self.assertEqual(single["relation_policy"], multi["relation_policy"])
        self.assertEqual(single["loss"], multi["loss"])
        self.assertEqual(single["tasks"], "single")
        self.assertEqual(multi["tasks"], "multi")
        self.assertEqual(single["variant"], "v3_single_shared")
        self.assertEqual(multi["variant"], "v3_multi_shared")

    def test_original_round1_configs_are_not_retargeted(self):
        for version in range(4):
            cfg = yaml.safe_load((ROOT / f"configs/experiments/s5_proposed_v{version}.yaml").read_text())
            self.assertEqual(cfg["out"], f"s5/proposed_v{version}")


if __name__ == "__main__":
    unittest.main()
