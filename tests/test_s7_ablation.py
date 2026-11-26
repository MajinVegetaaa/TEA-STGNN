"""Checks for the S7 post-test ablation driver (no model is trained)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/s7"), str(ROOT / "src/s6"), str(ROOT / "src/train"), str(ROOT / "src/models")]
import ablation_run as A  # noqa: E402
import run_step4_rolling as rolling  # noqa: E402


class AblationPlanTests(unittest.TestCase):
    def test_plan_covers_every_version_setup_and_origin_once(self):
        runs = A.plan()
        self.assertEqual(len(runs), len(set(runs)))
        everything = set(runs) | set(A.REUSED)
        for setup in A.SETUPS:
            for model in A.STGNN:
                for origin in A.ORIGINS:
                    self.assertIn((setup, model, origin), everything)
        self.assertFalse(any(m == "mtgnn" and s == "planA" for s, m, _ in everything))
        self.assertEqual(len(runs), 26)

    def test_reused_runs_and_planA_mtgnn_are_refused(self):
        with self.assertRaises(SystemExit):
            A.one("planA", "v0", 2023, "cpu")
        with self.assertRaises(SystemExit):
            A.one("planA", "mtgnn", 2023, "cpu")

    def test_versions_map_to_the_round2_variants(self):
        self.assertEqual(rolling.MODELS["v1s4"], ("v3_single_shared", "shared_four"))
        self.assertEqual(rolling.MODELS["v2"], ("v2_original", "target_specific"))
        self.assertEqual(rolling.MODELS["v3"], ("v3_planned", "union_targets"))
        self.assertEqual(rolling.MULTI_TASK, {"v3"})


if __name__ == "__main__":
    unittest.main()
