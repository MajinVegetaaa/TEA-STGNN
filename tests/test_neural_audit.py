"""S5 report must refuse unfair baseline comparisons."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/eval"))
from neural_audit import validate_s5_comparators  # noqa: E402


def protocols():
    runs = ["s4/mtgnn", *(f"s5/proposed_v{i}" for i in range(4))]
    return {
        run: {"snapshot_sha256": "same-snapshot", "budget": {"lr": 0.003},
              "grid_points": [{"hidden": 32, "dropout": 0}], "targets": ["a", "b"],
              "test_run": False,
              "config": {"graph": "v2", "label": "change",
                         "model": "mtgnn" if run == "s4/mtgnn" else "proposed",
                         "validation_only": run.startswith("s5/"),
                         **({"through_year": 2023} if run.startswith("s5/") else {})}}
        for run in runs
    }


class NeuralAuditTests(unittest.TestCase):
    def test_matching_s5_runs_are_comparable(self):
        validate_s5_comparators(protocols(), [0, 1, 2, 3, 4])

    def test_mismatched_snapshot_is_rejected(self):
        records = protocols()
        records["s5/proposed_v2"]["snapshot_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "same snapshot"):
            validate_s5_comparators(records, [0, 1, 2, 3, 4])

    def test_non_validation_run_is_rejected(self):
        records = protocols()
        records["s5/proposed_v1"]["test_run"] = True
        with self.assertRaisesRegex(ValueError, "2023-only"):
            validate_s5_comparators(records, [0, 1, 2, 3, 4])

    def test_wrong_model_family_is_rejected(self):
        records = protocols()
        records["s5/proposed_v0"]["config"]["model"] = "mtgnn"
        with self.assertRaisesRegex(ValueError, "expected model family"):
            validate_s5_comparators(records, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
