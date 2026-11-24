"""Synthetic report validation; no school outcomes or trained artifacts are read."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/eval"))
from report_s5_round2 import RUNS, SEEDS, build_report  # noqa: E402


def make_runs(root: Path, *, omit_seed=False, duplicate=False):
    targets = [f"target_{i}" for i in range(10)]
    for model, folder in RUNS.items():
        run = root / folder
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({"status": "completed", "snapshot_fingerprint": "synthetic"}))
        (run / "protocol.json").write_text(json.dumps({"training_years": [2018, 2019, 2020, 2021, 2022],
            "validation_years": [2023], "snapshot_fingerprint": "synthetic"}))
        rows = []
        seeds = [0, 1, 2, 3] if omit_seed else SEEDS
        score = 1.0 + list(RUNS).index(model) / 100
        for target in targets:
            for seed in seeds:
                rows.append({"target": target, "seed": seed, "normalized_mae": score + seed / 1000,
                             "val_mae": score + seed / 1000, "persistence_mae": 1.0,
                             "snapshot_fingerprint": "synthetic"})
        if duplicate:
            rows.append(rows[0].copy())
        pd.DataFrame(rows).to_csv(run / "per_seed_metrics.csv", index=False)


class Round2ReportTests(unittest.TestCase):
    def test_report_calculates_paired_results_and_compute_disclosure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_runs(root)
            report = build_report(root)
            summary = report["summary"].set_index("model")
            self.assertEqual(summary.loc["V1_shared4_multi", "fit_count"], 9)
            self.assertEqual(summary.loc["V3_planned", "fit_count"], 9)
            self.assertEqual(summary.loc["V0", "fit_count"], 90)
            self.assertEqual(len(report["paired_vs_mtgnn"]), 80)
            self.assertEqual(len(report["per_seed"]), 45)
            self.assertTrue(report["summary"].selected_proposed.any())

    def test_report_fails_closed_on_missing_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_runs(root, omit_seed=True)
            with self.assertRaisesRegex(ValueError, "missing, duplicate, or unexpected seed"):
                build_report(root)

    def test_report_fails_closed_on_duplicate_seed_target_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_runs(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, "missing, duplicate, or unexpected seed"):
                build_report(root)

    def test_tie_rule_uses_component_based_simplicity_order(self):
        from report_s5_round2 import COMPLEXITY
        self.assertEqual(set(COMPLEXITY), set(RUNS) - {"MTGNN"})
        self.assertLess(COMPLEXITY["V1_shared4_single"], COMPLEXITY["V1_original"])
        self.assertLess(COMPLEXITY["V1_original"], COMPLEXITY["V2_original"])
        self.assertLess(COMPLEXITY["V2_original"], COMPLEXITY["V3_planned"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_runs(root)
            # Every model within one seed SD of the best: the simplest (V0) must win.
            for folder in RUNS.values():
                path = root / folder / "per_seed_metrics.csv"
                frame = pd.read_csv(path)
                frame["normalized_mae"] = 1.0 + frame.seed / 10
                frame.to_csv(path, index=False)
            summary = build_report(root)["summary"].set_index("model")
            self.assertTrue(summary.loc["V0", "selected_proposed"])
            self.assertEqual(int(summary.selected_proposed.sum()), 1)


if __name__ == "__main__":
    unittest.main()
