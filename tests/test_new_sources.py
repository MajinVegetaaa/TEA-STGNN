"""Regression tests for data-loss, mask reconstruction and wrong-population risks."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/data"))
import new_sources as ns
import build_master as bm
import validate_additions as va


class SourceTests(unittest.TestCase):
    def test_masks_never_become_numeric_values(self):
        values, status = ns.numeric(pd.Series(["-1", "-2", "-3", "<1,000", "-", "", ".", "0", "1,002.5"]))
        self.assertTrue(values.iloc[:7].isna().all())
        self.assertEqual(values.iloc[7:].tolist(), [0, 1002.5])
        self.assertEqual(status.iloc[:3].tolist(), ["suppressed_1", "suppressed_2", "suppressed_3"])

    def test_unknown_tokens_fail_instead_of_becoming_missing(self):
        for token in ("oops", "-5", "inf"):
            with self.assertRaises(ValueError):
                ns.numeric(pd.Series([token]))

    def test_exam_blank_does_not_assert_suppression(self):
        value, status = ns.numeric(pd.Series([""]), ambiguous_blank=True)
        self.assertTrue(value.isna().all())
        self.assertEqual(status.iloc[0], "undefined_or_suppressed")

    def test_duplicate_graduation_variants_fail_after_filter(self):
        table = pd.DataFrame({"CAMPUS": ["1", "1", "1"], "CALC_FOR_STATE_ACCT": ["No", "No", "Yes"]})
        with patch.object(ns, "read_table", return_value=table):
            with self.assertRaises(ValueError):
                ns.selected_table("graduation_4yr", 2024, {})

    def test_graduation_uses_federal_variant(self):
        table = pd.DataFrame({"CAMPUS": ["1", "1"], "CALC_FOR_STATE_ACCT": ["Yes", "No"]})
        with patch.object(ns, "read_table", return_value=table):
            result = ns.selected_table("graduation_4yr", 2024, {})
        self.assertEqual(result.CALC_FOR_STATE_ACCT.tolist(), ["No"])
        self.assertEqual(result.campus_id.tolist(), ["000000001"])

    def test_missing_primary_group_rejected(self):
        with self.assertRaises(ValueError):
            ns.verify_year("staar_eoc", 2018, {}, pd.DataFrame({"CDG00AA11218N": [1]}), {})

    def staar_table(self, published):
        t = pd.DataFrame({"campus_id": ["000000001", "000000002", "000000003"]})
        for code in ns.SUBJECTS.values():
            t[f"CDA00{code}1224N"] = ["10", "-1", "0"]
            t[f"CDA00{code}1024D"] = ["20", "20", "0"]
            if published:
                t[f"CDA00{code}1224R"] = ["-1", "-1", "."]
        return t

    def test_published_suppressed_rate_not_reconstructed(self):
        with patch.object(ns, "selected_table", return_value=self.staar_table(True)), patch.object(ns, "verify_year", return_value="ok"):
            out = ns.load_source("staar_eoc", {"staar_eoc": {2024: {"file": "test.csv"}}}, lambda x: None)
        self.assertTrue(out.staar_algebra1_meets.isna().all())

    def test_derive_only_unsuppressed_positive_denominator(self):
        with patch.object(ns, "selected_table", return_value=self.staar_table(False)), patch.object(ns, "verify_year", return_value="ok"):
            out = ns.load_source("staar_eoc", {"staar_eoc": {2024: {"file": "test.csv"}}}, lambda x: None)
        self.assertEqual(out.staar_algebra1_meets.iloc[0], 50)
        self.assertTrue(out.staar_algebra1_meets.iloc[1:].isna().all())
        self.assertEqual(out.staar_algebra1_meets_status.iloc[2], "zero_denominator")

    def test_missing_calendar_year_is_not_bridged(self):
        m = pd.DataFrame({"campus_id": ["000000001"] * 3, "year": [2018, 2020, 2021]})
        for col in bm.TARGETS + bm.DEMOG_NAMES + ["teacher_experience"]:
            m[col] = [10., 20., 30.]
        result = bm.temporal_features(m, ["teacher_experience"]).set_index("year")
        self.assertTrue(pd.isna(result.at[2020, "teacher_experience_lag1"]))
        self.assertTrue(pd.isna(result.at[2021, "teacher_experience_ma3"]))
        self.assertTrue(pd.isna(result.at[2021, "teacher_experience_trend3"]))

    def test_stale_gate_does_not_authorize_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master, gate = root / "master.csv", root / "gate.json"
            master.write_text("original")
            gate.write_text(json.dumps({"passed": True, "files": {"master.csv": va.sha(master)}}))
            with patch.object(va, "ROOT", root), patch.object(va, "GATE", gate), patch.object(va, "gate_files", return_value=[master]):
                va.require_master_gate()
                master.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "Stale"):
                    va.require_master_gate()

    def test_failed_gate_does_not_authorize_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "gate.json"
            gate.write_text(json.dumps({"passed": False}))
            with patch.object(va, "GATE", gate):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    va.require_master_gate()


if __name__ == "__main__":
    unittest.main()
