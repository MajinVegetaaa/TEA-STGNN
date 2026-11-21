"""Regression tests for the PEIMS finance integration: the rules that, if wrong, silently change spending."""
import csv
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/data"))
import new_sources as ns

DISTRICT_HEAD = ["DISTRICT", "FUND", "FUNDYEAR", "FUNCTION", "OBJECT", "FIN_UNIT", "PROGRAM_INTENT", "ACTAMT", "DTUPDATE"]
CHARTER_HEAD = ["DISTRICT", "CS_NONPROF_ASSET", "FISCALYR", "CS_NONPROF_FUNC", "CS_NONPROF_OBJ", "FIN_UNIT",
                "CS_NONPROF_PGMIN", "ACTAMT", "DTUPDATE"]


def archive(directory: Path, name: str, head: list[str], rows: list[list], filler: int = 40) -> Path:
    """Write a one-member zip. `filler` rows of year-stamped, out-of-scope spending keep the year evidence above 95%."""
    year = rows[0][2] if rows else "2024"
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(head)
    writer.writerows(rows)
    for i in range(filler):
        writer.writerow(["999999", "199", year, "99", "5719", "000", "00", "1", "04MAR2025"])
    path = directory / name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), out.getvalue())
    return path


def row(district, fund, fy, function, obj, unit, amount, updated="04MAR2025"):
    return [district, fund, str(fy), function, obj, unit, "00", str(amount), updated]


class FinanceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def sums(self, rows, kind="district", year=2024):
        head = DISTRICT_HEAD if kind == "district" else CHARTER_HEAD
        path = archive(self.dir, "a.zip", head, rows)
        frame, _ = ns.finance_archive(path, kind, year)
        return frame

    def test_every_row_counts_whatever_its_fund_year(self):
        # FUNDYEAR is the grant year: a 2022-tagged row inside the 2024 archive is still 2024 spending.
        frame = self.sums([row("001902", "199", 2024, "11", "6119", "001", 100),
                           row("001902", "211", 2022, "11", "6119", "001", 40)])
        self.assertEqual(frame[frame.campus_id == "001902001"].amount.sum(), 140)

    def test_only_campus_coded_operating_objects_count(self):
        frame = self.sums([row("001902", "199", 2024, "11", "6119", "001", 100),
                           row("001902", "199", 2024, "11", "6119", "000", 7),      # district-wide
                           row("001902", "199", 2024, "11", "6500", "001", 11),     # debt service
                           row("001902", "199", 2024, "11", "6629", "001", 13),     # capital outlay
                           row("001902", "199", 2024, "11", "5719", "001", 17),     # revenue
                           row("001902", "199", 2024, "11", "6099", "001", 19),
                           row("001902", "199", 2024, "11", "6499", "001", 5)])     # still operating
        self.assertEqual(frame.campus_id.unique().tolist(), ["001902001"])
        self.assertEqual(frame.amount.sum(), 105)

    def test_charter_columns_are_mapped(self):
        frame = self.sums([["003801", "420", "2024", "11", "6119", "042", "30", "250", "04MAR2025"]], kind="charter")
        self.assertEqual(frame.loc[frame.campus_id == "003801042", ["function", "general_fund", "amount"]].values.tolist(),
                         [["11", True, 250]])

    def test_general_fund_codes_differ_by_archive_type(self):
        district = self.sums([row("001902", "420", 2024, "11", "6119", "001", 5)])
        charter = self.sums([["001902", "420", "2024", "11", "6119", "001", "30", "5", "04MAR2025"]], kind="charter")
        self.assertFalse(district.general_fund.iloc[0])    # fund 420 is a charter general fund only
        self.assertTrue(charter.general_fund.iloc[0])

    def test_wrong_year_or_unstable_update_date_is_rejected(self):
        path = archive(self.dir, "a.zip", DISTRICT_HEAD, [row("001902", "199", 2024, "11", "6119", "001", 1)])
        with self.assertRaisesRegex(ValueError, "does not support year"):
            ns.finance_archive(path, "district", 2023)
        mixed = archive(self.dir, "b.zip", DISTRICT_HEAD, [row("001902", "199", 2024, "11", "6119", "001", 1),
                                                           row("001902", "199", 2024, "11", "6119", "001", 1, "05MAR2025")])
        with self.assertRaisesRegex(ValueError, "one DTUPDATE"):
            ns.finance_archive(mixed, "district", 2024)
        late = archive(self.dir, "c.zip", DISTRICT_HEAD, [row("001902", "199", 2024, "11", "6119", "001", 1, "04OCT2025")],
                       filler=0)
        with self.assertRaisesRegex(ValueError, "not early"):
            ns.finance_archive(late, "district", 2024)

    def test_malformed_identifiers_fail(self):
        path = archive(self.dir, "a.zip", DISTRICT_HEAD, [row("1902", "199", 2024, "11", "6119", "001", 1)])
        with self.assertRaisesRegex(ValueError, "malformed DISTRICT"):
            ns.finance_archive(path, "district", 2024)


class LoadFinanceTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def load(self, district_rows, charter_rows, enrollment):
        archive(self.dir, "d.zip", DISTRICT_HEAD, district_rows)
        archive(self.dir, "c.zip", CHARTER_HEAD, charter_rows)
        master = pd.DataFrame({"campus_id": list(enrollment), "year": 2024, "enrollment": list(enrollment.values())})
        src = {"finance_peims": {2024: {"file": "d.zip", "charter_file": "c.zip"}}}
        with patch.object(ns, "RAW", self.dir):
            return ns.load_finance(src, master, log=lambda *a: None).set_index("campus_id")

    def test_negative_adjustments_net_out_and_never_become_a_zero_value(self):
        out = self.load([row("001902", "199", 2024, "11", "6119", "001", 100),
                         row("001902", "199", 2024, "11", "6119", "001", -30),
                         row("001902", "199", 2024, "11", "6119", "002", 10),
                         row("001902", "199", 2024, "11", "6119", "002", -25)], [],
                        {"001902001": 10, "001902002": 10})
        self.assertEqual(out.at["001902001", "finance_instruction_expenditure"], 70)
        self.assertEqual(out.at["001902001", "finance_instruction_per_student"], 7)
        self.assertTrue(np.isnan(out.at["001902002", "finance_instruction_expenditure"]))
        self.assertEqual(out.at["001902002", "finance_instruction_expenditure_status"], "nonpositive_total")
        self.assertEqual(out.at["001902002", "finance_instruction_per_student_status"], "nonpositive_total")

    def test_measures_use_their_own_functions_and_general_fund(self):
        out = self.load([row("001902", "199", 2024, "11", "6119", "001", 100),    # instruction, general fund
                         row("001902", "211", 2024, "11", "6119", "001", 50),     # instruction, grant
                         row("001902", "199", 2024, "23", "6119", "001", 20),     # leadership: core only
                         row("001902", "199", 2024, "35", "6119", "001", 9),      # food: in neither measure
                         row("001902", "199", 2024, "51", "6119", "001", 8)], [], {"001902001": 10})
        r = out.loc["001902001"]
        self.assertEqual((r.finance_instruction_expenditure, r.finance_instruction_gf_expenditure), (150, 100))
        self.assertEqual((r.finance_core_operating_expenditure, r.finance_core_operating_gf_expenditure), (170, 120))
        self.assertEqual(r.finance_core_operating_gf_per_student, 12)

    def test_denominator_statuses(self):
        out = self.load([row("001902", "199", 2024, "11", "6119", "001", 100),
                         row("001902", "199", 2024, "11", "6119", "002", 100),
                         row("001902", "199", 2024, "11", "6119", "003", 100)], [],
                        {"001902001": np.nan, "001902002": 0, "001902003": 20})
        statuses = out.finance_instruction_per_student_status
        self.assertEqual(statuses.to_dict(), {"001902001": "missing_enrollment", "001902002": "zero_denominator",
                                              "001902003": "observed"})
        self.assertTrue(out.loc[["001902001", "001902002"], "finance_instruction_per_student"].isna().all())

    def test_routing_follows_the_archive_that_holds_the_campus(self):
        out = self.load([row("001902", "199", 2024, "11", "6119", "001", 100)],
                        [["003801", "420", "2024", "11", "6119", "042", "30", "60", "04MAR2025"]],
                        {"001902001": 10, "003801042": 10})
        self.assertEqual(out.source_finance_type.to_dict(), {"001902001": "district", "003801042": "charter"})
        self.assertEqual(out.at["003801042", "finance_instruction_gf_expenditure"], 60)

    def test_campus_in_both_archives_fails(self):
        with self.assertRaisesRegex(ValueError, "in both"):
            self.load([row("001902", "199", 2024, "11", "6119", "001", 100)],
                      [["001902", "420", "2024", "11", "6119", "001", "30", "60", "04MAR2025"]], {"001902001": 10})

    def test_central_unit_never_becomes_a_campus(self):
        out = self.load([row("001902", "199", 2024, "11", "6119", "000", 999),
                         row("001902", "199", 2024, "11", "6119", "001", 100)], [], {"001902001": 10})
        self.assertEqual(out.index.tolist(), ["001902001"])


class AvailabilityPolicyTests(unittest.TestCase):
    def test_finance_lag_is_derived_from_publication_dates_not_assumed(self):
        finance = yaml.safe_load((ROOT / "configs/finance_availability.yaml").read_text())
        main = yaml.safe_load((ROOT / "configs/feature_availability.yaml").read_text())
        self.assertEqual(finance["cutoff_month_day"], main["cutoff_month_day"])
        bounds = finance["publication_bounds"]
        derived = None
        for lag in range(1, 6):
            # lag is allowed only if every target year with that source year observable has it released by the cutoff
            if all(bounds[t - lag] <= f"{t - 1}-{finance['cutoff_month_day']}" for t in range(2020, 2025) if t - lag in bounds):
                derived = lag
                break
        self.assertEqual(finance["minimum_lag"], derived)
        self.assertEqual(derived, 2)

    def test_bounds_cover_the_archives_own_update_dates(self):
        finance = yaml.safe_load((ROOT / "configs/finance_availability.yaml").read_text())
        src = yaml.safe_load((ROOT / "configs/data_sources.yaml").read_text())["finance_peims"]
        for year, entry in src.items():
            path = ROOT / "data/raw" / entry["file"]
            if not path.exists():
                self.skipTest("raw finance archives not present")
            with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as stream:
                updated = pd.read_csv(stream, dtype=str, usecols=["DTUPDATE"], nrows=2000).DTUPDATE.iloc[0]
            published = pd.to_datetime(updated, format="%d%b%Y")
            self.assertGreaterEqual(pd.Timestamp(finance["publication_bounds"][year]), published)
            self.assertLess(pd.Timestamp(finance["publication_bounds"][year]) - published, pd.Timedelta(days=31))

    def test_finance_is_not_yet_a_model_input(self):
        self.assertNotIn("finance", ns.FEATURE_GROUPS)
        self.assertFalse(set(ns.FINANCE_FIELDS) & set(ns.FEATURES))


if __name__ == "__main__":
    unittest.main()
