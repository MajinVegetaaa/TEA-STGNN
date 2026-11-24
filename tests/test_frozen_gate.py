"""The master-data gate must fingerprint its frozen comparison inputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/data"))
from validate_additions import FROZEN, gate_files  # noqa: E402


class FrozenGateTests(unittest.TestCase):
    def test_gate_requires_the_frozen_manifest_and_comparison_files(self):
        files = set(gate_files())
        required = {
            FROZEN / "manifest.json",
            FROZEN / "data/processed/master_school_year.csv",
            FROZEN / "data/processed/temporal_features.csv",
            FROZEN / "data/processed/school_metadata.csv",
            FROZEN / "data/processed/risk_index_components.csv",
            FROZEN / "data/processed/subgroup_school_year.csv",
        }
        self.assertTrue(required.issubset(files))


if __name__ == "__main__":
    unittest.main()
