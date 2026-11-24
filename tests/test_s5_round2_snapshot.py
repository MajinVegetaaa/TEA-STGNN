"""Synthetic validation of the 2023-capped, year-sharded Graph V2 snapshot path."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/graphs"), str(ROOT / "src/train")]
import build_s5_validation_snapshot as builder  # noqa: E402
from data_s5_round2 import load_validation_bundle  # noqa: E402


def write_synthetic_graph(source: Path) -> bytes:
    source.mkdir(parents=True, exist_ok=True)
    (source / "pyg").mkdir()
    original_blob = b"original monolithic all-years snapshot sentinel; must remain unopened"
    (source / "pyg/snapshots.pt").write_bytes(original_blob)
    targets = list(builder.converter.TARGETS)
    edge_sets = ["excl_sat", "excl_act", "excl_criterion", "excl_participation",
                 "excl_attendance", "excl_dropout", "excl_retention"]
    map_rows = [{"target": target, "edge_set": edge_sets[i % len(edge_sets)],
                 "excluded_features": "synthetic", "profile_features": "synthetic"}
                for i, target in enumerate(targets)]
    with (source / "academic_edge_sets.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(map_rows[0]))
        writer.writeheader(); writer.writerows(map_rows)
    cat = [{"column": "history_feature", "role": "input feature (temporal)"},
           {"column": "covid_affected_year", "role": "input feature"}]
    cat += [{"column": target, "role": "same-year observation (target)"} for target in targets]
    with (source / "node_feature_catalog.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["column", "role"])
        writer.writeheader(); writer.writerows(cat)
    (source / "graph_v2_parameters.json").write_text('{"train_years":[2018,2019,2020,2021,2022]}\n')

    fields = ["campus_id", "year", "node_index", "history_feature", "covid_affected_year",
              "is_charter", "is_magnet", "is_virtual_or_hybrid", "is_residential_facility",
              "serves_full_9_12", "latitude", "longitude", "urbanicity", "esc_region"]
    fields += targets + [f"{target}_lag1" for target in targets]
    with (source / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for year in range(2018, 2024):
            for i, campus in enumerate(("000000001", "000000002")):
                row = {"campus_id": campus, "year": year, "node_index": i,
                       "history_feature": year - 2017 + i, "covid_affected_year": 0,
                       "is_charter": 0, "is_magnet": 0, "is_virtual_or_hybrid": 0,
                       "is_residential_facility": 0, "serves_full_9_12": 1,
                       "latitude": 30.0 + i, "longitude": -97.0, "urbanicity": "Town",
                       "esc_region": "01"}
                for k, target in enumerate(targets):
                    row[target] = 50 + year - 2018 + k + i
                    row[f"{target}_lag1"] = 49 + year - 2018 + k + i
                writer.writerow(row)
        # The future record is deliberately unconvertible as an input. A safe
        # exporter stops at its year field and never fits preprocessing on it.
        future = {key: 0 for key in fields}
        future.update({"campus_id": "000000001", "year": 2024,
                       "history_feature": "MUST_NOT_BE_CONVERTED", "urbanicity": "FutureOnly"})
        writer.writerow(future)

    edge_fields = ["src_campus_id", "dst_campus_id", "year", "weight", "distance", "edge_set"]
    for layer in builder.LAYERS:
        rows = []
        for year in range(2018, 2024):
            sets = edge_sets if layer == "academic" else [""]
            for edge_set in sets:
                rows.append({"src_campus_id": "000000001", "dst_campus_id": "000000002", "year": year,
                             "weight": 1.0, "distance": 2.0, "edge_set": edge_set})
        with (source / f"edges_{layer}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=edge_fields)
            writer.writeheader(); writer.writerows(rows)
    return original_blob


class S5Round2SnapshotTests(unittest.TestCase):
    def test_csv_cutoff_stops_before_future_row_is_converted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nodes.csv"
            path.write_text("campus_id,year,value\n1,2022,2\n1,2023,3\n1,2024,not-a-number\n",
                            encoding="utf-8")
            frame, _ = builder.read_sorted_year_csv(path, cutoff=2023)
            self.assertEqual(frame.year.tolist(), [2022, 2023])
            self.assertNotIn("not-a-number", frame.to_numpy().astype(str))

    def test_unsorted_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nodes.csv"
            path.write_text("campus_id,year,value\n1,2023,3\n1,2022,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not sorted by year"):
                builder.read_sorted_year_csv(path)

    def test_bundle_builds_without_torch_loading_original_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, output = base / "graph_v2", base / "s5_validation"
            original_blob = write_synthetic_graph(source)
            # Graph V2 intentionally has no academic mapping for the derived
            # risk_index target. The exporter must preserve that partial map.
            map_path = source / "academic_edge_sets.csv"
            with map_path.open("r", newline="", encoding="utf-8") as handle:
                map_rows = list(csv.DictReader(handle))
            map_rows = [row for row in map_rows if row["target"] != "risk_index"]
            map_rows.append({"target": "auxiliary_target", "edge_set": "excl_sat",
                             "excluded_features": "synthetic", "profile_features": "synthetic"})
            with map_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(map_rows[0]))
                writer.writeheader(); writer.writerows(map_rows)
            with patch.object(builder, "SOURCE", source), patch.object(builder, "OUTPUT", output), \
                 patch.object(builder.torch, "load", side_effect=AssertionError("must not deserialize snapshot")):
                manifest = builder.build_bundle(source=source, output=output)
            self.assertEqual(manifest["years"], [2018, 2019, 2020, 2021, 2022, 2023])
            self.assertEqual(manifest["cutoff_year"], 2023)
            self.assertFalse(manifest["test_years_included"])
            self.assertNotIn(2024, manifest["years"])
            self.assertEqual((source / "pyg/snapshots.pt").read_bytes(), original_blob)
            self.assertFalse((output / "snapshots.pt").exists())
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertNotIn("year_stats", metadata)
            self.assertNotIn("risk_index", metadata["academic_relation"])
            self.assertNotIn("auxiliary_target", metadata["academic_relation"])
            self.assertEqual(metadata["scaler"]["fit_years"], [2018, 2019, 2020, 2021, 2022])
            self.assertEqual(metadata["scaler"]["categories_fit_years"], [2018, 2019, 2020, 2021, 2022])
            self.assertEqual(metadata["edge_attr"], ["weight", "distance_z"])
            with patch.object(builder.torch, "load", wraps=torch.load) as load_spy:
                panel = load_validation_bundle(output)
            loaded_paths = [str(call.args[0]) for call in load_spy.call_args_list]
            self.assertEqual(len(loaded_paths), 6)
            self.assertTrue(all(Path(p).name in {f"year_{y}.pt" for y in range(2018, 2024)} for p in loaded_paths))
            self.assertEqual(panel.years[-1], 2023)
            self.assertEqual(panel.val_index, 5)
            self.assertEqual(len(panel.snapshots), 6)
            self.assertTrue(all(int(s.year) <= 2023 for s in panel.snapshots))
            relation = panel.snapshots[-1]["campus", "district", "campus"]
            self.assertEqual(relation.edge_index.shape[1], 2)
            self.assertEqual(tuple(relation.edge_attr.shape), (2, 2))
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                builder.build_bundle(source=source, output=output)

    def test_loader_rejects_future_year_before_loading_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text("{}")
            (root / "manifest.json").write_text(json.dumps({"format_version": 1, "graph": "v2",
                "cutoff_year": 2023, "years": [2018, 2023, 2024], "test_years_included": False,
                "shards": {f"year_{y}.pt": "x" for y in (2018, 2023, 2024)}}))
            with patch("data_s5_round2.torch.load", side_effect=AssertionError("must reject before load")):
                with self.assertRaisesRegex(ValueError, "post-cutoff"):
                    load_validation_bundle(root)


if __name__ == "__main__":
    unittest.main()
