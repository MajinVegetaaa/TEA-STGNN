"""Synthetic checks for the S6 lock/run/report code; no school data is read."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/s6"), str(ROOT / "src/train"), str(ROOT / "src/models"), str(ROOT / "src/eval")]
import report  # noqa: E402
import run as s6run  # noqa: E402
import run_step4_rolling as rolling  # noqa: E402
from s6_data import content_hash  # noqa: E402


def snapshot(ids_present, all_ids, seed):
    g = torch.Generator().manual_seed(seed)
    n = len(all_ids)
    data = HeteroData()
    present = torch.tensor([i in ids_present for i in all_ids])
    data["campus"].present = present
    data["campus"].x = torch.where(present[:, None], torch.randn(n, 3, generator=g), torch.zeros(n, 3))
    data["campus"].y = torch.randn(n, 2, generator=g)
    data["campus"].y_lag1 = torch.randn(n, 2, generator=g)
    data["campus"].eval_mask = present[:, None].repeat(1, 2)
    pos = {c: i for i, c in enumerate(all_ids)}
    s, t = torch.tensor([pos["a"], pos["b"]]), torch.tensor([pos["b"], pos["c"]])
    store = data["campus", "district", "campus"]
    store.edge_index = torch.stack([torch.cat([s, t]), torch.cat([t, s])])
    store.edge_weight = torch.tensor([1.0, 2.0, 1.0, 2.0])
    store.edge_attr = torch.tensor([[1.0, 0.1], [2.0, 0.2], [1.0, 0.1], [2.0, 0.2]])
    return data


def permute(data, old_ids, new_ids):
    """The same snapshot with campuses at different row positions (as when a 2024-only campus is added)."""
    new_pos = {c: i for i, c in enumerate(new_ids)}
    order = torch.tensor([new_pos[c] for c in old_ids])
    n = len(new_ids)
    out = HeteroData()
    for name in ("present", "x", "y", "y_lag1", "eval_mask"):
        old = getattr(data["campus"], name)
        fill = torch.zeros((n,) + old.shape[1:], dtype=old.dtype)
        fill[order] = old
        setattr(out["campus"], name, fill)
    store, old = out["campus", "district", "campus"], data["campus", "district", "campus"]
    store.edge_index = order[old.edge_index]
    store.edge_weight, store.edge_attr = old.edge_weight.clone(), old.edge_attr.clone()
    return out


class S6Tests(unittest.TestCase):
    def test_content_hash_ignores_row_order_and_later_only_campuses_but_sees_value_changes(self):
        ids = ["a", "b", "c"]
        snaps = [snapshot({"a", "b", "c"}, ids, k) for k in range(2)]
        base = SimpleNamespace(years=[2022, 2023], snapshots=snaps, campus_ids=ids, feature_names=["f"] * 3,
                               targets=["t1", "t2"])
        ids2 = ["a", "aa", "b", "c"]                       # "aa" appears only in 2024
        moved = SimpleNamespace(**{**vars(base), "campus_ids": ids2,
                                   "snapshots": [permute(s, ids, ids2) for s in snaps]})
        self.assertEqual(content_hash(base, [2022, 2023]), content_hash(moved, [2022, 2023]))
        moved.snapshots[1]["campus"].y[0, 0] += 0.5
        self.assertNotEqual(content_hash(base, [2022, 2023]), content_hash(moved, [2022, 2023]))

    def test_bootstrap_sign_and_macro_interval(self):
        rng = np.random.default_rng(0)
        aligned = {}
        for t in ("t1", "t2"):
            y = rng.normal(size=500)
            aligned[t] = {"y": y, "pers": 1.0, "good": y + rng.normal(scale=0.2, size=500),
                          "bad": y + rng.normal(scale=1.0, size=500)}
        per_target, macro = report.bootstrap(aligned, "good", "bad", 500, 0)
        self.assertTrue((per_target["diff"] < 0).all())
        self.assertLess(macro["ci_high"], 0)
        self.assertEqual(macro["share_a_not_better"], 0.0)

    def test_finished_parts_refuse_to_run_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            s6run._refuse_if_done(out)                     # nothing there yet: allowed
            (out / "status.json").write_text(json.dumps({"status": "completed"}))
            with self.assertRaises(SystemExit):
                s6run._refuse_if_done(out)

    def test_mtgnn_keeps_its_designed_label_and_anchor(self):
        with self.assertRaises(ValueError):
            rolling.run("mtgnn", "cpu", [2023], "change", panel=SimpleNamespace(), scaler={}, use_anchor=True)
        with self.assertRaises(ValueError):
            rolling.run("mtgnn", "cpu", [2023], "change_relative", panel=SimpleNamespace(), scaler={},
                        use_anchor=False)


if __name__ == "__main__":
    unittest.main()
