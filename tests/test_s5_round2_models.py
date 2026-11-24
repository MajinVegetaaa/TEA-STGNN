"""Synthetic architecture, masking, gradient, normalization, and resume checks."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/models"), str(ROOT / "src/train")]
from proposed import ProposedSTGNN, ProposedWindow  # noqa: E402
from proposed_round2 import PositiveGateSTGNN, QueryAwareRelationSTGNN  # noqa: E402
from run_s5_round2 import (_atomic_torch_save, _masked_standardized_loss, fit_trial,
                           seed_everything, validate_config)  # noqa: E402


def window(seed=0, years=4, nodes=6, features=5, relations=2):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(years, nodes, features, generator=gen)
    edges, attrs = [], []
    edge = torch.tensor([[0, 2, 0, 1], [1, 1, 4, 5]], dtype=torch.long)
    for _ in range(years):
        edges.append([edge.clone() for _ in range(relations)])
        attrs.append([torch.randn(edge.shape[1], 2, generator=gen) for _ in range(relations)])
    return ProposedWindow(x, edges, attrs)


class S5Round2ModelTests(unittest.TestCase):
    def test_v0_remains_temporal_only(self):
        model = ProposedSTGNN(5, 8, 1, version=0)
        self.assertFalse(hasattr(model, "relation_gat"))
        self.assertFalse(hasattr(model, "alpha"))
        self.assertEqual(tuple(model(window()).shape), (4, 6, 1))

    def test_query_attention_masks_and_normalizes_available_relations(self):
        w = window()
        model = QueryAwareRelationSTGNN(5, 8, 1, relations=2)
        _, weights = model.forward_with_attention(w)
        self.assertEqual(tuple(weights.shape), (4, 2, 6))
        torch.testing.assert_close(weights[:, :, 1].sum(dim=1), torch.ones(4))
        torch.testing.assert_close(weights[:, :, 0], torch.zeros(4, 2))
        torch.testing.assert_close(weights[:, :, 3], torch.zeros(4, 2))
        torch.testing.assert_close(weights[:, :, 4].sum(dim=1), torch.ones(4))

    def test_query_attention_preserves_isolated_node_and_has_gradients(self):
        w = window()
        model = QueryAwareRelationSTGNN(5, 8, 1, relations=2).eval()
        changed = ProposedWindow(w.x.clone(), w.edges, w.edge_attrs)
        changed.x[:, 3] += 1
        with torch.no_grad():
            a, b = model(w), model(changed)
        self.assertFalse(torch.allclose(a[:, 3], b[:, 3]))
        model.train()(w).square().sum().backward()
        for name in ("query_proj.weight", "message_proj.weight", "relation_vector"):
            self.assertGreater(dict(model.named_parameters())[name].grad.abs().sum().item(), 0, name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.relation_gat.parameters()))

    def test_v2_positive_gate_flows_gradient_on_first_backward(self):
        torch.manual_seed(11)
        model = PositiveGateSTGNN(5, 8, 2, relations=2)
        output = model(window()).square().mean()
        output.backward()
        self.assertTrue((model.gate > 0).all())
        self.assertTrue(torch.allclose(model.gate, torch.full((2,), 0.1), atol=1e-6))
        self.assertGreater(model.gate_logit.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.input.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.gru.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.heads[0].weight.grad.abs().sum().item(), 0)

    def test_original_zero_gate_stays_available_as_diagnostic_control(self):
        model = ProposedSTGNN(5, 8, 1, version=2, relations=2)
        model(window()).sum().backward()
        self.assertEqual(float(model.alpha[0].detach()), 0.0)
        self.assertEqual(float(model.input.weight.grad.abs().sum()), 0.0)

    def test_v3_has_separate_heads_and_equal_target_loss(self):
        model = ProposedSTGNN(5, 8, 10, version=1, relations=4)
        self.assertEqual(len(model.heads), 10)
        self.assertEqual(len({head.weight.data_ptr() for head in model.heads}), 10)
        pred = torch.zeros((1, 12, 2), requires_grad=True)
        scaled = torch.zeros_like(pred)
        scaled[0, :, 0] = torch.arange(1, 13)
        scaled[0, :2, 1] = torch.tensor([3.0, 4.0])
        observed = torch.zeros_like(pred, dtype=torch.bool)
        observed[0, :, 0] = True
        observed[0, :2, 1] = True
        loss = _masked_standardized_loss(pred, scaled, observed)
        expected_equal = (torch.arange(1, 13, dtype=torch.float32).square().mean() + 12.5) / 2
        torch.testing.assert_close(loss, expected_equal)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_seed_setup_is_repeatable(self):
        seed_everything(19)
        first = torch.randn(4)
        seed_everything(19)
        second = torch.randn(4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_round_trip_and_interrupted_trial_resume(self):
        torch.manual_seed(3)
        reference = torch.nn.Linear(3, 2)
        x0 = torch.randn(2, 3)
        expected = reference(x0).detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "linear.pt"
            _atomic_torch_save(checkpoint, {"state_dict": reference.state_dict()})
            restored = torch.nn.Linear(3, 2)
            restored.load_state_dict(torch.load(checkpoint, weights_only=False)["state_dict"])
            torch.testing.assert_close(restored(x0), expected)

        # Small synthetic in-memory panel: interrupt after a checkpointed epoch,
        # then resume to completion. No repository dataset or model experiment is used.
        torch.manual_seed(22)
        t, n = 4, 12
        lag = torch.randn(t, n, 1) + 50
        y = lag + torch.randn(t, n, 1) * 0.5 + 1
        x = torch.randn(t, n, 3)
        mask = torch.ones(t, n, 1, dtype=torch.bool)
        fake_snapshots = [{"campus": {"x": x[k], "y": y[k], "y_lag1": lag[k],
                                       "eval_mask": mask[k]}} for k in range(t)]
        panel = SimpleNamespace(val_index=3, campus_ids=[str(i) for i in range(n)], snapshots=fake_snapshots)
        import run_s5_round2 as runner
        runner.stop_year, runner.campus_ids = 2023, panel.campus_ids
        def interrupt(_item):
            raise RuntimeError("synthetic interruption")

        with tempfile.TemporaryDirectory() as tmp:
            resume_path = Path(tmp) / "resume.pt"
            graphless_window = ProposedWindow(x, [[] for _ in range(t)], [[] for _ in range(t)])
            args = dict(variant="v0_temporal", x=graphless_window, y=y, lag=lag, eval_mask=mask,
                        target_indices=[0], target_names=["target"], panel=panel, relations=[],
                        hidden=4, dropout=0.0, epochs=2, lr=0.003, weight_decay=0.0001,
                        patience=2, seed=2, device="cpu", tune=True, gradient_clip_norm=None,
                        config_hash="synthetic-config", checkpoint_path=resume_path)
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                fit_trial(**args, on_epoch=interrupt)
            self.assertTrue(resume_path.is_file())
            resumed = fit_trial(**args)
            self.assertIn(resumed["best_epoch"], (1, 2))
            self.assertTrue(torch.isfinite(torch.tensor(resumed["val_ratio"])))

    def test_round2_config_guards(self):
        base = {"variant": "v1_query", "through_year": 2023, "validation_only": True,
                "graph": "v2", "label": "change", "out": "s5_round2/query",
                "tasks": "single", "relation_policy": "target_specific"}
        validate_config(base)
        with self.assertRaisesRegex(ValueError, "capped at 2023"):
            validate_config({**base, "through_year": 2024})
        with self.assertRaisesRegex(ValueError, "below results/s5_round2"):
            validate_config({**base, "out": "s5/query"})


if __name__ == "__main__":
    unittest.main()
