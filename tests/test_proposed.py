"""Synthetic unit checks for the proposed S5 model variants."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/models"))
from proposed import ProposedSTGNN, ProposedWindow  # noqa: E402


def make_window(years=4, nodes=7, features=5, relations=2):
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(years, nodes, features, generator=gen)
    edges, attrs = [], []
    for _ in range(years):
        e_year, a_year = [], []
        for _ in range(relations):
            edge = torch.randint(nodes, (2, 14), generator=gen)
            e_year.append(edge)
            a_year.append(torch.randn(edge.shape[1], 2, generator=gen))
        edges.append(e_year)
        attrs.append(a_year)
    return ProposedWindow(x, edges, attrs)


class ProposedTests(unittest.TestCase):
    def test_v0_and_v1_shapes(self):
        w = make_window()
        self.assertEqual(tuple(ProposedSTGNN(5, 8, 1, version=0)(w).shape), (4, 7, 1))
        self.assertEqual(tuple(ProposedSTGNN(5, 8, 1, version=1, relations=2)(w).shape), (4, 7, 1))

    def test_graph_attention_is_causal_and_trainable(self):
        torch.manual_seed(3)
        model, w = ProposedSTGNN(5, 8, 1, version=1, relations=2).eval(), make_window()
        changed = make_window()
        changed.x[3] += 2
        with torch.no_grad():
            a, b = model(w), model(changed)
        torch.testing.assert_close(a[:3], b[:3])
        self.assertFalse(torch.allclose(a[3:], b[3:]))
        model.train()(w).sum().backward()
        self.assertGreater(model.relation_score.grad.abs().sum().item(), 0)

    def test_gated_versions_start_at_mean_change(self):
        w = make_window()
        model = ProposedSTGNN(5, 8, 3, version=3, relations=2)
        out = model(w)
        torch.testing.assert_close(out, torch.zeros_like(out))
        self.assertEqual(tuple(model.alpha.shape), (3,))

    def test_multitask_has_independent_heads_and_gradients(self):
        model, w = ProposedSTGNN(5, 8, 3, version=1, relations=2), make_window()
        self.assertEqual(len(model.heads), 3)
        self.assertEqual(len({head.weight.data_ptr() for head in model.heads}), 3)
        model(w).square().sum().backward()
        for i, head in enumerate(model.heads):
            self.assertIsNotNone(head.weight.grad, f"target head {i}")
            self.assertGreater(head.weight.grad.abs().sum().item(), 0, f"target head {i}")

    def test_empty_relations_are_safe_and_keep_self_information(self):
        w = make_window()
        empty_edges = [torch.empty(2, 0, dtype=torch.long) for _ in range(2)]
        empty_attrs = [torch.empty(0, 2) for _ in range(2)]
        w.edges = [empty_edges.copy() for _ in range(w.x.shape[0])]
        w.edge_attrs = [empty_attrs.copy() for _ in range(w.x.shape[0])]
        model = ProposedSTGNN(5, 8, 1, version=1, relations=2).eval()
        changed = ProposedWindow(w.x.clone(), w.edges, w.edge_attrs)
        changed.x[:, 0] += 1
        with torch.no_grad():
            a, b = model(w), model(changed)
        self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(torch.allclose(a[:, 0], b[:, 0]))

    def test_dropout_is_off_when_scoring(self):
        model, w = ProposedSTGNN(5, 8, 1, dropout=0.5, version=1, relations=2).eval(), make_window()
        with torch.no_grad():
            a, b = model(w), model(w)
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
