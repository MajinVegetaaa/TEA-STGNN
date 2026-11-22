"""Tests for the MTGNN adaptation (src/models/mtgnn.py): causality, learned graph, absent schools, dropout off when scoring."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/models"))
from mtgnn import MTGNN, GraphLearner, row_normalise  # noqa: E402
from phase5_cells import GraphWindow  # noqa: E402

T, N, F, H = 5, 12, 4, 8


def window(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, N, F, generator=g)
    supports = []
    for _ in range(T):
        a = (torch.rand(N, N, generator=g) < 0.3).float()
        a = a + a.T + torch.eye(N)
        supports.append((a / a.sum(1, keepdim=True)).to_sparse_csr())
    return GraphWindow(x, supports)


def model(dropout=0.0):
    torch.manual_seed(0)
    return MTGNN(F, H, 2, N, dropout=dropout, embedding=6, top_k=4)


class MTGNNTests(unittest.TestCase):
    def test_output_shape(self):
        self.assertEqual(tuple(model()(window()).shape), (T, N, 2))

    def test_causal_in_time(self):
        """Changing later years' inputs or graphs never changes earlier forecasts."""
        m, w = model().eval(), window()
        later = window(seed=1)
        mixed = GraphWindow(torch.cat([w.x[:3], later.x[3:]]), w.supports[:3] + later.supports[3:])
        with torch.no_grad():
            torch.manual_seed(5); a = m(w)
            torch.manual_seed(5); b = m(mixed)
        torch.testing.assert_close(a[:3], b[:3])
        self.assertFalse(torch.allclose(a[3:], b[3:]))

    def test_learned_graph_is_sparse_non_negative_and_trainable(self):
        torch.manual_seed(0)
        g = GraphLearner(N, dim=6, k=4)
        adj = g()
        self.assertTrue((adj >= 0).all())
        self.assertTrue(((adj > 0).sum(1) <= 4).all())
        adj.sum().backward()
        self.assertGreater(g.emb1.weight.grad.abs().sum().item(), 0)
        rows = row_normalise(adj.detach()).sum(1)
        torch.testing.assert_close(rows, torch.ones(N))

    def test_absent_schools_do_not_break_the_forward_pass(self):
        w = window()
        w.x[:, :3] = 0.0                                   # three schools with no inputs in any year
        out = model()(w)
        self.assertTrue(torch.isfinite(out).all())

    def test_dropout_is_off_when_scoring(self):
        m, w = model(dropout=0.5), window()
        m.eval()
        with torch.no_grad():
            torch.manual_seed(1); a = m(w)
            torch.manual_seed(2); b = m(w)                  # different dropout masks would show up here
        torch.testing.assert_close(a, b)

    def test_gradients_reach_every_part(self):
        m, w = model(), window()
        m(w).sum().backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)


if __name__ == "__main__":
    unittest.main()
