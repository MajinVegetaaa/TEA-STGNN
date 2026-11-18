"""Detect accidental use of later campus observations in earlier forecasts."""
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Phase 5 model tests require PyTorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/models"))
from phase5_cells import GConvGRU, GraphWaveNet, GraphWindow, TGCN


class CausalSequenceTest(unittest.TestCase):
    def test_prefix_forecasts_and_gradients(self):
        torch.manual_seed(12)
        x = torch.randn(5, 4, 6)
        graph = torch.eye(4).roll(1, dims=1).to_sparse_csr()
        for name, model in [('tgcn', TGCN(6, 3)), ('a3tgcn', TGCN(6, 3, True)),
                            ('gconvgru', GConvGRU(6, 3)), ('graphwavenet', GraphWaveNet(6, 3))]:
            with self.subTest(name=name):
                features = x
                if name == 'gconvgru':
                    features = torch.cat((x, torch.stack([-(graph @ frame) for frame in x])), -1)
                if name in ('gconvgru', 'graphwavenet'):
                    features = GraphWindow(features, [graph] * len(x))
                full = model(features)
                torch.testing.assert_close(full[:3], model(features[:3]))
                full.square().mean().backward()
                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())
