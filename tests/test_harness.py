"""Tests for the neural training harness loop (src/train/trainer.py): labels, masks, multi-task, epoch choice."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/train"))
from trainer import fit  # noqa: E402

T, N, K = 4, 30, 2          # years (index 0 has no label), schools, targets


class Zero(nn.Module):
    """Always outputs 0: after de-standardisation that is the training-label mean."""
    def __init__(self, outputs=K):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))          # Adam needs a parameter
        self.outputs = outputs

    def forward(self, x):
        return x.new_zeros(x.shape[0], x.shape[1], self.outputs) + 0 * self.bias


class Linear(nn.Module):
    def __init__(self, features, outputs):
        super().__init__()
        self.lin = nn.Linear(features, outputs)

    def forward(self, x):
        return self.lin(x)


def panel(seed=0):
    g = torch.Generator().manual_seed(seed)
    lag = torch.randn(T, N, K, generator=g) * 3 + 50
    y = lag + 1.0 + 0.1 * torch.randn(T, N, K, generator=g)          # every school rises by ~1 per year
    x = torch.cat([lag, torch.ones(T, N, 1)], -1)
    eval_mask = torch.ones(T, N, K, dtype=torch.bool)
    return x, y, lag, eval_mask


class HarnessTests(unittest.TestCase):
    def test_change_label_zero_output_is_persistence_plus_mean_change(self):
        x, y, lag, m = panel()
        f = fit(lambda: Zero(), x, y, lag, m, stop=T - 1, epochs=1, lr=0.0, seed=0, label="change")
        mean_change = (y[1:T - 1] - lag[1:T - 1]).mean(dim=(0, 1)).numpy()
        np.testing.assert_allclose(f.forecast, (lag[T - 1].numpy() + mean_change), rtol=1e-5)

    def test_level_label_zero_output_is_training_mean(self):
        x, y, lag, m = panel()
        f = fit(lambda: Zero(), x, y, lag, m, stop=T - 1, epochs=1, lr=0.0, seed=0, label="level")
        np.testing.assert_allclose(f.forecast, np.broadcast_to(y[1:T - 1].mean(dim=(0, 1)).numpy(), (N, K)), rtol=1e-5)

    def test_missing_labels_do_not_enter_the_loss_or_scaling(self):
        x, y, lag, m = panel()
        y_nan = y.clone()
        y_nan[1, :5, 0] = float("nan")
        f = fit(lambda: Zero(), x, y_nan, lag, m, stop=T - 1, epochs=2, lr=0.01, seed=0, label="change")
        self.assertTrue(np.isfinite(f.forecast).all())
        self.assertTrue(torch.isfinite(f.center).all() and torch.isfinite(f.scale).all())

    def test_multitask_scores_every_target_and_picks_an_epoch(self):
        x, y, lag, m = panel()
        f = fit(lambda: Linear(x.shape[-1], K), x, y, lag, m, stop=T - 1, epochs=20, lr=0.05, seed=0,
                label="change", tune=True)
        self.assertEqual(len(f.val_mae), K)
        self.assertTrue(1 <= f.best_epoch <= 20)
        self.assertTrue(np.isfinite(f.val_ratio))
        self.assertEqual(f.forecast.shape, (N, K))

    def test_same_seed_same_result(self):
        x, y, lag, m = panel()
        a = fit(lambda: Linear(x.shape[-1], K), x, y, lag, m, stop=T - 1, epochs=5, lr=0.05, seed=3, label="change")
        b = fit(lambda: Linear(x.shape[-1], K), x, y, lag, m, stop=T - 1, epochs=5, lr=0.05, seed=3, label="change")
        np.testing.assert_array_equal(a.forecast, b.forecast)

    def test_forecast_year_label_never_used_in_training(self):
        x, y, lag, m = panel()
        y2 = y.clone()
        y2[T - 1] = y2[T - 1] + 1000.0                   # change only the forecast year's outcome
        a = fit(lambda: Linear(x.shape[-1], K), x, y, lag, m, stop=T - 1, epochs=5, lr=0.05, seed=0, label="change")
        b = fit(lambda: Linear(x.shape[-1], K), x, y2, lag, m, stop=T - 1, epochs=5, lr=0.05, seed=0, label="change")
        np.testing.assert_array_equal(a.forecast, b.forecast)


if __name__ == "__main__":
    unittest.main()
