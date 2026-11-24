"""Train one model on one graph for one or more targets (full batch, all years at once).

Labels (configs/protocol.yaml, D6):
  level   the raw outcome, standardised on the training labels (Phase 5)
  change  this year minus last year, standardised; forecast = last year + predicted change
With one target this reproduces the Phase 5 training loop exactly (same seeding order, loss and epoch choice).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Fit:
    best_epoch: int
    val_mae: list[float]          # per target, at the best epoch (NaN when not tuning)
    val_ratio: float              # mean over targets of val MAE / persistence MAE at the best epoch
    forecast: np.ndarray          # [schools, targets] forecast for year index `stop`, raw units
    model: torch.nn.Module
    center: torch.Tensor
    scale: torch.Tensor


def fit(build, x, y, y_lag1, eval_mask, stop, epochs, lr, seed, label="level", tune=False, device="cpu",
        weight_decay=0.0, patience=None, equal_target_loss=False) -> Fit:
    """Train on label years 1..stop-1 and forecast year index `stop`.

    tune=True scores every epoch on year `stop` (the validation year) and returns the best epoch; the model returned
    is the one after the final epoch, so callers refit with best_epoch epochs for a forecast. With `patience`, tuning
    stops after that many epochs without a better 2023 score. Dropout is on while training and off when forecasting.
    """
    torch.manual_seed(seed)
    model = build().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raw = y[1:stop]
    labels = raw - y_lag1[1:stop] if label == "change" else raw
    observed = labels.isfinite()
    center = torch.stack([labels[..., k][observed[..., k]].mean() for k in range(labels.shape[-1])])
    scale = torch.stack([labels[..., k][observed[..., k]].std() for k in range(labels.shape[-1])])
    scaled = (labels - center) / scale
    rows = eval_mask[stop]
    truth = y[stop]
    persistence = torch.stack([(y_lag1[stop, rows[:, k], k] - truth[rows[:, k], k]).abs().mean()
                               for k in range(truth.shape[-1])])

    def forecast():
        model.eval()
        out = model(x[:stop + 1])[-1] * scale + center
        return out + y_lag1[stop] if label == "change" else out

    best = (float("inf"), 0, [float("nan")] * truth.shape[-1])
    for epoch in range(1, epochs + 1):
        model.train()
        pred = model(x[:stop])[1:]
        if equal_target_loss:
            loss = torch.stack([(pred[..., k][observed[..., k]] - scaled[..., k][observed[..., k]]).square().mean()
                                for k in range(labels.shape[-1])]).mean()
        else:
            loss = (pred[observed] - scaled[observed]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if tune:
            with torch.no_grad():
                f = forecast()
                errors = [(f[rows[:, k], k] - truth[rows[:, k], k]).abs().mean().item() for k in range(truth.shape[-1])]
                ratio = float(np.mean([e / p.item() for e, p in zip(errors, persistence)]))
                if (ratio, epoch) < best[:2]:
                    best = (ratio, epoch, errors)
                if patience and epoch - best[1] >= patience:
                    break
    with torch.no_grad():
        final = forecast().cpu().numpy()
    ratio, epoch, errors = best if tune else (float("nan"), epochs, [float("nan")] * truth.shape[-1])
    return Fit(best_epoch=epoch, val_mae=errors, val_ratio=ratio, forecast=final, model=model,
               center=center, scale=scale)
