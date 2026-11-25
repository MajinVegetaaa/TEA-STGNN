"""Resumable 2023-only S5 Round 2 runner. This CLI has no test-year mode."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT / "src" / "models"))
from data_s5_round2 import load_validation_bundle  # noqa: E402
from proposed import ProposedSTGNN, ProposedWindow, prepare_proposed  # noqa: E402
from proposed_round2 import PositiveGateSTGNN, QueryAwareRelationSTGNN, prediction_parameter_grad_norm  # noqa: E402
from mtgnn import MTGNN, yearly_supports  # noqa: E402

COMMON_RELATIONS = ["district", "geographic", "demographic", "region"]
VALID_VARIANTS = {"v0_temporal", "v1_original", "v1_query", "v2_original", "v2_positive",
                  "v3_single_shared", "v3_multi_shared", "v3_planned", "mtgnn"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **event}, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_csv(path: Path, rows: list[dict], key_columns: list[str]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=key_columns, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    new.to_csv(temp, index=False)
    os.replace(temp, path)


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Inputs:
    model_input: object
    y: torch.Tensor
    y_lag1: torch.Tensor
    eval_mask: torch.Tensor


class GraphWindow:
    def __init__(self, x, supports):
        self.x, self.supports = x, supports

    def __getitem__(self, item):
        return GraphWindow(self.x[item], self.supports[item] if isinstance(self.supports, list) else self.supports)

    def to(self, device):
        return GraphWindow(self.x.to(device), [s.to(device) for s in self.supports])


TARGET_MODES = {"change", "change_relative"}
EXTRA_INPUTS = {"dist_2019"}


def statewide_mean_change(change: torch.Tensor) -> torch.Tensor:
    """Per year and target, the mean over schools of the observed change: [years, schools, k] -> [years, 1, k]."""
    observed = torch.isfinite(change)
    total = torch.where(observed, change, torch.zeros_like(change)).sum(dim=1, keepdim=True)
    return total / observed.sum(dim=1, keepdim=True).clamp(min=1)


def training_labels(local_y: torch.Tensor, local_lag: torch.Tensor, stop: int, target_mode: str) -> torch.Tensor:
    """Training labels for target years 1..stop-1.

    change           the school's change from last year (Round 2 default).
    change_relative  that change minus the year's statewide mean change, so a common shock (the COVID swings)
                     is not learned; forecasts still add `center` (~0 here), i.e. no statewide change is assumed.
    """
    labels = local_y[1:stop] - local_lag[1:stop]
    if target_mode == "change_relative":
        labels = labels - statewide_mean_change(labels)
    elif target_mode != "change":
        raise ValueError(f"unknown target_mode: {target_mode!r}")
    return labels


def add_dist_2019(panel) -> None:
    """Append, per target, last year's value minus the school's 2019 value to every snapshot's inputs.

    Known at forecast time for snapshots from 2020 on (both values are from earlier years); zero for the 2018
    and 2019 snapshots, where the 2019 value is not yet observed. Standardised with training-year statistics;
    a missing value becomes 0 after standardisation.
    """
    years = panel.years
    y2019 = panel.snapshots[years.index(2019)]["campus"].y
    raw = [s["campus"].y_lag1 - y2019 if year >= 2020 else torch.full_like(y2019, float("nan"))
           for s, year in zip(panel.snapshots, years)]
    train = torch.stack([raw[i] for i, year in enumerate(years) if year in panel.split["train_years"]])
    finite = torch.isfinite(train)
    count = finite.sum(dim=(0, 1)).clamp(min=2)
    mean = torch.where(finite, train, torch.zeros_like(train)).sum(dim=(0, 1)) / count
    var = torch.where(finite, (train - mean).square(), torch.zeros_like(train)).sum(dim=(0, 1)) / (count - 1)
    std = var.sqrt().clamp(min=1e-6)
    for snap, r in zip(panel.snapshots, raw):
        z = torch.nan_to_num((r - mean) / std, nan=0.0)
        snap["campus"].x = torch.cat([snap["campus"].x, z.to(snap["campus"].x.dtype)], dim=1)
    panel.feature_names = list(panel.feature_names) + [f"dist_2019_{t}" for t in panel.targets]


def validate_config(cfg: dict) -> None:
    variant = cfg.get("variant")
    if variant not in VALID_VARIANTS:
        raise ValueError(f"unknown S5 Round 2 variant: {variant!r}")
    if cfg.get("through_year") != 2023 or cfg.get("validation_only") is not True:
        raise ValueError("Round 2 configs must be validation-only and capped at 2023")
    if cfg.get("graph") != "v2" or cfg.get("label") != "change":
        raise ValueError("Round 2 requires Graph V2 and change-from-last-year labels")
    if not str(cfg.get("out", "")).startswith("s5_round2/"):
        raise ValueError("all Round 2 outputs must be below results/s5_round2/")
    if cfg.get("tasks") not in {"single", "multi"}:
        raise ValueError("tasks must be single or multi")
    if cfg.get("relation_policy") not in {"target_specific", "shared_four", "union_targets", "none"}:
        raise ValueError("invalid relation_policy")
    if cfg.get("variant") == "v3_multi_shared" and (cfg["tasks"] != "multi" or cfg["relation_policy"] != "shared_four"):
        raise ValueError("V3 multi-task treatment must use all targets and the shared four relations")
    if cfg.get("variant") == "v3_single_shared" and (cfg["tasks"] != "single" or cfg["relation_policy"] != "shared_four"):
        raise ValueError("V3 single-task control must use single targets and the shared four relations")
    if cfg.get("variant") == "v3_planned" and (cfg["tasks"] != "multi" or cfg["relation_policy"] != "union_targets"
                                               or cfg.get("loss") != "equal_targets"):
        raise ValueError("planned V3 must be the multi-task V2 with equal target losses over the union of "
                         "every target's relations")
    if cfg.get("target_mode", "change") not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {sorted(TARGET_MODES)}")
    extra = cfg.get("extra_inputs", [])
    if not set(extra) <= EXTRA_INPUTS or (extra and cfg.get("variant") == "mtgnn"):
        raise ValueError(f"extra_inputs must be a subset of {sorted(EXTRA_INPUTS)} and are not used with MTGNN")
    if cfg.get("variant") == "v0_temporal" and cfg.get("relation_policy") != "none":
        raise ValueError("V0 must remain temporal-only")


def _prepare_inputs(panel, relations: list[str], variant: str, device: str) -> object:
    if variant == "mtgnn":
        x = torch.stack([snap["campus"].x for snap in panel.snapshots])
        supports = yearly_supports(panel.snapshots, relations)
        return GraphWindow(x, supports).to(device)
    if variant == "v0_temporal":
        x = torch.stack([snap["campus"].x for snap in panel.snapshots]).to(device)
        empty_edges = [[] for _ in panel.snapshots]
        empty_attrs = [[] for _ in panel.snapshots]
        return ProposedWindow(x, empty_edges, empty_attrs)
    return prepare_proposed(panel.snapshots, relations).to(device)


def _build_model(variant: str, features: int, hidden: int, outputs: int, dropout: float,
                 schools: int, relations: int):
    if variant == "v0_temporal":
        return ProposedSTGNN(features, hidden, outputs, dropout=dropout, version=0)
    if variant in {"v1_original", "v3_single_shared", "v3_multi_shared"}:
        return ProposedSTGNN(features, hidden, outputs, dropout=dropout, version=1, relations=relations)
    if variant == "v1_query":
        return QueryAwareRelationSTGNN(features, hidden, outputs, dropout=dropout, relations=relations)
    if variant == "v2_original":
        return ProposedSTGNN(features, hidden, outputs, dropout=dropout, version=2, relations=relations)
    if variant == "v3_planned":
        # Roadmap v3: V2 (zero-initialised per-target gate) shared across all targets.
        return ProposedSTGNN(features, hidden, outputs, dropout=dropout, version=3, relations=relations)
    if variant == "v2_positive":
        return PositiveGateSTGNN(features, hidden, outputs, dropout=dropout, relations=relations,
                                 initial_scale=float(cfg_model_args.get("initial_gate_scale", 0.1)))
    if variant == "mtgnn":
        args = {"embedding": 40, "alpha": 3.0, "top_k": 20, "beta": 0.05, "depth": 2,
                "kernels": (1, 2, 3), "dilations": (1, 2)}
        args.update({k: v for k, v in cfg_model_args.items()})
        return MTGNN(features, hidden, outputs, schools, dropout=dropout, **args)
    raise ValueError(variant)


cfg_model_args: dict = {}


def _group_relations(panel, group: list[str], policy: str) -> list[str]:
    if policy == "none":
        return []
    if policy == "shared_four":
        return COMMON_RELATIONS.copy()
    if policy == "union_targets":
        # Same relation set as the Round 1 v3 (`all_target_relations` in run.py): the shared four plus
        # every target's own academic relation, de-duplicated in first-seen order.
        return list(dict.fromkeys(r for target in group for r in panel.relations(target)))
    if len(group) != 1:
        raise ValueError("target-specific relations are only defined for single-target fits")
    return panel.relations(group[0])


def _target_tensors(panel, device):
    return (torch.stack([s["campus"].y for s in panel.snapshots]).to(device),
            torch.stack([s["campus"].y_lag1 for s in panel.snapshots]).to(device),
            torch.stack([s["campus"].eval_mask for s in panel.snapshots]).to(device))


def _masked_standardized_loss(prediction: torch.Tensor, scaled: torch.Tensor,
                              observed: torch.Tensor) -> torch.Tensor:
    """Equal target MSE after each target's training-only standardization."""
    losses = []
    for k in range(scaled.shape[-1]):
        mask = observed[..., k]
        if not mask.any():
            raise ValueError(f"target {k} has no observed training labels")
        losses.append((prediction[..., k][mask] - scaled[..., k][mask]).square().mean())
    return torch.stack(losses).mean()


def _prediction_and_metrics(model, x, y, lag, eval_mask, targets, target_indices, center, scale,
                            stop: int, label: str, *, device: str, target_mode: str = "change"):
    model.eval()
    with torch.no_grad():
        normalized = model(x[:stop + 1])[-1]
        forecast_change = normalized * scale + center
        forecast = forecast_change + lag[stop, :, target_indices] if label == "change" else forecast_change
        truth = y[stop, :, target_indices]
        mask = eval_mask[stop, :, target_indices]
        rows = []
        errors, ratios, losses = [], [], []
        val_label = truth - lag[stop, :, target_indices] if label == "change" else truth
        if target_mode == "change_relative":      # validation loss only (logged, not used for selection)
            val_label = val_label - statewide_mean_change(val_label.unsqueeze(0))[0]
        val_observed = mask & torch.isfinite(val_label)
        val_scaled = (val_label - center) / scale
        valid_loss = (normalized - val_scaled)[val_observed].square()
        val_loss = float(valid_loss.mean().item()) if valid_loss.numel() else float("nan")
        for k, target in enumerate(targets):
            active = mask[:, k] & torch.isfinite(truth[:, k]) & torch.isfinite(lag[stop, :, target_indices[k]])
            if not active.any():
                raise ValueError(f"no validation rows for target {target}")
            mae = float((forecast[active, k] - truth[active, k]).abs().mean().item())
            persistence = float((lag[stop, active, target_indices[k]] - truth[active, k]).abs().mean().item())
            if not np.isfinite(persistence) or persistence <= 0:
                raise ValueError(f"invalid persistence MAE for target {target}")
            errors.append(mae)
            ratios.append(mae / persistence)
            losses.append(val_loss)
            for i in active.nonzero(as_tuple=False).flatten().tolist():
                rows.append({"target": target, "year": stop_year, "campus_id": campus_ids[i],
                             "y_true": float(truth[i, k].item()), "y_pred": float(forecast[i, k].item()),
                             "persistence_pred": float(lag[stop, i, target_indices[k]].item())})
    return rows, errors, ratios, val_loss


def _make_checkpoint(model, optimizer, epoch, best_score, best_epoch, best_state, wait,
                     history, seed, config_hash):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "best_score": best_score, "best_epoch": best_epoch, "best_state": best_state,
            "wait": wait, "history": history, "seed": seed, "config_hash": config_hash,
            "rng": _rng_state()}


def fit_trial(*, variant: str, x, y, lag, eval_mask, target_indices: list[int], target_names: list[str],
              panel, relations: list[str], hidden: int, dropout: float, epochs: int, lr: float,
              weight_decay: float, patience: int | None, seed: int, device: str, tune: bool,
              gradient_clip_norm: float | None, config_hash: str, checkpoint_path: Path | None = None,
              on_epoch=None, target_mode: str = "change"):
    """One fit on training years only; optionally select its epoch on 2023."""
    seed_everything(seed)
    stop = panel.val_index
    local_y = y[..., target_indices]
    local_lag = lag[..., target_indices]
    local_mask = eval_mask[..., target_indices]
    labels = training_labels(local_y, local_lag, stop, target_mode)
    observed = labels.isfinite()
    centers, scales = [], []
    for k, target in enumerate(target_names):
        vals = labels[..., k][observed[..., k]]
        if vals.numel() < 2:
            raise ValueError(f"insufficient training labels for {target}")
        center, scale = vals.mean(), vals.std()
        if not torch.isfinite(center) or not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid training-only change normalization for {target}")
        centers.append(center)
        scales.append(scale)
    center, scale = torch.stack(centers), torch.stack(scales)
    scaled = (labels - center) / scale
    rows = local_mask[stop]
    truth = local_y[stop]
    if variant == "mtgnn":
        model = _build_model(variant, x.x.shape[-1], hidden, len(target_indices), dropout,
                             len(panel.campus_ids), len(relations)).to(device)
    else:
        features = x.x.shape[-1] if isinstance(x, ProposedWindow) else x.shape[-1]
        model = _build_model(variant, features, hidden, len(target_indices), dropout,
                             len(panel.campus_ids), len(relations)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trial_config_hash = config_hash
    history = []
    best_score, best_epoch, best_state, wait = float("inf"), 0, None, 0
    start_epoch = 1
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("config_hash") != trial_config_hash or checkpoint.get("seed") != seed:
            raise ValueError("resume checkpoint does not match this trial")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_score, best_epoch, best_state = checkpoint["best_score"], checkpoint["best_epoch"], checkpoint["best_state"]
        wait, history = checkpoint["wait"], checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        _restore_rng(checkpoint["rng"])

    max_epochs = epochs
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        prediction = model(x[:stop])[1:]
        loss = _masked_standardized_loss(prediction, scaled, observed)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        prediction_grad_norm = prediction_parameter_grad_norm(model) if variant == "v2_positive" else float(
            torch.sqrt(sum((p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None),
                           torch.zeros((), device=device))).item())
        gate_grad_norm = (float(model.gate_logit.grad.detach().float().norm().item())
                          if variant == "v2_positive" and model.gate_logit.grad is not None else 0.0)
        clip_norm = gradient_clip_norm
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        optimizer.step()
        rows_out, val_mae, ratios, val_loss = _prediction_and_metrics(
            model, x, y, lag, eval_mask, target_names, target_indices, center, scale,
            stop, "change", device=device, target_mode=target_mode)
        score = float(np.mean(ratios))
        gates = model.gate.detach().cpu().tolist() if variant == "v2_positive" else []
        item = {"epoch": epoch, "train_loss": float(loss.detach().item()), "validation_loss": val_loss,
                "val_ratio": score, "prediction_grad_norm": prediction_grad_norm,
                "gate_grad_norm": gate_grad_norm, "gate_values": gates}
        history.append(item)
        improved = score < best_score - 1e-12
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if checkpoint_path is not None:
            _atomic_torch_save(checkpoint_path, _make_checkpoint(model, optimizer, epoch, best_score,
                                best_epoch, best_state, wait, history, seed, trial_config_hash))
        if on_epoch is not None:
            on_epoch(item)
        if tune and patience is not None and wait >= patience:
            break
    if tune:
        if best_state is None:
            raise RuntimeError("epoch selection failed to produce a finite validation score")
        model.load_state_dict(best_state)
        selected_epochs = best_epoch
    else:
        selected_epochs = max_epochs
        best_epoch = max_epochs
        model.eval()
        with torch.no_grad():
            normalized = model(x[:stop + 1])[-1]
            forecast = normalized * scale + center + local_lag[stop]
            target_mae, target_ratios = [], []
            for k in range(len(target_indices)):
                mask = rows[:, k]
                mae = float((forecast[mask, k] - truth[mask, k]).abs().mean().item())
                baseline = float((local_lag[stop, mask, k] - truth[mask, k]).abs().mean().item())
                target_mae.append(mae)
                target_ratios.append(mae / baseline)
            best_score = float(np.mean(target_ratios))
    model.eval()
    with torch.no_grad():
        normalized = model(x[:stop + 1])[-1]
        forecasts = normalized * scale + center + local_lag[stop]
        output_rows, maes, ratios, val_loss = _prediction_and_metrics(
            model, x, y, lag, eval_mask, target_names, target_indices, center, scale,
            stop, "change", device=device, target_mode=target_mode)
    return {"model": model, "center": center.detach().cpu(), "scale": scale.detach().cpu(),
            "forecast": forecasts.detach().cpu(), "prediction_rows": output_rows,
            "val_mae": maes, "val_ratios": ratios, "val_ratio": float(np.mean(ratios)),
            "val_loss": val_loss, "best_epoch": selected_epochs, "history": history,
            "prediction_grad_norm": history[-1]["prediction_grad_norm"] if history else 0.0,
            "gate_values": model.gate.detach().cpu().tolist() if variant == "v2_positive" else []}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _environment(device: str) -> dict:
    import platform
    import torch_geometric

    return {"python": platform.python_version(), "torch": torch.__version__,
            "pyg": torch_geometric.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda, "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else None,
            "git_commit": _git_commit()}


def run(config_path: Path, *, device: str | None = None, resume: bool = False) -> None:
    global cfg_model_args, stop_year, campus_ids
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_config(cfg)
    bundle_path = ROOT / cfg.get("snapshot_bundle", "data/graph_v2/pyg/s5_validation")
    panel = load_validation_bundle(bundle_path)
    if "dist_2019" in cfg.get("extra_inputs", []):
        add_dist_2019(panel)
    target_mode = cfg.get("target_mode", "change")
    budget = yaml.safe_load((ROOT / cfg["budget"]).read_text(encoding="utf-8"))
    cfg_model_args = cfg.get("model_args", {})
    device = device or cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run the Kaggle preflight first")
    out = ROOT / "results" / cfg["out"]
    cfg_hash = _json_hash({"config": cfg, "budget": budget, "snapshot_fingerprint": panel.fingerprint})
    if out.exists():
        if not resume:
            from run_guards import require_fresh_output
            require_fresh_output(out)
        else:
            manifest_path = out / "run_manifest.json"
            if not manifest_path.exists():
                raise FileExistsError("cannot resume: output directory has no matching run manifest")
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("config_hash") != cfg_hash:
                raise ValueError("cannot resume with a different config or snapshot fingerprint")
            if prior.get("status") == "completed":
                raise FileExistsError("completed Round 2 experiments cannot be resumed or overwritten")
    else:
        out.mkdir(parents=True)
    if not resume or not (out / "run_manifest.json").exists():
        _atomic_json(out / "config.yaml.json", cfg)
        _atomic_json(out / "protocol.json", {"config": cfg, "budget": budget,
                                               "snapshot_fingerprint": panel.fingerprint,
                                               "snapshot_path": str(bundle_path.relative_to(ROOT)),
                                               "targets": panel.targets, "years": panel.years,
                                               "training_years": panel.split["train_years"],
                                               "validation_years": panel.split["val_years"],
                                               "environment": _environment(device),
                                               "fit_count_formula": "grid trials plus locked seed fits; V3 multi is one shared fit per seed"})
        _atomic_json(out / "run_manifest.json", {"config_hash": cfg_hash,
                                                   "snapshot_fingerprint": panel.fingerprint,
                                                   "status": "running", "created_at": _now()})
    stop_year = 2023
    campus_ids = panel.campus_ids
    val_idx = panel.val_index
    y, lag, eval_mask = _target_tensors(panel, device)
    policy, variant = cfg["relation_policy"], cfg["variant"]
    targets = cfg.get("targets", panel.targets)
    if cfg["tasks"] == "multi" and targets != panel.targets:
        raise ValueError("multi-task run must include all targets in the snapshot order")
    groups = [[t] for t in targets] if cfg["tasks"] == "single" else [targets]
    grid = [{"hidden": int(h), "dropout": float(d)} for h in budget["grid"]["hidden"]
            for d in budget["grid"]["dropout"]]
    ledger = out / "trial_events.jsonl"
    candidate_csv = out / "candidate_scores.csv"

    for group in groups:
        group_id = "all_targets" if len(group) > 1 else group[0]
        indices = [panel.targets.index(t) for t in group]
        relations = _group_relations(panel, group, policy)
        x = _prepare_inputs(panel, relations, variant, device)
        candidate_rows = []
        for point in grid:
            trial_id = f"tune_{group_id}_h{point['hidden']}_d{str(point['dropout']).replace('.', 'p')}"
            trial_file = out / "trials" / f"{trial_id}.json"
            if trial_file.exists():
                saved = json.loads(trial_file.read_text(encoding="utf-8"))
                if saved.get("config_hash") != cfg_hash:
                    raise ValueError(f"inconsistent prior candidate trial: {trial_id}")
                if saved.get("status") == "completed":
                    candidate_rows.append(saved["result"])
                    continue
                if not resume:
                    raise ValueError(f"incomplete candidate trial requires --resume: {trial_id}")
            checkpoint = out / "checkpoints" / "resume" / f"{trial_id}.pt"
            _append_event(ledger, {"event": "trial_started", "trial_id": trial_id, "seed": 0,
                                   "targets": group, **point})
            _append_csv(out / "epoch_metrics.csv", [], ["trial_id", "epoch"])
            try:
                fit = fit_trial(variant=variant, x=x, y=y, lag=lag, eval_mask=eval_mask,
                                target_indices=indices, target_names=group, panel=panel, relations=relations,
                                hidden=point["hidden"], dropout=point["dropout"], epochs=budget["max_epochs"],
                                lr=budget["lr"], weight_decay=budget["weight_decay"],
                                patience=budget["patience"], seed=0, device=device, tune=True,
                                gradient_clip_norm=cfg.get("gradient_clip_norm"), config_hash=cfg_hash,
                                checkpoint_path=checkpoint, target_mode=target_mode,
                                on_epoch=lambda item, tid=trial_id: _append_csv(
                                    out / "epoch_metrics.csv", [{"trial_id": tid, "phase": "tune", **item}],
                                    ["trial_id", "epoch"]))
                result = {"trial_id": trial_id, "targets": group, **point, "seed": 0,
                          "selected_epoch": fit["best_epoch"], "val_ratio": fit["val_ratio"],
                          "val_mae": dict(zip(group, fit["val_mae"])), "status": "completed"}
                candidate_rows.append(result)
                _atomic_json(trial_file, {"config_hash": cfg_hash, "status": "completed", "result": result})
                _append_csv(candidate_csv, [result], ["trial_id"])
                if checkpoint.exists():
                    checkpoint.unlink()
                _append_event(ledger, {"event": "trial_completed", **result})
            except Exception as exc:
                _atomic_json(trial_file, {"config_hash": cfg_hash, "status": "failed",
                                          "trial_id": trial_id, "error": repr(exc),
                                          "traceback": traceback.format_exc()})
                _append_event(ledger, {"event": "trial_failed", "trial_id": trial_id,
                                       "targets": group, "error": repr(exc)})
                raise
        selected = min(candidate_rows, key=lambda row: (row["val_ratio"], row["selected_epoch"]))
        selection_path = out / "selected_configs.json"
        current = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        current[group_id] = {"targets": group, "relations": relations, "hidden": selected["hidden"],
                             "dropout": selected["dropout"], "epochs": selected["selected_epoch"],
                             "selection_val_ratio": selected["val_ratio"]}
        _atomic_json(selection_path, current)

        group_seed_rows = []
        for seed in panel.split["seeds"]:
            seed_id = f"{group_id}_seed_{seed}"
            result_path = out / "seed_results" / f"{seed_id}.json"
            prediction_path = out / "predictions" / f"{seed_id}.csv"
            if result_path.exists() and prediction_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("config_hash") != cfg_hash or saved.get("status") != "completed":
                    raise ValueError(f"inconsistent previous seed result: {seed_id}")
                group_seed_rows.extend(saved["metrics"])
                continue
            seed_checkpoint = out / "checkpoints" / "final" / f"{seed_id}.pt"
            _append_event(ledger, {"event": "seed_started", "seed_id": seed_id, "seed": seed,
                                   "targets": group, "selected_config": selected})
            fit = fit_trial(variant=variant, x=x, y=y, lag=lag, eval_mask=eval_mask,
                            target_indices=indices, target_names=group, panel=panel, relations=relations,
                            hidden=selected["hidden"], dropout=selected["dropout"], epochs=selected["selected_epoch"],
                            lr=budget["lr"], weight_decay=budget["weight_decay"], patience=None, seed=seed,
                            device=device, tune=False, gradient_clip_norm=cfg.get("gradient_clip_norm"),
                            config_hash=cfg_hash, target_mode=target_mode,
                            checkpoint_path=out / "checkpoints" / "resume" / f"score_{seed_id}.pt",
                            on_epoch=lambda item, sid=seed_id: _append_csv(
                                out / "epoch_metrics.csv", [{"trial_id": sid, "phase": "score", **item}],
                                ["trial_id", "epoch"]))
            rows = fit["prediction_rows"]
            pred_frame = pd.DataFrame(rows)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            temp_pred = prediction_path.with_suffix(".csv.tmp")
            pred_frame.to_csv(temp_pred, index=False)
            os.replace(temp_pred, prediction_path)
            metrics_rows = []
            for k, target in enumerate(group):
                metrics_rows.append({"model": cfg["name"], "variant": variant, "target": target,
                                     "seed": int(seed), "n": int((pred_frame.target == target).sum()),
                                     "val_mae": fit["val_mae"][k],
                                     "persistence_mae": fit["val_mae"][k] / fit["val_ratios"][k],
                                     "normalized_mae": fit["val_ratios"][k],
                                     "selected_epoch": int(selected["selected_epoch"]),
                                     "hidden": int(selected["hidden"]), "dropout": float(selected["dropout"]),
                                     "snapshot_fingerprint": panel.fingerprint})
            _atomic_json(result_path, {"config_hash": cfg_hash, "status": "completed", "seed": seed,
                                       "targets": group, "selected_config": selected, "metrics": metrics_rows,
                                       "training_history": fit["history"]})
            _atomic_torch_save(seed_checkpoint, {"variant": variant, "seed": seed,
                                 "state_dict": fit["model"].state_dict(), "center": fit["center"],
                                 "scale": fit["scale"], "selected_config": selected,
                                 "snapshot_fingerprint": panel.fingerprint})
            group_seed_rows.extend(metrics_rows)
            _append_csv(out / "per_seed_metrics.csv", metrics_rows, ["target", "seed"])
            score_resume = out / "checkpoints" / "resume" / f"score_{seed_id}.pt"
            if score_resume.exists():
                score_resume.unlink()
            _append_event(ledger, {"event": "seed_completed", "seed_id": seed_id, "seed": seed,
                                   "targets": group, "selected_epoch": fit["best_epoch"],
                                   "val_ratio": fit["val_ratio"], "prediction_file": str(prediction_path.relative_to(out)),
                                   "prediction_grad_norm": fit["prediction_grad_norm"],
                                   "gate_values": fit["gate_values"]})
        if group_seed_rows:
            _append_csv(out / "per_seed_metrics.csv", group_seed_rows, ["target", "seed"])
    _atomic_json(out / "run_manifest.json", {"config_hash": cfg_hash,
                                               "snapshot_fingerprint": panel.fingerprint,
                                               "status": "completed", "completed_at": _now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--resume", action="store_true", help="resume only a matching incomplete Round 2 run")
    args = parser.parse_args()
    run(args.config, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
