"""Monte Carlo Channel SHAP for 1D ResNet (9 EMG channels).

v(A) = -MAE(A) on the test set when channels in A use real values and others use a
train-set baseline (mean or zero). Shapley values are approximated by random
permutations; then top-k curves and k* = min{k | Delta_k <= tau} are reported.

Optional: ``--random_mask_train`` (with ``--train``) samples many mask patterns per epoch;
``--channel_search`` runs multistart hill-climb on a fixed model;
``--greedy_forward_train``: forward channel search with **test MAE** (full **train** split). Use
``--greedy_beam_width 1`` (default) for classic greedy (45 runs), or ``--greedy_beam_width 3`` for beam search.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.ml_method import is_sklearn_model_type
from models.model import build_model
from train_prediction import (
    WindowSpasmDataset,
    collate_batch,
    eval_one_epoch,
    parse_period_filter,
    regression_metrics,
    set_seed,
    train_one_epoch,
)
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[misc, assignment]


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def channel_train_mean(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Per-channel mean over all samples and time: shape (C,)."""
    sum_c: torch.Tensor | None = None
    n_tokens = 0
    for x, _y in loader:
        x = x.to(device, non_blocking=True)
        b, c, t = x.shape
        s = x.sum(dim=(0, 2))
        if sum_c is None:
            sum_c = torch.zeros(c, device=device, dtype=x.dtype)
        sum_c = sum_c + s
        n_tokens += b * t
    assert sum_c is not None and n_tokens > 0
    return sum_c / float(n_tokens)


def apply_channel_mask(
    x: torch.Tensor,
    active: set[int],
    baseline: torch.Tensor,
    *,
    num_channels: int,
) -> torch.Tensor:
    """Replace inactive channels with baseline[c] broadcast to (B, T)."""
    b, c, t = x.shape
    if c != num_channels:
        raise ValueError(f"Expected {num_channels} channels, got {c}")
    base = baseline.to(device=x.device, dtype=x.dtype).view(1, c, 1).expand(b, c, t)
    out = base.clone()
    for ch in active:
        out[:, ch, :] = x[:, ch, :]
    return out


@torch.no_grad()
def eval_metrics_masked(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    active: set[int],
    baseline: torch.Tensor,
    *,
    num_channels: int,
) -> dict[str, float]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        xm = apply_channel_mask(x, active, baseline, num_channels=num_channels)
        pred = model(xm)
        ys.append(y.detach().cpu().numpy().reshape(-1))
        ps.append(pred.detach().cpu().numpy().reshape(-1))
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return regression_metrics(y_true, y_pred)


def mc_channel_shap(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    baseline: torch.Tensor,
    *,
    num_channels: int,
    n_permutations: int,
    seed: int,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Permutation-based MC Shapley; returns (phi, list of permutation_marginal_mae_drop)."""
    rng = np.random.default_rng(int(seed))
    phi = np.zeros(num_channels, dtype=np.float64)
    mae_trace: list[float] = []

    evals_per_perm = num_channels + 1
    total_evals = n_permutations * evals_per_perm
    pbar = None
    if show_progress and _tqdm is not None:
        pbar = _tqdm(
            total=total_evals,
            desc="MC Channel SHAP (test passes)",
            unit="eval",
            dynamic_ncols=True,
            mininterval=0.3,
        )

    def _tick() -> None:
        if pbar is not None:
            pbar.update(1)

    channels = list(range(num_channels))
    for pi in range(n_permutations):
        order = channels.copy()
        rng.shuffle(order)
        active: set[int] = set()
        mae_prev = eval_metrics_masked(
            model, test_loader, device, active, baseline, num_channels=num_channels
        )["mae"]
        _tick()
        for ch in order:
            active.add(ch)
            mae_new = eval_metrics_masked(
                model, test_loader, device, active, baseline, num_channels=num_channels
            )["mae"]
            # v = -MAE => marginal = (-mae_new) - (-mae_prev) = mae_prev - mae_new
            phi[ch] += mae_prev - mae_new
            mae_prev = mae_new
            _tick()
        mae_trace.append(float(mae_prev))
        if show_progress and pbar is None:
            print(
                f"[progress] MC permutations {pi + 1}/{n_permutations} "
                f"({100.0 * (pi + 1) / n_permutations:.1f}%)",
                flush=True,
            )

    if pbar is not None:
        pbar.close()

    phi /= float(n_permutations)
    return phi, mae_trace


def train_one_epoch_random_channel_mask(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    baseline: torch.Tensor,
    num_channels: int,
    py_rng: random.Random,
    *,
    mask_p_full: float,
    aug_noise_std: float = 0.0,
    aug_scale: float = 0.0,
    aug_channel_dropout: float = 0.0,
    max_grad_norm: float = 0.0,
) -> tuple[dict[str, float], int]:
    """Like train_prediction.train_one_epoch but each batch uses random retained channels (rest → baseline)."""
    model.train()
    loss_fn = nn.SmoothL1Loss()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if aug_noise_std > 0:
            x = x + torch.randn_like(x) * aug_noise_std
        if aug_scale > 0:
            scale = 1.0 + (torch.rand(x.size(0), 1, 1, device=x.device) * 2.0 - 1.0) * aug_scale
            x = x * scale
        if aug_channel_dropout > 0:
            keep = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > aug_channel_dropout).float()
            x = x * keep

        if py_rng.random() < float(mask_p_full):
            active: set[int] = set(range(num_channels))
        else:
            n_mask = py_rng.randint(1, num_channels)
            masked = set(py_rng.sample(range(num_channels), n_mask))
            active = set(range(num_channels)) - masked
        xm = apply_channel_mask(x, active, baseline, num_channels=num_channels)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xm)
        loss = loss_fn(pred, y)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            mae = torch.mean(torch.abs(pred - y)).item()
            writer.add_scalar("batch/train_loss", float(loss.item()), step)
            writer.add_scalar("batch/train_mae", float(mae), step)
        step += 1
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return regression_metrics(y_true, y_pred), step


def train_one_epoch_fixed_channels(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    baseline: torch.Tensor,
    active: set[int],
    num_channels: int,
    *,
    aug_noise_std: float = 0.0,
    aug_scale: float = 0.0,
    aug_channel_dropout: float = 0.0,
    max_grad_norm: float = 0.0,
) -> tuple[dict[str, float], int]:
    """One epoch: only ``active`` channels carry signal; others → baseline."""
    model.train()
    loss_fn = nn.SmoothL1Loss()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if aug_noise_std > 0:
            x = x + torch.randn_like(x) * aug_noise_std
        if aug_scale > 0:
            scale = 1.0 + (torch.rand(x.size(0), 1, 1, device=x.device) * 2.0 - 1.0) * aug_scale
            x = x * scale
        if aug_channel_dropout > 0:
            keep = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > aug_channel_dropout).float()
            x = x * keep
        xm = apply_channel_mask(x, active, baseline, num_channels=num_channels)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xm)
        loss = loss_fn(pred, y)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            mae = torch.mean(torch.abs(pred - y)).item()
            writer.add_scalar("batch/train_loss", float(loss.item()), step)
            writer.add_scalar("batch/train_mae", float(mae), step)
        step += 1
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return regression_metrics(y_true, y_pred), step


def train_fixed_active_resnet(
    *,
    root: Path,
    pickle_path: Path,
    period_filter: list[int] | None,
    cfg: dict[str, Any],
    train_indices: list[int],
    train_seed: int,
    device: torch.device,
    model_type: str,
    hidden_dim: int,
    resnet_dropout: float,
    resnet_head_dropout: float,
    active: set[int],
    baseline: torch.Tensor,
    epochs: int,
    batch_size: int,
    num_workers: int,
    ckpt_path: Path,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    aug_noise_std: float,
    aug_scale: float,
    aug_channel_dropout: float,
    lr_patience: int,
    lr_factor: float,
    min_lr: float,
    log_tag: str,
) -> dict[str, Any]:
    """Train one ResNet with fixed retained channels; LR schedule + checkpoint by **test** MAE (masked).

    No val split: greedy path uses full ``train`` split for optimization; test is used for model
    selection (same caveat as train_prediction when tuning on test).
    """
    num_ch = 9
    if not active:
        raise ValueError("active must be non-empty")
    train_full = WindowSpasmDataset(pickle_path, split="train", period_indices=period_filter)
    test_ds = WindowSpasmDataset(pickle_path, split="test", period_indices=period_filter)
    train_ds = Subset(train_full, train_indices)
    train_loader_fit = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    test_loader_tr = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    set_seed(int(train_seed))
    model = build_model(
        model_type=model_type,
        chunk_length=0,
        max_chunks=0,
        hidden_dim=hidden_dim,
        resnet_dropout=resnet_dropout,
        resnet_head_dropout=resnet_head_dropout,
        sklearn_cfg=cfg,
        seed=int(train_seed),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_factor),
        patience=int(lr_patience),
        min_lr=float(min_lr),
    )
    logdir = root / "runs" / "greedy_forward_train" / log_tag
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    global_step = 0
    best_test_mae = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_test_metrics: dict[str, float] | None = None

    for ep in range(1, epochs + 1):
        tr_m, global_step = train_one_epoch_fixed_channels(
            model,
            train_loader_fit,
            optimizer,
            device,
            writer,
            global_step,
            baseline,
            active,
            num_ch,
            aug_noise_std=aug_noise_std,
            aug_scale=aug_scale,
            aug_channel_dropout=aug_channel_dropout,
            max_grad_norm=max_grad_norm,
        )
        te_m = eval_metrics_masked(model, test_loader_tr, device, active, baseline, num_channels=num_ch)
        scheduler.step(te_m["mae"])
        for k, v in tr_m.items():
            writer.add_scalar(f"epoch/train_{k}", v, ep)
        for k, v in te_m.items():
            writer.add_scalar(f"epoch/test_{k}", v, ep)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], ep)
        if float(te_m["mae"]) < best_test_mae:
            best_test_mae = float(te_m["mae"])
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_metrics = dict(te_m)
    writer.close()
    if best_state is None or best_test_metrics is None:
        raise RuntimeError("train_fixed_active_resnet: no best state")
    model.load_state_dict(best_state)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "epoch": best_epoch,
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "resnet_dropout": resnet_dropout,
            "resnet_head_dropout": resnet_head_dropout,
            "active_channels": sorted(active),
            "best_selection": "test_mae_masked",
            "metrics_test": best_test_metrics,
            "pickle_path": str(pickle_path.resolve()),
            "period_indices": period_filter,
        },
        ckpt_path,
    )
    return {
        "active": sorted(active),
        "best_test_mae": best_test_mae,
        "best_epoch": best_epoch,
        "metrics_test": {k: float(best_test_metrics[k]) for k in best_test_metrics},
        "checkpoint": str(ckpt_path.resolve()),
    }


def _format_greedy_ch_combo(active_sorted: list[int]) -> str:
    return "[" + ",".join(f"c{x}" for x in active_sorted) + "]"


def _print_greedy_test_table_header() -> None:
    print("k\t贪心通道组合\tTest MAE\tTest RMSE\tTest R²", flush=True)


def _print_greedy_test_table_row(k: int, active_sorted: list[int], mae: float, rmse: float, r2: float) -> None:
    print(
        f"{k}\t{_format_greedy_ch_combo(active_sorted)}\t{mae:.6f}\t{rmse:.6f}\t{r2:.6f}",
        flush=True,
    )


def _eval_greedy_checkpoint_on_test(
    ck_path: Path,
    *,
    device: torch.device,
    baseline: torch.Tensor,
    test_loader: DataLoader,
    num_ch: int,
    model_type: str,
    hidden_dim: int,
    resnet_dropout: float,
    resnet_head_dropout: float,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[list[int], dict[str, float]]:
    try:
        ck_d = torch.load(ck_path, map_location=device, weights_only=False)
    except TypeError:
        ck_d = torch.load(ck_path, map_location=device)
    active_k = set(int(c) for c in ck_d["active_channels"])
    mt = build_model(
        model_type=str(ck_d.get("model_type", model_type)),
        chunk_length=0,
        max_chunks=0,
        hidden_dim=int(ck_d.get("hidden_dim", hidden_dim)),
        resnet_dropout=float(ck_d.get("resnet_dropout", resnet_dropout)),
        resnet_head_dropout=float(ck_d.get("resnet_head_dropout", resnet_head_dropout)),
        sklearn_cfg=cfg,
        seed=seed,
    ).to(device)
    mt.load_state_dict(ck_d["model"])
    mt.eval()
    m = eval_metrics_masked(mt, test_loader, device, active_k, baseline, num_channels=num_ch)
    return sorted(active_k), m


def _beam_hypothesis_to_json(h: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON fields (sets) for steps_log."""
    return {
        "active": list(h["active"]),
        "best_test_mae": float(h["best_test_mae"]),
        "best_epoch": int(h["best_epoch"]),
        "metrics_test": {k: float(h["metrics_test"][k]) for k in h["metrics_test"]},
        "checkpoint": str(h["checkpoint"]),
        "parent_checkpoint": h.get("parent_checkpoint"),
        "parent_active": list(h["parent_active"]) if h.get("parent_active") is not None else None,
        "added_channel": h.get("added_channel"),
        "step_k": int(h["step_k"]),
    }


def _beam_find_parent(steps_log: list[dict[str, Any]], parent_ckpt: str) -> dict[str, Any] | None:
    for entry in reversed(steps_log):
        for h in entry["beam"]:
            if str(h["checkpoint"]) == parent_ckpt:
                return h
    return None


def _beam_backtrack_best_path(steps_log: list[dict[str, Any]], winner: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered k=1..k=9 along parent_checkpoint chain (winner last)."""
    chain: list[dict[str, Any]] = [winner]
    cur = winner
    while cur.get("parent_checkpoint"):
        p = _beam_find_parent(steps_log, str(cur["parent_checkpoint"]))
        if p is None:
            break
        chain.append(p)
        cur = p
    chain.reverse()
    return chain


def run_beam_forward_channel_training(
    *,
    root: Path,
    pickle_path: Path,
    period_filter: list[int] | None,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    model_type: str,
    hidden_dim: int,
    resnet_dropout: float,
    resnet_head_dropout: float,
    epochs_per_model: int,
    batch_size: int,
    num_workers: int,
    out_dir: Path,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    aug_noise_std: float,
    aug_scale: float,
    aug_channel_dropout: float,
    lr_patience: int,
    lr_factor: float,
    min_lr: float,
    baseline_mode: str,
    keep_all_candidate_ckpts: bool,
    beam_width: int,
) -> dict[str, Any]:
    """Beam search on channel sets: keep top ``beam_width`` partial hypotheses by test MAE; train expansions each step."""
    num_ch = 9
    all_ch = set(range(num_ch))
    bw = max(1, int(beam_width))
    train_full = WindowSpasmDataset(pickle_path, split="train", period_indices=period_filter)
    test_ds = WindowSpasmDataset(pickle_path, split="test", period_indices=period_filter)
    n_full = len(train_full)
    train_idx = list(range(n_full))
    train_ds = Subset(train_full, train_idx)
    mean_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    if baseline_mode == "mean":
        baseline = channel_train_mean(mean_loader, device)
    else:
        baseline = torch.zeros(num_ch, device=device, dtype=torch.float32)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_candidates"
    scratch.mkdir(parents=True, exist_ok=True)

    steps_log: list[dict[str, Any]] = []
    train_counter = 0

    print(
        f"\n[greedy_train] **Beam search** beam_width={bw} | 子模型与 beam 剪枝均按 **test MAE**；"
        "训练用 train 全量。注意：用 test 选模偏乐观。\n",
        flush=True,
    )

    # ----- step k=1: all singletons -----
    step1_hyps: list[dict[str, Any]] = []
    for ci, c in enumerate(range(num_ch)):
        train_counter += 1
        active = {c}
        train_seed = int(seed) + 1 * 10007 + ci * 917
        log_tag = f"beam_step01_cand{ci:02d}_c{c}"
        ck_try = scratch / f"beam_step01_cand{ci:02d}.pt"
        print(
            f"[greedy_train] beam run {train_counter} k=1 active={sorted(active)} -> {ck_try.name}",
            flush=True,
        )
        row = train_fixed_active_resnet(
            root=root,
            pickle_path=pickle_path,
            period_filter=period_filter,
            cfg=cfg,
            train_indices=train_idx,
            train_seed=train_seed,
            device=device,
            model_type=model_type,
            hidden_dim=hidden_dim,
            resnet_dropout=resnet_dropout,
            resnet_head_dropout=resnet_head_dropout,
            active=active,
            baseline=baseline,
            epochs=epochs_per_model,
            batch_size=batch_size,
            num_workers=num_workers,
            ckpt_path=ck_try,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            aug_noise_std=aug_noise_std,
            aug_scale=aug_scale,
            aug_channel_dropout=aug_channel_dropout,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            min_lr=min_lr,
            log_tag=log_tag[:120],
        )
        mt = row["metrics_test"]
        print(
            f"         test@best_ckpt: MAE={mt['mae']:.6f} RMSE={mt['rmse']:.6f} R2={mt['r2']:.6f} "
            f"| best_epoch={row['best_epoch']}",
            flush=True,
        )
        step1_hyps.append(
            {
                "active": sorted(active),
                "active_set": set(active),
                "fkey": frozenset(active),
                "best_test_mae": float(row["best_test_mae"]),
                "best_epoch": int(row["best_epoch"]),
                "metrics_test": dict(row["metrics_test"]),
                "checkpoint": str(ck_try.resolve()),
                "parent_checkpoint": None,
                "parent_active": None,
                "added_channel": None,
                "step_k": 1,
            }
        )

    ranked1 = sorted(step1_hyps, key=lambda x: x["best_test_mae"])
    beam = ranked1[: min(bw, len(ranked1))]
    steps_log.append({"step_k": 1, "beam": [_beam_hypothesis_to_json(h) for h in beam]})
    print(f"[greedy_train] beam step 1: kept top-{len(beam)} / 9 by test MAE", flush=True)

    # ----- steps k=2..9 -----
    for step_k in range(2, num_ch + 1):
        merged: dict[frozenset[int], dict[str, Any]] = {}
        expand_i = 0
        for b in beam:
            parent_active = set(b["active_set"])
            for ch in sorted(all_ch - parent_active):
                train_counter += 1
                expand_i += 1
                na = set(parent_active) | {ch}
                fk = frozenset(na)
                train_seed = int(seed) + step_k * 10007 + expand_i * 919 + sum(na) * 13
                log_tag = f"beam_step{step_k:02d}_exp{expand_i:03d}_" + "_".join(str(x) for x in sorted(na))
                ck_try = scratch / f"beam_step{step_k:02d}_exp{expand_i:03d}.pt"
                print(
                    f"[greedy_train] beam run {train_counter} k={step_k} active={sorted(na)} -> {ck_try.name}",
                    flush=True,
                )
                row = train_fixed_active_resnet(
                    root=root,
                    pickle_path=pickle_path,
                    period_filter=period_filter,
                    cfg=cfg,
                    train_indices=train_idx,
                    train_seed=train_seed,
                    device=device,
                    model_type=model_type,
                    hidden_dim=hidden_dim,
                    resnet_dropout=resnet_dropout,
                    resnet_head_dropout=resnet_head_dropout,
                    active=na,
                    baseline=baseline,
                    epochs=epochs_per_model,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    ckpt_path=ck_try,
                    lr=lr,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    aug_noise_std=aug_noise_std,
                    aug_scale=aug_scale,
                    aug_channel_dropout=aug_channel_dropout,
                    lr_patience=lr_patience,
                    lr_factor=lr_factor,
                    min_lr=min_lr,
                    log_tag=log_tag[:120],
                )
                mt = row["metrics_test"]
                print(
                    f"         test@best_ckpt: MAE={mt['mae']:.6f} RMSE={mt['rmse']:.6f} R2={mt['r2']:.6f} "
                    f"| best_epoch={row['best_epoch']}",
                    flush=True,
                )
                mae = float(row["best_test_mae"])
                hyp = {
                    "active": sorted(na),
                    "active_set": set(na),
                    "fkey": fk,
                    "best_test_mae": mae,
                    "best_epoch": int(row["best_epoch"]),
                    "metrics_test": dict(row["metrics_test"]),
                    "checkpoint": str(ck_try.resolve()),
                    "parent_checkpoint": str(b["checkpoint"]),
                    "parent_active": sorted(parent_active),
                    "added_channel": int(ch),
                    "step_k": step_k,
                }
                if fk not in merged or mae < merged[fk]["best_test_mae"]:
                    merged[fk] = hyp

        ranked = sorted(merged.values(), key=lambda x: x["best_test_mae"])
        beam = ranked[: min(bw, len(ranked))]
        steps_log.append({"step_k": step_k, "beam": [_beam_hypothesis_to_json(h) for h in beam]})
        print(
            f"[greedy_train] beam step {step_k}: expanded {len(merged)} unique sets, kept top-{len(beam)}",
            flush=True,
        )

    assert beam
    winner = min(beam, key=lambda x: x["best_test_mae"])
    path_chain = _beam_backtrack_best_path(steps_log, winner)

    path_checkpoints: list[str] = []
    test_eval_rows: list[dict[str, Any]] = []
    for step_k, hyp in enumerate(path_chain, start=1):
        path_ck = out_dir / f"beam_best_path_step{step_k:02d}_k{step_k}.pt"
        shutil.copy2(hyp["checkpoint"], path_ck)
        path_checkpoints.append(str(path_ck.resolve()))
        act_sorted, m_te = _eval_greedy_checkpoint_on_test(
            path_ck,
            device=device,
            baseline=baseline,
            test_loader=test_loader,
            num_ch=num_ch,
            model_type=model_type,
            hidden_dim=hidden_dim,
            resnet_dropout=resnet_dropout,
            resnet_head_dropout=resnet_head_dropout,
            cfg=cfg,
            seed=seed,
        )
        test_eval_rows.append(
            {
                "step_k": step_k,
                "k_channels": len(act_sorted),
                "active": act_sorted,
                "mae": float(m_te["mae"]),
                "rmse": float(m_te["rmse"]),
                "r2": float(m_te["r2"]),
                "checkpoint": str(path_ck.resolve()),
            }
        )

    if not keep_all_candidate_ckpts:
        for p in scratch.glob("*.pt"):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    print("\n[greedy_train] beam — 最优回溯路径上各 k 的 test 指标\n", flush=True)
    _print_greedy_test_table_header()
    for row in test_eval_rows:
        _print_greedy_test_table_row(
            int(row["step_k"]),
            list(row["active"]),
            float(row["mae"]),
            float(row["rmse"]),
            float(row["r2"]),
        )

    mae_full = test_eval_rows[-1]["mae"]
    for row in test_eval_rows:
        row["delta_mae_vs_full"] = float(row["mae"] - mae_full)
        row["mae_improvement_vs_full"] = float(mae_full - row["mae"])

    best_row = min(test_eval_rows, key=lambda r: r["mae"])
    summary: dict[str, Any] = {
        "greedy_forward_train": True,
        "greedy_mode": "beam",
        "beam_width": bw,
        "n_trains": train_counter,
        "selection_metric": "test_mae_masked",
        "test_eval_on_path_models": test_eval_rows,
        "best_on_test_mae": {
            "step_k": best_row["step_k"],
            "k_channels": best_row["k_channels"],
            "active": best_row["active"],
            "mae": best_row["mae"],
            "rmse": best_row["rmse"],
            "r2": best_row["r2"],
            "checkpoint": best_row["checkpoint"],
        },
        "test_mae_full_model_path": mae_full,
        "beam_steps": steps_log,
        "beam_winner_terminal": _beam_hypothesis_to_json(winner),
        "beam_best_path_chain": [_beam_hypothesis_to_json(h) for h in path_chain],
        "path_checkpoints": path_checkpoints,
    }
    print("\n[greedy_train] beam 汇总（相对 k=9 全通道 ΔMAE）\n", flush=True)
    _print_greedy_test_table_header()
    for row in test_eval_rows:
        _print_greedy_test_table_row(
            int(row["step_k"]),
            list(row["active"]),
            float(row["mae"]),
            float(row["rmse"]),
            float(row["r2"]),
        )
    print(
        "\nΔMAE vs k=9 (test): "
        + "  ".join(f"k{row['step_k']}={row['delta_mae_vs_full']:+.6f}" for row in test_eval_rows),
        flush=True,
    )
    b = summary["best_on_test_mae"]
    print(
        f"\n[greedy_train] beam **best test MAE** at step_k={b['step_k']} (k={b['k_channels']} ch) "
        f"MAE={b['mae']:.6f} RMSE={b['rmse']:.6f} R2={b['r2']:.6f} | active={b['active']}",
        flush=True,
    )
    return summary


def run_greedy_forward_channel_training(
    *,
    root: Path,
    pickle_path: Path,
    period_filter: list[int] | None,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    model_type: str,
    hidden_dim: int,
    resnet_dropout: float,
    resnet_head_dropout: float,
    epochs_per_model: int,
    batch_size: int,
    num_workers: int,
    out_dir: Path,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    aug_noise_std: float,
    aug_scale: float,
    aug_channel_dropout: float,
    lr_patience: int,
    lr_factor: float,
    min_lr: float,
    baseline_mode: str,
    keep_all_candidate_ckpts: bool,
) -> dict[str, Any]:
    """45 trainings (beam_width=1): step 1 train 9 single-channel models, …, step 9 full channel; **test MAE** picks each step."""
    num_ch = 9
    all_ch = set(range(num_ch))
    train_full = WindowSpasmDataset(pickle_path, split="train", period_indices=period_filter)
    test_ds = WindowSpasmDataset(pickle_path, split="test", period_indices=period_filter)
    n_full = len(train_full)
    train_idx = list(range(n_full))
    train_ds = Subset(train_full, train_idx)
    mean_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    if baseline_mode == "mean":
        baseline = channel_train_mean(mean_loader, device)
    else:
        baseline = torch.zeros(num_ch, device=device, dtype=torch.float32)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_candidates"
    scratch.mkdir(parents=True, exist_ok=True)

    path_active: list[int] = []
    path_checkpoints: list[str] = []
    steps_log: list[dict[str, Any]] = []
    test_eval_rows: list[dict[str, Any]] = []
    train_counter = 0

    print("[greedy_train] 每步结束后在 test 上评估路径模型（贪心通道组合）\n", flush=True)
    _print_greedy_test_table_header()

    for step_k in range(1, num_ch + 1):
        remaining = sorted(all_ch - set(path_active))
        if step_k == 1:
            candidates = [{c} for c in range(num_ch)]
        else:
            candidates = [set(path_active) | {c} for c in remaining]

        best_test = math.inf
        winner: dict[str, Any] | None = None
        cand_rows: list[dict[str, Any]] = []

        for ci, cand_active in enumerate(candidates):
            train_counter += 1
            train_seed = int(seed) + step_k * 10007 + ci * 917
            log_tag = f"step{step_k:02d}_cand{ci:02d}_" + "_".join(str(c) for c in sorted(cand_active))
            ck_try = scratch / f"step{step_k:02d}_cand{ci:02d}.pt"
            print(
                f"[greedy_train] run {train_counter}/45 k={step_k} active={sorted(cand_active)} -> {ck_try.name}",
                flush=True,
            )
            row = train_fixed_active_resnet(
                root=root,
                pickle_path=pickle_path,
                period_filter=period_filter,
                cfg=cfg,
                train_indices=train_idx,
                train_seed=train_seed,
                device=device,
                model_type=model_type,
                hidden_dim=hidden_dim,
                resnet_dropout=resnet_dropout,
                resnet_head_dropout=resnet_head_dropout,
                active=set(cand_active),
                baseline=baseline,
                epochs=epochs_per_model,
                batch_size=batch_size,
                num_workers=num_workers,
                ckpt_path=ck_try,
                lr=lr,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                aug_noise_std=aug_noise_std,
                aug_scale=aug_scale,
                aug_channel_dropout=aug_channel_dropout,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                min_lr=min_lr,
                log_tag=log_tag[:120],
            )
            mt = row["metrics_test"]
            print(
                f"         test@best_ckpt: MAE={mt['mae']:.6f} RMSE={mt['rmse']:.6f} R2={mt['r2']:.6f} "
                f"| best_epoch={row['best_epoch']}",
                flush=True,
            )
            cand_rows.append(
                {
                    "candidate_index": ci,
                    "active": row["active"],
                    "best_test_mae": row["best_test_mae"],
                    "best_epoch": row["best_epoch"],
                    "metrics_test": row["metrics_test"],
                    "checkpoint": row["checkpoint"],
                }
            )
            if row["best_test_mae"] < best_test:
                best_test = row["best_test_mae"]
                winner = row

        assert winner is not None
        path_active = list(winner["active"])
        path_ck = out_dir / f"greedy_path_step{step_k:02d}_k{step_k}.pt"
        shutil.copy2(winner["checkpoint"], path_ck)
        if not keep_all_candidate_ckpts:
            for row in cand_rows:
                p = Path(row["checkpoint"])
                if p.resolve() != path_ck.resolve():
                    p.unlink(missing_ok=True)
        path_checkpoints.append(str(path_ck.resolve()))
        steps_log.append(
            {
                "step_k": step_k,
                "winner_active": path_active,
                "winner_best_test_mae": best_test,
                "candidates": cand_rows,
                "path_checkpoint": str(path_ck.resolve()),
            }
        )
        act_sorted, m_te = _eval_greedy_checkpoint_on_test(
            path_ck,
            device=device,
            baseline=baseline,
            test_loader=test_loader,
            num_ch=num_ch,
            model_type=model_type,
            hidden_dim=hidden_dim,
            resnet_dropout=resnet_dropout,
            resnet_head_dropout=resnet_head_dropout,
            cfg=cfg,
            seed=seed,
        )
        ck_s = str(path_ck.resolve())
        test_eval_rows.append(
            {
                "step_k": step_k,
                "k_channels": len(act_sorted),
                "active": act_sorted,
                "mae": float(m_te["mae"]),
                "rmse": float(m_te["rmse"]),
                "r2": float(m_te["r2"]),
                "checkpoint": ck_s,
            }
        )
        _print_greedy_test_table_row(
            step_k,
            act_sorted,
            float(m_te["mae"]),
            float(m_te["rmse"]),
            float(m_te["r2"]),
        )

    mae_full = test_eval_rows[-1]["mae"]
    for row in test_eval_rows:
        row["delta_mae_vs_full"] = float(row["mae"] - mae_full)
        row["mae_improvement_vs_full"] = float(mae_full - row["mae"])

    best_row = min(test_eval_rows, key=lambda r: r["mae"])
    summary = {
        "greedy_forward_train": True,
        "greedy_mode": "greedy",
        "greedy_beam_width": 1,
        "n_trains": train_counter,
        "selection_metric": "test_mae_masked",
        "test_eval_on_path_models": test_eval_rows,
        "best_on_test_mae": {
            "step_k": best_row["step_k"],
            "k_channels": best_row["k_channels"],
            "active": best_row["active"],
            "mae": best_row["mae"],
            "rmse": best_row["rmse"],
            "r2": best_row["r2"],
            "checkpoint": best_row["checkpoint"],
        },
        "test_mae_full_model_path": mae_full,
        "greedy_steps": steps_log,
        "path_checkpoints": path_checkpoints,
    }
    print("\n[greedy_train] 汇总（相对全通道 k=9 的 ΔMAE）\n", flush=True)
    _print_greedy_test_table_header()
    for row in test_eval_rows:
        _print_greedy_test_table_row(
            int(row["step_k"]),
            list(row["active"]),
            float(row["mae"]),
            float(row["rmse"]),
            float(row["r2"]),
        )
    print(
        "\nΔMAE vs k=9 (test): "
        + "  ".join(f"k{row['step_k']}={row['delta_mae_vs_full']:+.6f}" for row in test_eval_rows),
        flush=True,
    )
    b = summary["best_on_test_mae"]
    print(
        f"\n[greedy_train] **best test MAE** at step_k={b['step_k']} (k={b['k_channels']} ch) "
        f"MAE={b['mae']:.6f} RMSE={b['rmse']:.6f} R2={b['r2']:.6f} | active={b['active']}",
        flush=True,
    )
    return summary


def hill_climb_retained_channels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    baseline: torch.Tensor,
    *,
    num_channels: int,
    init_active: set[int],
) -> tuple[set[int], float, list[dict[str, Any]]]:
    """Greedy coordinate descent on binary channel mask; minimize MAE (inactive → baseline)."""
    active = set(init_active)
    if not active:
        raise ValueError("init_active must be non-empty")
    metrics = eval_metrics_masked(model, loader, device, active, baseline, num_channels=num_channels)
    best_mae = float(metrics["mae"])
    history: list[dict[str, Any]] = [
        {"step": 0, "active": sorted(active), "mae": best_mae, "rmse": metrics["rmse"], "r2": metrics["r2"]}
    ]
    step_i = 0
    while True:
        improved = False
        for ch in range(num_channels):
            cand = set(active)
            if ch in cand:
                cand.remove(ch)
            else:
                cand.add(ch)
            if not cand:
                continue
            m = eval_metrics_masked(model, loader, device, cand, baseline, num_channels=num_channels)
            mae_c = float(m["mae"])
            if mae_c < best_mae - 1e-9:
                active = cand
                best_mae = mae_c
                improved = True
                step_i += 1
                history.append(
                    {
                        "step": step_i,
                        "flip_channel": ch,
                        "active": sorted(active),
                        "mae": best_mae,
                        "rmse": float(m["rmse"]),
                        "r2": float(m["r2"]),
                    }
                )
                break
        if not improved:
            break
    return active, best_mae, history


def channel_search_multistart(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    baseline: torch.Tensor,
    *,
    num_channels: int,
    n_restarts: int,
    seed: int,
) -> dict[str, Any]:
    """Several hill-climbs; each restart starts from «all but one random channel» (one masked)."""
    rng = random.Random(int(seed) + 7919)
    best_active: set[int] | None = None
    best_mae = math.inf
    runs: list[dict[str, Any]] = []
    for r in range(int(n_restarts)):
        masked = rng.randrange(num_channels)
        init_active = set(range(num_channels)) - {masked}
        active, mae, hist = hill_climb_retained_channels(
            model, loader, device, baseline, num_channels=num_channels, init_active=init_active
        )
        runs.append(
            {
                "restart": r,
                "init_masked_channel": masked,
                "local_best_active": sorted(active),
                "local_best_mae": mae,
                "steps": len(hist) - 1,
                "trace": hist,
            }
        )
        if mae < best_mae:
            best_mae = mae
            best_active = active
    assert best_active is not None
    return {
        "n_restarts": int(n_restarts),
        "best_active": sorted(best_active),
        "best_mae": best_mae,
        "runs": runs,
    }


def train_resnet_then_save(
    *,
    root: Path,
    pickle_path: Path,
    period_filter: list[int] | None,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    model_type: str,
    hidden_dim: int,
    resnet_dropout: float,
    resnet_head_dropout: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    ckpt_path: Path,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    aug_noise_std: float,
    aug_scale: float,
    aug_channel_dropout: float,
    lr_patience: int,
    lr_factor: float,
    min_lr: float,
    random_mask_train: bool,
    mask_p_full: float,
) -> tuple[int, dict[str, float]]:
    """Match train_prediction window ResNet loop; save best by test MAE. Returns (best_epoch, best_test_metrics)."""
    num_ch = 9
    train_full = WindowSpasmDataset(pickle_path, split="train", period_indices=period_filter)
    test_ds = WindowSpasmDataset(pickle_path, split="test", period_indices=period_filter)
    n_full = len(train_full)
    val_ratio = float(cfg.get("val_ratio", 0.1))
    n_val = max(1, int(round(n_full * val_ratio)))
    n_val = min(n_val, n_full - 1) if n_full > 1 else n_val
    indices = list(range(n_full))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_full, val_idx)
    train_loader_fit = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    test_loader_tr = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    py_rng = random.Random(int(seed))
    baseline_train: torch.Tensor | None = None
    if random_mask_train:
        mean_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
        baseline_train = channel_train_mean(mean_loader, device)

    model = build_model(
        model_type=model_type,
        chunk_length=0,
        max_chunks=0,
        hidden_dim=hidden_dim,
        resnet_dropout=resnet_dropout,
        resnet_head_dropout=resnet_head_dropout,
        sklearn_cfg=cfg,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_factor),
        patience=int(lr_patience),
        min_lr=float(min_lr),
    )

    logdir = root / "runs" / "channel_shap_pretrain"
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    global_step = 0
    best_test_mae = math.inf
    best_epoch = 0
    best_test_metrics: dict[str, float] | None = None

    print(
        f"[train] fresh {model_type} | epochs={epochs} | train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} | "
        f"random_mask_train={random_mask_train} mask_p_full={mask_p_full} | checkpoint -> {ckpt_path.resolve()}",
        flush=True,
    )
    for ep in range(1, epochs + 1):
        if random_mask_train:
            assert baseline_train is not None
            tr_m, global_step = train_one_epoch_random_channel_mask(
                model,
                train_loader_fit,
                optimizer,
                device,
                writer,
                global_step,
                baseline_train,
                num_ch,
                py_rng,
                mask_p_full=float(mask_p_full),
                aug_noise_std=aug_noise_std,
                aug_scale=aug_scale,
                aug_channel_dropout=aug_channel_dropout,
                max_grad_norm=max_grad_norm,
            )
        else:
            tr_m, global_step = train_one_epoch(
                model,
                train_loader_fit,
                optimizer,
                device,
                writer,
                global_step,
                aug_noise_std=aug_noise_std,
                aug_scale=aug_scale,
                aug_channel_dropout=aug_channel_dropout,
                max_grad_norm=max_grad_norm,
            )
        va_m = eval_one_epoch(model, val_loader, device)
        te_m = eval_one_epoch(model, test_loader_tr, device)
        scheduler.step(te_m["mae"])
        for k, v in tr_m.items():
            writer.add_scalar(f"epoch/train_{k}", v, ep)
        for k, v in va_m.items():
            writer.add_scalar(f"epoch/val_{k}", v, ep)
        for k, v in te_m.items():
            writer.add_scalar(f"epoch/test_{k}", v, ep)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], ep)

        if te_m["mae"] < best_test_mae:
            best_test_mae = te_m["mae"]
            best_epoch = ep
            best_test_metrics = dict(te_m)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "model_type": model_type,
                    "hidden_dim": hidden_dim,
                    "resnet_dropout": resnet_dropout,
                    "resnet_head_dropout": resnet_head_dropout,
                    "max_grad_norm": max_grad_norm,
                    "weight_decay": weight_decay,
                    "period_indices": period_filter,
                    "best_selection": "test_mae",
                    "metrics_val": va_m,
                    "metrics_test": te_m,
                    "seed": seed,
                    "pickle_path": str(pickle_path.resolve()),
                    "random_mask_train": bool(random_mask_train),
                    "mask_p_full": float(mask_p_full),
                },
                ckpt_path,
            )
        print(
            f"[train] epoch {ep:03d}/{epochs} | test MAE {te_m['mae']:.6f} RMSE {te_m['rmse']:.6f} R2 {te_m['r2']:.6f} | "
            f"best_test_MAE {best_test_mae:.6f} @ ep {best_epoch}",
            flush=True,
        )

    writer.close()
    if best_test_metrics is None:
        raise RuntimeError("Training produced no checkpoint (empty test set?).")
    bm = best_test_metrics
    print(
        "[train] summary: best on test (by MAE, same file as checkpoint) | "
        f"epoch={best_epoch} | MAE={bm['mae']:.6f} | RMSE={bm['rmse']:.6f} | R2={bm['r2']:.6f}",
        flush=True,
    )
    return best_epoch, best_test_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Channel SHAP (MC) for ResNet1D spasm regression.")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--pickle_path", type=str, default="init_window_cache.pkl")
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/resnet_spasm_best.pt")
    p.add_argument(
        "--train",
        action="store_true",
        help="Train ResNet from scratch (no resume), save to --checkpoint_path, then run SHAP.",
    )
    p.add_argument(
        "--random_mask_train",
        action="store_true",
        help="With --train: each batch random retained channels (1..C masked or all-on with prob mask_p_full).",
    )
    p.add_argument(
        "--mask_p_full",
        type=float,
        default=0.25,
        help="With --random_mask_train: probability of keeping all channels for that batch.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=32,
        help="Only with --train: training epochs (default 32).",
    )
    p.add_argument("--lr", type=float, default=None, help="Only with --train; default from config.")
    p.add_argument("--weight_decay", type=float, default=None, help="Only with --train; default from config.")
    p.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        help="With --train: architecture to train. Without --train: used only if checkpoint has no model_type.",
    )
    p.add_argument("--periods", type=str, default="0,1,7", help="Same as train_prediction; use 'all' for every period.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed if set.")
    p.add_argument("--mc_samples", type=int, default=200, help="Random permutations for MC Shapley.")
    p.add_argument(
        "--skip_shap",
        action="store_true",
        help="Skip MC SHAP and top-k curve (e.g. only channel search / faster).",
    )
    p.add_argument(
        "--channel_search",
        action="store_true",
        help="After loading model: multistart hill-climb on retained channels to minimize MAE.",
    )
    p.add_argument(
        "--channel_search_split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Split used to score each mask during search (test leaks into selection if test).",
    )
    p.add_argument(
        "--opt_restarts",
        type=int,
        default=9,
        help="Channel search: hill-climb restarts (each init = random single masked channel).",
    )
    p.add_argument("--baseline", type=str, default="mean", choices=("mean", "zero"))
    p.add_argument("--tau", type=float, default=0.05, help="Relative MAE slack for k*: (MAE_k - MAE_all) / MAE_all <= tau.")
    p.add_argument("--out_json", type=str, default="channel_shap_results.json")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm / permutation prints.")
    p.add_argument(
        "--greedy_forward_train",
        action="store_true",
        help="Channel forward search: test MAE; greedy (beam_width=1) or beam; writes JSON.",
    )
    p.add_argument(
        "--greedy_beam_width",
        type=int,
        default=1,
        help="1 = forward greedy (9+8+…+1 trains). >=2 = beam search keeping top-W hypotheses each depth.",
    )
    p.add_argument(
        "--greedy_out_dir",
        type=str,
        default="checkpoints/greedy_forward_path",
        help="Directory for greedy path checkpoints (9) and _candidates/ scratch.",
    )
    p.add_argument(
        "--greedy_epochs",
        type=int,
        default=None,
        help="Epochs per sub-model (default: same as --epochs, i.e. 32).",
    )
    p.add_argument(
        "--greedy_keep_all_ckpts",
        action="store_true",
        help="Keep every candidate checkpoint under _candidates/ (default: delete losers each step).",
    )
    p.add_argument(
        "--greedy_out_json",
        type=str,
        default="greedy_forward_path.json",
        help="JSON summary for --greedy_forward_train.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}

    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    pickle_path = Path(args.pickle_path)
    if not pickle_path.is_absolute():
        pickle_path = root / pickle_path

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path

    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 128))
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg.get("num_workers", 0))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    resnet_dropout = float(cfg.get("resnet_dropout", 0.15))
    resnet_head_dropout = float(cfg.get("resnet_head_dropout", 0.2))

    device_s = str(cfg.get("device", "cuda"))
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")

    period_filter = parse_period_filter(args.periods)
    train_full = WindowSpasmDataset(pickle_path, split="train", period_indices=period_filter)
    test_ds = WindowSpasmDataset(pickle_path, split="test", period_indices=period_filter)

    # Match train_prediction: same val split from train for mean baseline (use full train index list)
    n_full = len(train_full)
    val_ratio = float(cfg.get("val_ratio", 0.1))
    n_val = max(1, int(round(n_full * val_ratio)))
    n_val = min(n_val, n_full - 1) if n_full > 1 else n_val
    indices = list(range(n_full))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    val_ds = Subset(train_full, val_idx)
    train_ds = Subset(train_full, train_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    if bool(args.greedy_forward_train):
        if bool(args.train):
            raise ValueError("--greedy_forward_train is exclusive with --train (use greedy only).")
        if is_sklearn_model_type(str(args.model_type)):
            raise ValueError("--greedy_forward_train requires a ResNet model_type (e.g. resnet18).")
        greedy_out = Path(args.greedy_out_dir)
        if not greedy_out.is_absolute():
            greedy_out = root / greedy_out
        ep_g = int(args.greedy_epochs) if args.greedy_epochs is not None else int(args.epochs)
        lr = float(args.lr) if args.lr is not None else float(cfg.get("lr", 1e-3))
        weight_decay = float(args.weight_decay) if args.weight_decay is not None else float(
            cfg.get("prediction_weight_decay", cfg.get("weight_decay", 2e-3))
        )
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        aug_noise_std = float(cfg.get("aug_noise_std", 0.012))
        aug_scale = float(cfg.get("aug_scale", 0.15))
        aug_channel_dropout = float(cfg.get("aug_channel_dropout", 0.07))
        lr_patience = int(cfg.get("lr_patience", 8))
        lr_factor = float(cfg.get("lr_factor", 0.5))
        min_lr = float(cfg.get("min_lr", 1e-6))
        bw = int(args.greedy_beam_width)
        if bw < 1:
            raise ValueError("--greedy_beam_width must be >= 1.")
        if bw >= 2:
            summary = run_beam_forward_channel_training(
                root=root,
                pickle_path=pickle_path,
                period_filter=period_filter,
                cfg=cfg,
                seed=seed,
                device=device,
                model_type=str(args.model_type),
                hidden_dim=hidden_dim,
                resnet_dropout=resnet_dropout,
                resnet_head_dropout=resnet_head_dropout,
                epochs_per_model=ep_g,
                batch_size=batch_size,
                num_workers=num_workers,
                out_dir=greedy_out,
                lr=lr,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                aug_noise_std=aug_noise_std,
                aug_scale=aug_scale,
                aug_channel_dropout=aug_channel_dropout,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                min_lr=min_lr,
                baseline_mode=str(args.baseline),
                keep_all_candidate_ckpts=bool(args.greedy_keep_all_ckpts),
                beam_width=bw,
            )
        else:
            summary = run_greedy_forward_channel_training(
                root=root,
                pickle_path=pickle_path,
                period_filter=period_filter,
                cfg=cfg,
                seed=seed,
                device=device,
                model_type=str(args.model_type),
                hidden_dim=hidden_dim,
                resnet_dropout=resnet_dropout,
                resnet_head_dropout=resnet_head_dropout,
                epochs_per_model=ep_g,
                batch_size=batch_size,
                num_workers=num_workers,
                out_dir=greedy_out,
                lr=lr,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                aug_noise_std=aug_noise_std,
                aug_scale=aug_scale,
                aug_channel_dropout=aug_channel_dropout,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                min_lr=min_lr,
                baseline_mode=str(args.baseline),
                keep_all_candidate_ckpts=bool(args.greedy_keep_all_ckpts),
            )
        gj = Path(args.greedy_out_json)
        if not gj.is_absolute():
            gj = root / gj
        gj.parent.mkdir(parents=True, exist_ok=True)
        with gj.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[greedy_train] wrote {gj.resolve()}", flush=True)
        return

    if bool(args.random_mask_train) and not bool(args.train):
        raise ValueError("--random_mask_train requires --train.")
    if bool(args.random_mask_train) and not (0.0 <= float(args.mask_p_full) <= 1.0):
        raise ValueError("--mask_p_full must be in [0, 1].")

    if bool(args.train):
        if is_sklearn_model_type(str(args.model_type)):
            raise ValueError("--train requires a ResNet model_type (e.g. resnet18), not sklearn.")
        lr = float(args.lr) if args.lr is not None else float(cfg.get("lr", 1e-3))
        weight_decay = float(args.weight_decay) if args.weight_decay is not None else float(
            cfg.get("prediction_weight_decay", cfg.get("weight_decay", 2e-3))
        )
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        aug_noise_std = float(cfg.get("aug_noise_std", 0.012))
        aug_scale = float(cfg.get("aug_scale", 0.15))
        aug_channel_dropout = float(cfg.get("aug_channel_dropout", 0.07))
        lr_patience = int(cfg.get("lr_patience", 8))
        lr_factor = float(cfg.get("lr_factor", 0.5))
        min_lr = float(cfg.get("min_lr", 1e-6))
        train_resnet_then_save(
            root=root,
            pickle_path=pickle_path,
            period_filter=period_filter,
            cfg=cfg,
            seed=seed,
            device=device,
            model_type=str(args.model_type),
            hidden_dim=hidden_dim,
            resnet_dropout=resnet_dropout,
            resnet_head_dropout=resnet_head_dropout,
            epochs=int(args.epochs),
            batch_size=batch_size,
            num_workers=num_workers,
            ckpt_path=ckpt_path,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            aug_noise_std=aug_noise_std,
            aug_scale=aug_scale,
            aug_channel_dropout=aug_channel_dropout,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            min_lr=min_lr,
            random_mask_train=bool(args.random_mask_train),
            mask_p_full=float(args.mask_p_full),
        )
    elif not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (use --train to create one)")

    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)
    if ck.get("backend") == "sklearn":
        raise ValueError("Channel SHAP script expects a PyTorch ResNet checkpoint, not sklearn.")

    resnet_dropout = float(ck.get("resnet_dropout", resnet_dropout))
    resnet_head_dropout = float(ck.get("resnet_head_dropout", resnet_head_dropout))
    hidden_dim = int(ck.get("hidden_dim", hidden_dim))

    ck_model_type = ck.get("model_type")
    model_type = str(args.model_type)
    if ck_model_type is not None and str(ck_model_type) != model_type:
        print(
            f"[warn] checkpoint model_type={ck_model_type!r} != CLI --model_type={model_type!r}; "
            "using checkpoint value so state_dict loads."
        )
        model_type = str(ck_model_type)
    elif ck_model_type is not None:
        model_type = str(ck_model_type)

    model = build_model(
        model_type=model_type,
        chunk_length=0,
        max_chunks=0,
        hidden_dim=hidden_dim,
        resnet_dropout=resnet_dropout,
        resnet_head_dropout=resnet_head_dropout,
        sklearn_cfg=cfg,
        seed=seed,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    num_channels = 9
    if str(args.baseline) == "mean":
        baseline = channel_train_mean(train_loader, device)
    else:
        baseline = torch.zeros(num_channels, device=device, dtype=torch.float32)

    metrics_all = eval_metrics_masked(
        model, test_loader, device, set(range(num_channels)), baseline, num_channels=num_channels
    )
    mae_all = metrics_all["mae"]

    mc_n = int(args.mc_samples)
    evals_per_perm = num_channels + 1
    total_mc_evals = mc_n * evals_per_perm
    print(
        f"[info] device={device} | test n={len(test_ds)} | train(mean) n={len(train_ds)} | "
        f"test (all ch, masked pipeline) MAE={metrics_all['mae']:.6f} RMSE={metrics_all['rmse']:.6f} "
        f"R2={metrics_all['r2']:.6f} | MC permutations={mc_n} (~{total_mc_evals} full test-set evals)",
        flush=True,
    )

    channel_search_payload: dict[str, Any] | None = None
    if bool(args.channel_search):
        search_loader = val_loader if str(args.channel_search_split) == "val" else test_loader
        print(
            f"[channel_search] multistart hill-climb (min MAE) on {args.channel_search_split} | "
            f"restarts={int(args.opt_restarts)}",
            flush=True,
        )
        channel_search_payload = channel_search_multistart(
            model,
            search_loader,
            device,
            baseline,
            num_channels=num_channels,
            n_restarts=int(args.opt_restarts),
            seed=seed,
        )
        best_set = set(int(c) for c in channel_search_payload["best_active"])
        m_search = eval_metrics_masked(model, search_loader, device, best_set, baseline, num_channels=num_channels)
        m_test_best = eval_metrics_masked(model, test_loader, device, best_set, baseline, num_channels=num_channels)
        channel_search_payload["metrics_at_best_on_search_split"] = {k: float(m_search[k]) for k in ("mae", "rmse", "r2")}
        channel_search_payload["metrics_at_best_on_test"] = {k: float(m_test_best[k]) for k in ("mae", "rmse", "r2")}
        print(
            f"[channel_search] best active={channel_search_payload['best_active']} | "
            f"MAE on {args.channel_search_split}={m_search['mae']:.6f} | "
            f"MAE on test (report only)={m_test_best['mae']:.6f} RMSE={m_test_best['rmse']:.6f} R2={m_test_best['r2']:.6f}",
            flush=True,
        )

    tau = float(args.tau)
    show_p = not bool(args.no_progress)
    if bool(args.skip_shap):
        phi = np.zeros(num_channels, dtype=np.float64)
        order = list(range(num_channels))
        topk_rows = []
        k_star = None
        print("[info] --skip_shap: skipped MC Shapley and top-k curve.", flush=True)
    else:
        phi, _trace = mc_channel_shap(
            model,
            test_loader,
            device,
            baseline,
            num_channels=num_channels,
            n_permutations=mc_n,
            seed=seed,
            show_progress=show_p,
        )
        order = np.argsort(-phi).tolist()

        topk_pbar = None
        if show_p and _tqdm is not None:
            topk_pbar = _tqdm(total=num_channels, desc="Top-k curve", unit="k", dynamic_ncols=True, mininterval=0.2)

        topk_rows = []
        for k in range(1, num_channels + 1):
            keep = set(order[:k])
            m = eval_metrics_masked(model, test_loader, device, keep, baseline, num_channels=num_channels)
            if topk_pbar is not None:
                topk_pbar.update(1)
            elif show_p and _tqdm is None:
                print(f"[progress] top-k eval k={k}/{num_channels}", flush=True)
            delta_k = (m["mae"] - mae_all) / mae_all if mae_all > 1e-12 else float("nan")
            topk_rows.append(
                {
                    "k": k,
                    "channels": order[:k],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                    "delta_rel_mae": delta_k,
                }
            )

        if topk_pbar is not None:
            topk_pbar.close()

        k_star = None
        for row in topk_rows:
            if row["delta_rel_mae"] <= tau:
                k_star = int(row["k"])
                break

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = root / out_path

    payload: dict[str, Any] = {
        "checkpoint": str(ckpt_path.resolve()),
        "pickle_path": str(pickle_path.resolve()),
        "periods": "all" if period_filter is None else period_filter,
        "model_type": model_type,
        "trained_here": bool(args.train),
        "train_epochs": int(args.epochs) if bool(args.train) else None,
        "random_mask_train": bool(args.random_mask_train),
        "mask_p_full": float(args.mask_p_full) if bool(args.random_mask_train) else None,
        "baseline": args.baseline,
        "mc_samples": int(args.mc_samples),
        "skip_shap": bool(args.skip_shap),
        "tau": tau,
        "mae_all": mae_all,
        "test_all_channels_metrics": {k: float(metrics_all[k]) for k in ("mae", "rmse", "r2", "pearson", "spearman")},
        "phi": {str(i): float(phi[i]) for i in range(num_channels)},
        "rank_high_to_low": order,
        "topk_curve": topk_rows,
        "k_star": k_star,
    }
    if channel_search_payload is not None:
        payload["channel_search"] = channel_search_payload
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if not bool(args.skip_shap):
        print("\nChannel SHAP (phi, higher = more helpful on average):\n")
        for r, ch in enumerate(order):
            print(f"  rank {r + 1}: channel {ch}  phi={phi[ch]:.6f}")
        print("\nTop-k performance (fixed model, masked inputs):\n")
        print(f"{'k':>3}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}  {'Delta_k':>10}")
        for row in topk_rows:
            print(
                f"{row['k']:3d}  {row['mae']:10.6f}  {row['rmse']:10.6f}  {row['r2']:8.4f}  {row['delta_rel_mae']:10.4f}"
            )
        print(f"\n[done] k* (smallest k with Delta_k <= tau={tau}) = {k_star}")
    print(f"[done] wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
