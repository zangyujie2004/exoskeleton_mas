from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from dataset import NUM_PERIODS, SEGS_PER_PERIOD, build_sample_index, mat_to_period_arrays, split_period_emg_equal_parts
from model import build_model

PERIOD_NAMES = [
    "wrist",
    "four_fingers",
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "static_stretch",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PeriodStretchDataset(Dataset):
    def __init__(
        self,
        session_indices: list[int],
        period_idx: int,
        init_data_root: Path | str,
        label_xlsx: Path | str | None,
    ) -> None:
        if period_idx < 0 or period_idx >= NUM_PERIODS:
            raise ValueError(f"period_idx must be in [0, {NUM_PERIODS}), got {period_idx}")
        self.session_indices = list(session_indices)
        self.period_idx = int(period_idx)
        self.init_data_root = Path(init_data_root)
        self.label_xlsx = label_xlsx
        self.metas = build_sample_index(self.init_data_root, label_xlsx)
        self._period_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.session_indices) * SEGS_PER_PERIOD

    def _period(self, global_session_idx: int) -> np.ndarray:
        if global_session_idx not in self._period_cache:
            meta = self.metas[global_session_idx]
            self._period_cache[global_session_idx] = mat_to_period_arrays(meta.mat_path)[self.period_idx]
        return self._period_cache[global_session_idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s_local, stretch_idx = divmod(idx, SEGS_PER_PERIOD)
        global_s = self.session_indices[s_local]
        meta = self.metas[global_s]
        full_period = self._period(global_s)
        parts = split_period_emg_equal_parts(full_period, SEGS_PER_PERIOD)
        x = parts[stretch_idx]
        y = float(meta.label[self.period_idx, stretch_idx])
        return {
            "emg": x,
            "target": y,
            "session": global_s,
            "period": self.period_idx,
            "stretch": stretch_idx,
        }


def stretch_emg_to_chunk_batch(
    batch: list[dict[str, Any]],
    chunk_length: int,
    max_chunks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = len(batch)
    m, l = max_chunks, chunk_length
    t_cap = m * l
    c = batch[0]["emg"].shape[0]
    x_out = torch.zeros(bsz, m, c, l, dtype=torch.float32, device=device)
    mask = torch.zeros(bsz, m, dtype=torch.float32, device=device)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.float32, device=device)

    for i, item in enumerate(batch):
        emg = item["emg"]
        t_orig = emg.shape[1]
        t_use = min(t_orig, t_cap)
        if t_use > 0:
            sl = emg[:, :t_use]
            if t_use < t_cap:
                pad = np.zeros((c, t_cap - t_use), dtype=np.float32)
                sl = np.concatenate([sl, pad], axis=1)
            arr = sl.reshape(c, m, l).transpose(1, 0, 2)
            x_out[i] = torch.from_numpy(arr).to(device)
            n_used = (t_use + l - 1) // l
            mask[i, :n_used] = 1.0
    return x_out, mask, targets


def _batch_mae_mse(pred: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    mae = (pred - y).abs().mean().item()
    mse = ((pred - y) ** 2).mean().item()
    return mae, mse


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    chunk_length: int,
    max_chunks: int,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    period_idx: int,
) -> tuple[float, float, int]:
    model.train()
    loss_fn = nn.L1Loss()
    sum_mae, sum_mse, n = 0.0, 0.0, 0
    for batch in loader:
        xb, mask, y = stretch_emg_to_chunk_batch(batch, chunk_length, max_chunks, device)
        opt.zero_grad(set_to_none=True)
        pred = model(xb, mask)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        mae, mse = _batch_mae_mse(pred.detach(), y)
        writer.add_scalar(f"period_{period_idx}/train_mae_batch", mae, step)
        writer.add_scalar(f"period_{period_idx}/train_mse_batch", mse, step)
        step += 1
        sum_mae += mae * len(batch)
        sum_mse += mse * len(batch)
        n += len(batch)
    denom = max(n, 1)
    return sum_mae / denom, sum_mse / denom, step


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    chunk_length: int,
    max_chunks: int,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    period_idx: int,
) -> tuple[float, float, int]:
    model.eval()
    sum_mae, sum_mse, n = 0.0, 0.0, 0
    for batch in loader:
        xb, mask, y = stretch_emg_to_chunk_batch(batch, chunk_length, max_chunks, device)
        pred = model(xb, mask)
        mae, mse = _batch_mae_mse(pred, y)
        writer.add_scalar(f"period_{period_idx}/test_mae_batch", mae, step)
        writer.add_scalar(f"period_{period_idx}/test_mse_batch", mse, step)
        step += 1
        sum_mae += mae * len(batch)
        sum_mse += mse * len(batch)
        n += len(batch)
    denom = max(n, 1)
    return sum_mae / denom, sum_mse / denom, step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 8 period-specific models and rank by test metrics."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--logdir", type=str, default=None, help="override tensorboard logdir")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = Path(__file__).resolve().parent / args.config
    cfg = load_yaml(cfg_path)

    init_root = Path(cfg["init_data_root"])
    if not init_root.is_absolute():
        init_root = Path(__file__).resolve().parent / init_root
    label_xlsx = cfg.get("label_xlsx")
    if label_xlsx:
        label_xlsx = Path(label_xlsx)
        if not label_xlsx.is_absolute():
            label_xlsx = Path(__file__).resolve().parent / label_xlsx

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    chunk_length = int(cfg["chunk_length"])
    max_chunks = int(cfg["max_chunks"])
    hidden_dim = int(cfg["hidden_dim"])
    model_type = str(cfg.get("model_type", "cnn_transformer"))
    nhead = int(cfg.get("transformer_nhead", 8))
    num_layers = int(cfg.get("transformer_layers", 2))
    t_ff_mult = int(cfg.get("transformer_ff_mult", 4))
    dim_feedforward = int(hidden_dim * t_ff_mult)
    transformer_dropout = float(cfg.get("transformer_dropout", 0.1))

    batch_size = int(cfg["batch_size"])
    epochs = int(args.epochs) if args.epochs is not None else int(cfg["epochs"])
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    train_ratio = float(cfg.get("train_ratio", 0.7))
    test_ratio = float(cfg.get("test_ratio", 0.3))
    num_workers = int(cfg.get("num_workers", 0))
    device_s = str(cfg.get("device", "cuda"))
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")

    if abs(train_ratio + test_ratio - 1.0) > 1e-3:
        raise ValueError(f"train_ratio ({train_ratio}) + test_ratio ({test_ratio}) should sum to 1.0")

    logdir_cfg = args.logdir or cfg.get("tensorboard_logdir") or "runs/mas"
    logdir = Path(logdir_cfg)
    if not logdir.is_absolute():
        logdir = Path(__file__).resolve().parent / logdir
    exp_logdir = logdir / "train_period"
    exp_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(exp_logdir))

    ckpt_dir = Path(cfg.get("checkpoint_dir") or "checkpoints")
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(__file__).resolve().parent / ckpt_dir
    period_ckpt_dir = ckpt_dir / "train_period"
    period_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 固定同一组 session 划分，保证 8 个阶段比较公平。
    n_sessions = len(build_sample_index(init_root, label_xlsx))
    all_idx = list(range(n_sessions))
    random.shuffle(all_idx)
    n_test = max(1, int(round(n_sessions * test_ratio)))
    n_train = max(1, n_sessions - n_test)
    test_sessions = all_idx[:n_test]
    train_sessions = all_idx[n_test : n_test + n_train]

    period_results: list[dict[str, Any]] = []
    for period_idx in range(NUM_PERIODS):
        period_name = PERIOD_NAMES[period_idx] if period_idx < len(PERIOD_NAMES) else f"period_{period_idx}"
        train_ds = PeriodStretchDataset(train_sessions, period_idx, init_root, label_xlsx)
        test_ds = PeriodStretchDataset(test_sessions, period_idx, init_root, label_xlsx)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda b: b,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda b: b,
        )

        model = build_model(
            model_type,
            chunk_length,
            max_chunks,
            hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            transformer_dropout=transformer_dropout,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        tb_step = 0
        best_test_mae = float("inf")
        best_test_mse = float("inf")
        best_epoch = 0
        period_ckpt = period_ckpt_dir / f"best_p{period_idx}_{period_name}.pt"
        for ep in range(1, epochs + 1):
            tr_mae, tr_mse, tb_step = train_one_epoch(
                model, train_loader, opt, chunk_length, max_chunks, device, writer, tb_step, period_idx
            )
            te_mae, te_mse, tb_step = eval_epoch(
                model, test_loader, chunk_length, max_chunks, device, writer, tb_step, period_idx
            )
            writer.add_scalar(f"period_{period_idx}/epoch_train_mae", tr_mae, ep)
            writer.add_scalar(f"period_{period_idx}/epoch_train_mse", tr_mse, ep)
            writer.add_scalar(f"period_{period_idx}/epoch_test_mae", te_mae, ep)
            writer.add_scalar(f"period_{period_idx}/epoch_test_mse", te_mse, ep)
            if te_mae < best_test_mae:
                best_test_mae = te_mae
                best_test_mse = te_mse
                best_epoch = ep
                torch.save(
                    {
                        "model": model.state_dict(),
                        "period_idx": period_idx,
                        "period_name": period_name,
                        "epoch": ep,
                        "test_mae": te_mae,
                        "test_mse": te_mse,
                        "model_type": model_type,
                        "chunk_length": chunk_length,
                        "max_chunks": max_chunks,
                        "hidden_dim": hidden_dim,
                        "transformer_nhead": nhead,
                        "transformer_layers": num_layers,
                        "transformer_dim_feedforward": dim_feedforward,
                        "transformer_dropout": transformer_dropout,
                        "train_sessions": train_sessions,
                        "test_sessions": test_sessions,
                        "seed": seed,
                        "init_root": str(init_root.resolve()),
                        "label_xlsx": str(label_xlsx.resolve()) if label_xlsx else None,
                    },
                    period_ckpt,
                )
            print(
                f"[period {period_idx} {period_name}] epoch {ep:03d}  "
                f"train_mae {tr_mae:.4f}  train_mse {tr_mse:.4f}  "
                f"test_mae {te_mae:.4f}  test_mse {te_mse:.4f}  best_test_mae {best_test_mae:.4f}"
            )

        period_results.append(
            {
                "period_idx": period_idx,
                "period_name": period_name,
                "best_epoch": best_epoch,
                "best_test_mae": best_test_mae,
                "best_test_mse": best_test_mse,
                "checkpoint": str(period_ckpt.resolve()),
            }
        )

    writer.close()

    ranked = sorted(period_results, key=lambda x: x["best_test_mae"])
    print("\n=== Ranking by test MAE (lower is better) ===")
    for i, r in enumerate(ranked, start=1):
        print(
            f"#{i}  period {r['period_idx']} ({r['period_name']})  "
            f"best_epoch {r['best_epoch']}  test_mae {r['best_test_mae']:.4f}  test_mse {r['best_test_mse']:.4f}"
        )

    summary_path = period_ckpt_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ranked": ranked,
                "all_results": period_results,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"train_period done. tensorboard: {exp_logdir.resolve()}  summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
