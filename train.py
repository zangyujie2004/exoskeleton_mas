from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    NUM_PERIODS,
    SEGS_PER_PERIOD,
    build_sample_index,
    mat_to_period_arrays,
    split_period_emg_equal_parts,
)
from model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FineStretchDataset(Dataset):
    """
    精细样本：每个 period 整段肌电 ``(9, T)`` 沿时间**均分为 6 份**，第 ``k`` 份对应
    该阶段第 ``k`` 次牵伸的**绝对 MAS**（与标签矩阵 ``(8,6)`` 逐项一致）。
    """

    def __init__(self, session_indices: list[int], init_data_root: Path | str, label_xlsx: Path | str | None) -> None:
        self.session_indices = list(session_indices)
        self.init_data_root = Path(init_data_root)
        self.label_xlsx = label_xlsx
        self.metas = build_sample_index(self.init_data_root, label_xlsx)
        self._period_cache: dict[int, list[np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.session_indices) * NUM_PERIODS * SEGS_PER_PERIOD

    def _periods(self, global_session_idx: int) -> list[np.ndarray]:
        if global_session_idx not in self._period_cache:
            meta = self.metas[global_session_idx]
            self._period_cache[global_session_idx] = mat_to_period_arrays(meta.mat_path)
        return self._period_cache[global_session_idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        per_session = NUM_PERIODS * SEGS_PER_PERIOD
        s_local, rest = divmod(idx, per_session)
        period_idx = rest // SEGS_PER_PERIOD
        stretch_idx = rest % SEGS_PER_PERIOD
        global_s = self.session_indices[s_local]
        meta = self.metas[global_s]
        # ① 整段 period → 时间均分 6 份（与 MAS 列顺序对齐）；② chunk 化仅在 collate 中对该子段做
        full_period = self._periods(global_s)[period_idx]
        parts = split_period_emg_equal_parts(full_period, SEGS_PER_PERIOD)
        x = parts[stretch_idx]
        y = float(meta.label[period_idx, stretch_idx])
        return {
            "emg": x,
            "target": y,
            "session": global_s,
            "period": period_idx,
            "stretch": stretch_idx,
        }


def stretch_emg_to_chunk_batch(
    batch: list[dict[str, Any]],
    chunk_length: int,
    max_chunks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将**已是单次牵伸子段**的肌电 ``(9, T)``（Dataset 里已对整 period 做完 6 等分）再做成模型输入。

    ``max_chunks`` / ``chunk_length`` **只影响**该子段如何截断/补零并切成 ``(M,9,L)`` token，
    **不会**改变 period 的 6 等分边界；若子段长度超过 ``M*L``，仅**截断尾部**（见 ``t_use``）。
    """
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


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
) -> tuple[float, float, int]:
    """Returns (epoch_mean_mae, epoch_mean_mse, next_global_step). Logs per train batch."""
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
        writer.add_scalar("train/mae", mae, step)
        writer.add_scalar("train/mse", mse, step)
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
    tb_prefix: str = "test",
) -> tuple[float, float, int]:
    """Returns (epoch_mean_mae, epoch_mean_mse, next_global_step). Logs per batch under ``tb_prefix``."""
    model.eval()
    sum_mae, sum_mse, n = 0.0, 0.0, 0
    for batch in loader:
        xb, mask, y = stretch_emg_to_chunk_batch(batch, chunk_length, max_chunks, device)
        pred = model(xb, mask)
        mae, mse = _batch_mae_mse(pred, y)
        writer.add_scalar(f"{tb_prefix}/mae", mae, step)
        writer.add_scalar(f"{tb_prefix}/mse", mse, step)
        step += 1
        sum_mae += mae * len(batch)
        sum_mse += mse * len(batch)
        n += len(batch)
    denom = max(n, 1)
    return sum_mae / denom, sum_mse / denom, step


def _load_model_from_checkpoint(
    ck: dict[str, Any],
    chunk_length: int,
    max_chunks: int,
    hidden_dim: int,
    device: torch.device,
) -> nn.Module:
    mt = str(ck.get("model_type", "cnn_mlp"))
    if mt in ("cnn_transformer", "cnn-transformer", "transformer"):
        d_ff = ck.get("transformer_dim_feedforward")
        if d_ff is None:
            mult = int(ck.get("transformer_ff_mult", 4))
            d_ff = int(hidden_dim * mult)
        model = build_model(
            "cnn_transformer",
            chunk_length,
            max_chunks,
            hidden_dim,
            nhead=int(ck.get("transformer_nhead", 8)),
            num_layers=int(ck.get("transformer_layers", 2)),
            dim_feedforward=int(d_ff),
            transformer_dropout=float(ck.get("transformer_dropout", 0.1)),
        )
    else:
        model = build_model("cnn_mlp", chunk_length, max_chunks, hidden_dim)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device)


@torch.no_grad()
def run_eval_checkpoint(
    ck: dict[str, Any],
    init_root: Path,
    label_xlsx: Path | None,
    test_sessions: list[int],
    chunk_length: int,
    max_chunks: int,
    hidden_dim: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
) -> tuple[float, float, int]:
    """在 checkpoint 记录的测试会话划分上评估。"""
    test_ds = FineStretchDataset(test_sessions, init_root, label_xlsx)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: b,
    )
    model = _load_model_from_checkpoint(ck, chunk_length, max_chunks, hidden_dim, device)
    return eval_epoch(model, test_loader, chunk_length, max_chunks, device, writer, step, tb_prefix="test")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EMG→absolute MAS per stretch: each period EMG split into 6 equal time parts; CNN-Transformer; 70/30 split; L1 + TensorBoard.",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--logdir", type=str, default=None, help="override tensorboard logdir")
    parser.add_argument(
        "--eval_only",
        type=str,
        default=None,
        metavar="CKPT",
        help="only evaluate on test split stored in checkpoint",
    )
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

    if args.eval_only:
        ckpt_path = Path(args.eval_only)
        if not ckpt_path.is_file():
            ckpt_path = Path(__file__).resolve().parent / args.eval_only
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(ckpt_path, map_location=device)
        init_root = Path(ck["init_root"])
        lx = ck.get("label_xlsx")
        label_xlsx = Path(lx) if lx else None
        test_sessions = list(ck.get("test_sessions", ck.get("val_sessions", [])))
        chunk_length = int(ck["chunk_length"])
        max_chunks = int(ck["max_chunks"])
        hidden_dim = int(ck["hidden_dim"])
        logdir_cfg = args.logdir or cfg.get("tensorboard_logdir") or "runs/mas_eval"
        logdir = Path(logdir_cfg)
        if not logdir.is_absolute():
            logdir = Path(__file__).resolve().parent / logdir
        logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(logdir))
        te_mae, te_mse, _ = run_eval_checkpoint(
            ck,
            init_root,
            label_xlsx,
            test_sessions,
            chunk_length,
            max_chunks,
            hidden_dim,
            int(cfg.get("batch_size", 16)),
            int(cfg.get("num_workers", 0)),
            device,
            writer,
            0,
        )
        writer.add_scalar("eval/test_mae", te_mae, 0)
        writer.add_scalar("eval/test_mse", te_mse, 0)
        writer.close()
        print(f"eval_only  test_mae {te_mae:.4f}  test_mse {te_mse:.4f}  tb: {logdir.resolve()}")
        return

    logdir_cfg = args.logdir or cfg.get("tensorboard_logdir") or "runs/mas"
    logdir = Path(logdir_cfg)
    if not logdir.is_absolute():
        logdir = Path(__file__).resolve().parent / logdir
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    tb_step = 0

    n_sessions = len(build_sample_index(init_root, label_xlsx))
    all_idx = list(range(n_sessions))
    random.shuffle(all_idx)
    n_test = max(1, int(round(n_sessions * test_ratio)))
    n_train = max(1, n_sessions - n_test)
    test_sessions = all_idx[:n_test]
    train_sessions = all_idx[n_test : n_test + n_train]

    train_ds = FineStretchDataset(train_sessions, init_root, label_xlsx)
    test_ds = FineStretchDataset(test_sessions, init_root, label_xlsx)

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

    ckpt_dir = Path(cfg.get("checkpoint_dir") or "checkpoints")
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(__file__).resolve().parent / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    best_test = float("inf")
    for ep in range(1, epochs + 1):
        tr_mae, tr_mse, tb_step = train_one_epoch(
            model, train_loader, opt, chunk_length, max_chunks, device, writer, tb_step
        )
        te_mae, te_mse, tb_step = eval_epoch(
            model, test_loader, chunk_length, max_chunks, device, writer, tb_step, tb_prefix="test"
        )
        writer.add_scalar("epoch/train_mae", tr_mae, ep)
        writer.add_scalar("epoch/train_mse", tr_mse, ep)
        writer.add_scalar("epoch/test_mae", te_mae, ep)
        writer.add_scalar("epoch/test_mse", te_mse, ep)
        if te_mae < best_test:
            best_test = te_mae
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "test_mae": te_mae,
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
                    "train_ratio": train_ratio,
                    "test_ratio": test_ratio,
                    "seed": seed,
                    "init_root": str(init_root.resolve()),
                    "label_xlsx": str(label_xlsx.resolve()) if label_xlsx else None,
                },
                ckpt_path,
            )
        print(
            f"epoch {ep:03d}  train_mae {tr_mae:.4f}  train_mse {tr_mse:.4f}  "
            f"test_mae {te_mae:.4f}  test_mse {te_mse:.4f}  best_test_mae {best_test:.4f}"
        )

    writer.close()
    print(
        f"done. best test MAE: {best_test:.4f}  tensorboard: {logdir.resolve()}  "
        f"checkpoint: {ckpt_path.resolve()}"
    )


if __name__ == "__main__":
    main()
