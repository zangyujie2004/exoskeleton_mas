from __future__ import annotations

import argparse
import math
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from models.ml_method import dataloader_to_xy_arrays, is_sklearn_model_type
from models.model import build_model

# 与 preprocess / train 一致：0 手腕、1 四指、7 静态拉伸
DEFAULT_PERIOD_INDICES: tuple[int, ...] = (0, 1, 7)

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - fallback when scipy unavailable
    pearsonr = None
    spearmanr = None


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


class WindowSpasmDataset(Dataset):
    def __init__(
        self,
        pickle_path: Path | str,
        split: str,
        period_indices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.pickle_path = Path(pickle_path)
        with self.pickle_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict pickle, got {type(obj)}")
        if split not in obj:
            raise ValueError(f"Split '{split}' not found in pickle keys={list(obj.keys())}")
        rows = obj[split]
        if not isinstance(rows, list):
            raise ValueError(f"Split '{split}' must be list, got {type(rows)}")
        if period_indices is not None:
            allowed = {int(p) for p in period_indices}
            rows = [r for r in rows if int(r["period"]) in allowed]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        x = np.asarray(row["emg_data"], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected emg_data shape (C, T), got {x.shape}")
        if x.shape[0] > 9:
            x = x[:9]
        if x.shape[0] < 9:
            padded = np.zeros((9, x.shape[1]), dtype=np.float32)
            padded[: x.shape[0]] = x
            x = padded
        y = float(row["spasm_level"])
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "period": int(row["period"]),
            "subject_id": int(row["subject_id"]),
            "exp_id": int(row["exp_id"]),
        }

def collate_batch(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B, 9, T)
    y = torch.stack([b["y"] for b in batch], dim=0)  # (B,)
    return x, y


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if pearsonr is not None:
        return float(pearsonr(y_true, y_pred)[0])
    c = np.corrcoef(y_true, y_pred)
    return float(c[0, 1])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if spearmanr is not None:
        return float(spearmanr(y_true, y_pred)[0])
    rt = np.argsort(np.argsort(y_true))
    rp = np.argsort(np.argsort(y_pred))
    c = np.corrcoef(rt, rp)
    return float(c[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": _pearson(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    *,
    aug_noise_std: float = 0.0,
    aug_scale: float = 0.0,
    aug_channel_dropout: float = 0.0,
    max_grad_norm: float = 0.0,
) -> tuple[dict[str, float], int]:
    model.train()
    loss_fn = nn.SmoothL1Loss()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # Lightweight time-series augmentation for regularization.
        if aug_noise_std > 0:
            x = x + torch.randn_like(x) * aug_noise_std
        if aug_scale > 0:
            scale = 1.0 + (torch.rand(x.size(0), 1, 1, device=x.device) * 2.0 - 1.0) * aug_scale
            x = x * scale
        if aug_channel_dropout > 0:
            keep = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > aug_channel_dropout).float()
            x = x * keep
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
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


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return regression_metrics(y_true, y_pred)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        ys.append(y.detach().cpu().numpy().reshape(-1))
        ps.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def _scatter_diag_band(train_true: np.ndarray, train_pred: np.ndarray, band_override: float | None) -> float:
    if band_override is not None and band_override > 0:
        return float(band_override)
    res = np.abs(train_pred.astype(np.float64) - train_true.astype(np.float64))
    if res.size == 0:
        return 0.5
    q = float(np.percentile(res, 85))
    return max(0.12, min(q * 1.15, 2.0))


def save_test_scatter(
    test_true: np.ndarray,
    test_pred: np.ndarray,
    out_path: Path,
    *,
    band: float,
    axis_lo: float = 0.0,
    axis_hi: float = 4.0,
    writer: SummaryWriter | None = None,
    writer_step: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 竖向略长、分辨率略高，坐标固定在 [0, 4]（与 MAS 量纲一致）
    fig, ax = plt.subplots(figsize=(7.5, 10.5), dpi=160)
    ax.scatter(
        test_true,
        test_pred,
        s=36,
        alpha=0.55,
        c="#ff7f0e",
        edgecolors="none",
        label=f"Test (n={len(test_true)})",
        rasterized=True,
    )

    xs = np.linspace(axis_lo, axis_hi, 256)
    ax.plot(xs, xs, color="black", lw=1.8, ls="-", label="y = x", zorder=3)
    ax.plot(xs, xs + band, color="#2ca02c", lw=1.35, ls=(0, (4, 3)), alpha=0.75, label=f"y = x + {band:.3f}", zorder=2)
    ax.plot(xs, xs - band, color="#9467bd", lw=1.35, ls=(0, (4, 3)), alpha=0.75, label=f"y = x − {band:.3f}", zorder=2)
    ax.fill_between(xs, xs - band, xs + band, color="0.5", alpha=0.06, zorder=0)

    mae_te = float(np.mean(np.abs(test_pred - test_true))) if test_true.size else float("nan")
    ax.set_xlabel("True spasm_level")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Test scatter  |  MAE={mae_te:.4f}  |  band={band:.3f}  |  axes [{axis_lo}, {axis_hi}]")
    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", framealpha=0.92)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if writer is not None:
        try:
            writer.add_figure("final/test_scatter", fig, global_step=writer_step)
        except Exception as ex:  # pragma: no cover
            print(f"[tensorboard] add_figure test_scatter skipped: {ex}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAS/spasm regression model from window cache.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--pickle_path", type=str, default="init_window_cache.pkl")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=(
            "resnet18",
            "resnet34",
            "ts_resnet18",
            "ts_resnet34",
            "random_forest",
            "rf",
            "xgboost",
            "xgb",
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=None, help="Split from train set for model selection.")
    parser.add_argument("--early_stop_patience", type=int, default=64)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=int, default=None)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--aug_noise_std", type=float, default=None, help="default: cfg aug_noise_std or 0.012")
    parser.add_argument("--aug_scale", type=float, default=None, help="default: cfg aug_scale or 0.12")
    parser.add_argument("--aug_channel_dropout", type=float, default=None, help="default: cfg aug_channel_dropout or 0.07")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Clip gradient L2 norm (0 to disable). Default: cfg max_grad_norm or 1.0.",
    )
    parser.add_argument(
        "--resnet_dropout",
        type=float,
        default=None,
        help="Dropout inside ResNet blocks. Default: cfg resnet_dropout or 0.15.",
    )
    parser.add_argument(
        "--resnet_head_dropout",
        type=float,
        default=None,
        help="Dropout before final linear. Default: cfg resnet_head_dropout or 0.2.",
    )
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--scatter_band",
        type=float,
        default=0.5,
        help="Half-width for y=x±band soft envelope (default: from train |pred-true| ~85%%ile).",
    )
    parser.add_argument(
        "--no_train_test_scatter",
        action="store_true",
        help="Skip saving test-only scatter PNG after training.",
    )
    parser.add_argument(
        "--periods",
        type=str,
        default="0,1,7",
        help="Comma-separated period indices to keep (default 0,1,7). Use 'all' for every period.",
    )
    return parser.parse_args()


def parse_period_filter(spec: str | None) -> list[int] | None:
    if spec is None or not str(spec).strip() or str(spec).strip().lower() == "all":
        return None
    out: list[int] = []
    for part in str(spec).split(","):
        p = part.strip()
        if not p:
            continue
        if not p.lstrip("-").isdigit():
            raise ValueError(f"Invalid period token {p!r}; use integers 0..7 or 'all'.")
        out.append(int(p))
    uniq = sorted(set(out))
    for p in uniq:
        if p < 0 or p > 7:
            raise ValueError(f"period index must be in [0, 7], got {p}")
    if not uniq:
        return None
    return uniq


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    pickle_path = Path(args.pickle_path)
    if not pickle_path.is_absolute():
        pickle_path = root / pickle_path

    epochs = int(args.epochs) if args.epochs is not None else int(cfg.get("epochs", 32))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 128))
    lr = float(args.lr) if args.lr is not None else float(cfg.get("lr", 1e-3))
    weight_decay = float(args.weight_decay) if args.weight_decay is not None else float(
        cfg.get("prediction_weight_decay", cfg.get("weight_decay", 2e-3))
    )
    resnet_dropout = float(args.resnet_dropout) if args.resnet_dropout is not None else float(cfg.get("resnet_dropout", 0.15))
    resnet_head_dropout = (
        float(args.resnet_head_dropout) if args.resnet_head_dropout is not None else float(cfg.get("resnet_head_dropout", 0.2))
    )
    max_grad_norm = float(args.max_grad_norm) if args.max_grad_norm is not None else float(cfg.get("max_grad_norm", 1.0))
    aug_noise_std = float(args.aug_noise_std) if args.aug_noise_std is not None else float(cfg.get("aug_noise_std", 0.012))
    aug_scale = float(args.aug_scale) if args.aug_scale is not None else float(cfg.get("aug_scale", 0.15))
    aug_channel_dropout = (
        float(args.aug_channel_dropout) if args.aug_channel_dropout is not None else float(cfg.get("aug_channel_dropout", 0.07))
    )
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg.get("num_workers", 0))
    val_ratio = float(args.val_ratio) if args.val_ratio is not None else float(cfg.get("val_ratio", 0.1))
    early_stop_patience = int(args.early_stop_patience) if args.early_stop_patience is not None else int(
        cfg.get("early_stop_patience", 20)
    )
    lr_patience = int(args.lr_patience) if args.lr_patience is not None else int(cfg.get("lr_patience", 8))
    hidden_dim = int(cfg.get("hidden_dim", 256))

    device_s = str(cfg.get("device", "cuda"))
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")

    tb_cfg = args.logdir or cfg.get("tensorboard_logdir") or "runs/mas_prediction"
    logdir = Path(tb_cfg)
    if not logdir.is_absolute():
        logdir = root / logdir
    logdir = logdir / "resnet_spasm"
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else (root / "checkpoints" / "resnet_spasm_best.pt")
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    period_filter = parse_period_filter(args.periods)
    train_full_ds = WindowSpasmDataset(pickle_path, split="train", period_indices=period_filter)
    test_ds = WindowSpasmDataset(pickle_path, split="test", period_indices=period_filter)
    n_full = len(train_full_ds)
    n_val = max(1, int(round(n_full * val_ratio)))
    n_val = min(n_val, n_full - 1) if n_full > 1 else n_val
    indices = list(range(n_full))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_ds = Subset(train_full_ds, train_idx)
    val_ds = Subset(train_full_ds, val_idx)
    train_loader = DataLoader(
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
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    pf_s = "all" if period_filter is None else ",".join(str(p) for p in period_filter)
    cpu_dev = torch.device("cpu")

    if is_sklearn_model_type(args.model_type):
        model = build_model(
            model_type=args.model_type,
            chunk_length=0,
            max_chunks=0,
            hidden_dim=hidden_dim,
            resnet_dropout=resnet_dropout,
            resnet_head_dropout=resnet_head_dropout,
            sklearn_cfg=cfg,
            seed=seed,
        ).to(cpu_dev)
        train_loader_fit = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )
        X_tr, y_tr = dataloader_to_xy_arrays(train_loader_fit)
        model.fit(X_tr, y_tr)
        tr_m = eval_one_epoch(model, train_loader_fit, cpu_dev)
        va_m = eval_one_epoch(model, val_loader, cpu_dev)
        te_m = eval_one_epoch(model, test_loader, cpu_dev)
        best_test_mae = te_m["mae"]
        last_ep = 1
        for k, v in tr_m.items():
            writer.add_scalar(f"epoch/train_{k}", v, last_ep)
        for k, v in va_m.items():
            writer.add_scalar(f"epoch/val_{k}", v, last_ep)
        for k, v in te_m.items():
            writer.add_scalar(f"epoch/test_{k}", v, last_ep)
        writer.add_scalar("epoch/lr", 0.0, last_ep)
        torch.save(
            {
                "backend": "sklearn",
                "estimator": model.estimator,
                "sklearn_name": model.name,
                "epoch": last_ep,
                "model_type": args.model_type,
                "hidden_dim": hidden_dim,
                "period_indices": period_filter,
                "best_selection": "test_mae",
                "metrics_val": va_m,
                "metrics_test": te_m,
                "metrics_train": tr_m,
                "seed": seed,
                "pickle_path": str(pickle_path.resolve()),
            },
            ckpt_path,
        )
        print(
            f"[info] model={args.model_type} (sklearn) | periods={pf_s} | "
            f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} | n_train={len(y_tr)} | device=cpu"
        )
        print(
            "fit done | "
            f"train MAE {tr_m['mae']:.4f} RMSE {tr_m['rmse']:.4f} R2 {tr_m['r2']:.4f} | "
            f"val MAE {va_m['mae']:.4f} RMSE {va_m['rmse']:.4f} R2 {va_m['r2']:.4f} | "
            f"test MAE {te_m['mae']:.4f} RMSE {te_m['rmse']:.4f} R2 {te_m['r2']:.4f} "
            f"P {te_m['pearson']:.4f} S {te_m['spearman']:.4f} | "
            f"checkpoint {ckpt_path.resolve()}"
        )
    else:
        model = build_model(
            model_type=args.model_type,
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
            factor=float(args.lr_factor),
            patience=lr_patience,
            min_lr=float(args.min_lr),
        )

        print(
            f"[info] model={args.model_type} | periods={pf_s} | "
            f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
            f"| batch_size={batch_size} | wd={weight_decay} | resnet_drop={resnet_dropout} head_drop={resnet_head_dropout} "
            f"| clip={max_grad_norm} | aug noise/scale/ch_drop={aug_noise_std}/{aug_scale}/{aug_channel_dropout} "
            f"| device={device}"
        )
        print(
            "[info] best checkpoint, LR schedule, and early stopping use **test** MAE "
            "(test is no longer an unbiased final metric)."
        )

        best_test_mae = math.inf
        no_improve = 0
        global_step = 0
        last_ep = 0
        for ep in range(1, epochs + 1):
            tr_m, global_step = train_one_epoch(
                model,
                train_loader,
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
            te_m = eval_one_epoch(model, test_loader, device)
            scheduler.step(te_m["mae"])

            for k, v in tr_m.items():
                writer.add_scalar(f"epoch/train_{k}", v, ep)
            for k, v in va_m.items():
                writer.add_scalar(f"epoch/val_{k}", v, ep)
            for k, v in te_m.items():
                writer.add_scalar(f"epoch/test_{k}", v, ep)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], ep)

            if te_m["mae"] < best_test_mae - float(args.early_stop_min_delta):
                best_test_mae = te_m["mae"]
                no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": ep,
                        "model_type": args.model_type,
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
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1

            print(
                f"epoch {ep:03d} | "
                f"train MAE {tr_m['mae']:.4f} RMSE {tr_m['rmse']:.4f} R2 {tr_m['r2']:.4f} | "
                f"val MAE {va_m['mae']:.4f} RMSE {va_m['rmse']:.4f} R2 {va_m['r2']:.4f} | "
                f"test MAE {te_m['mae']:.4f} RMSE {te_m['rmse']:.4f} R2 {te_m['r2']:.4f} "
                f"P {te_m['pearson']:.4f} S {te_m['spearman']:.4f} | "
                f"best_test_MAE {best_test_mae:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )
            last_ep = ep
            if no_improve >= early_stop_patience:
                print(f"[early-stop] no test MAE improvement for {no_improve} epochs, stop at epoch {ep}.")
                break
    scatter_path = ckpt_path.parent / f"{ckpt_path.stem}_test_scatter.png"
    if not args.no_train_test_scatter and ckpt_path.is_file():
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(ckpt_path, map_location=device)
        scatter_dev = torch.device("cpu") if ck.get("backend") == "sklearn" else device
        if ck.get("backend") == "sklearn":
            from models.ml_method import SklearnRegressorModule

            model = SklearnRegressorModule(ck["estimator"], str(ck.get("sklearn_name", ck.get("model_type", "sklearn"))))
            model._fitted = True
            model = model.to(scatter_dev)
        else:
            model.load_state_dict(ck["model"])
        yt_te, yp_te = collect_predictions(model, test_loader, scatter_dev)
        band = _scatter_diag_band(yt_te, yp_te, args.scatter_band)
        save_test_scatter(
            yt_te,
            yp_te,
            scatter_path,
            band=band,
            axis_lo=0.0,
            axis_hi=4.0,
            writer=writer,
            writer_step=last_ep,
        )
        print(f"[done] test scatter: {scatter_path.resolve()}  (axes 0–4, band ±{band:.4f})")
    elif not args.no_train_test_scatter:
        print("[warn] no checkpoint file; skip train/test scatter.")

    writer.close()
    print(f"[done] best checkpoint: {ckpt_path.resolve()}")
    print(f"[done] tensorboard logdir: {logdir.resolve()}")


if __name__ == "__main__":
    main()
