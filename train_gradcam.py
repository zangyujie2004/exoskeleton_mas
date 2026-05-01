"""
1D Grad-CAM for the 2-channel ResNet18 (``models/res18.py`` / ``resnet18_2ch``).

Loads a ``train_prediction2`` checkpoint, samples windows from the **train** split.
For each window, saves **one PNG per EMG channel**; only **raw EMG** is plotted as a
**segment-colored line** (Grad-CAM; default **research_blue**: 0 = light blue, 1 = dark blue). Horizontal
axis is **time in seconds** (default sampling rate 1000 Hz). Filenames include channel
id and spasm level.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize

from models.model import build_model


def make_research_blue_cmap() -> LinearSegmentedColormap:
    """Sequential blues: normalized CAM 0 → light blue, 1 → dark blue (no mid-band white)."""
    hex_stops = [
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#02101f",
    ]
    return LinearSegmentedColormap.from_list("research_blue", hex_stops, N=256)


def resolve_gradcam_cmap(key: str) -> Colormap:
    k = key.strip().lower().replace(" ", "_")
    if k in ("research_blue", "blue_seq", "pub_blue"):
        return make_research_blue_cmap()
    try:
        if hasattr(matplotlib, "colormaps"):
            return matplotlib.colormaps.get_cmap(key)
    except (ValueError, KeyError):
        pass
    return plt.get_cmap(key)


def setup_cjk_matplotlib() -> bool:
    """Return True if a CJK-capable sans font was selected (for Chinese titles)."""
    for family in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "SimHei",
        "WenQuanYi Micro Hei",
    ):
        try:
            path = fm.findfont(family, fallback_to_default=False)
        except (ValueError, OSError):
            continue
        if path and "dejavu" not in str(path).lower():
            plt.rcParams["font.sans-serif"] = [family] + list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


from train_prediction2 import (
    WindowSpasmDataset,
    collate_two_emg,
    load_yaml,
    parse_emg_channels,
    parse_period_filter,
    set_seed,
)


def grad_cam_1d_layer4(
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """
    Grad-CAM w.r.t. regression scalar output, target ``layer4`` activations.

    Returns
    -------
    cam_input_len : (T,) on CPU, non-negative, min-max normalized to [0, 1]
    pred : scalar prediction from the same forward pass
    """
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"Expected x shape (1, C, T), got {tuple(x.shape)}")
    if not hasattr(model, "layer4"):
        raise ValueError("Model has no layer4; use ResNet1D from cnn_base.")

    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []

    def fwd(_m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        acts.clear()
        acts.append(out)

    def full_bwd(_m: nn.Module, _gi: Any, go: tuple[torch.Tensor, ...]) -> None:
        grads.clear()
        grads.append(go[0])

    h1 = model.layer4.register_forward_hook(fwd)
    h2 = model.layer4.register_full_backward_hook(full_bwd)
    try:
        model.zero_grad(set_to_none=True)
        x_in = x.to(device, dtype=torch.float32, non_blocking=True)
        out = model(x_in)
        if out.ndim == 0:
            scalar = out
        else:
            scalar = out.squeeze()
        if scalar.numel() != 1:
            raise ValueError(f"Expected scalar prediction, got shape {tuple(out.shape)}")
        scalar.backward()
    finally:
        h1.remove()
        h2.remove()

    if not acts or not grads:
        raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

    a = acts[0][0]  # (C, L)
    g = grads[0][0]  # (C, L)
    # Classic Grad-CAM: channel weights = global average of gradient over length
    weights = g.mean(dim=1, keepdim=True)  # (C, 1)
    cam = (weights * a).sum(dim=0)  # (L,)
    cam = torch.relu(cam)
    cmin = float(cam.min().item())
    cmax = float(cam.max().item())
    if cmax - cmin < 1e-12:
        cam_n = torch.zeros_like(cam)
    else:
        cam_n = (cam - cmin) / (cmax - cmin)

    t_in = x.shape[-1]
    cam_up = torch.nn.functional.interpolate(
        cam_n.view(1, 1, -1),
        size=t_in,
        mode="linear",
        align_corners=False,
    ).view(-1)
    pred = float(out.squeeze().detach().cpu().item())
    return cam_up.detach().cpu(), pred


def _pick_indices_stratified(rows: list[dict[str, Any]], n_total: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_bin: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        y = float(r["spasm_level"])
        b = int(np.clip(round(y), 0, 4))
        by_bin[b].append(i)
    bins = sorted(by_bin.keys())
    if not bins:
        return []
    per = max(1, n_total // len(bins))
    out: list[int] = []
    for b in bins:
        idxs = by_bin[b][:]
        rng.shuffle(idxs)
        out.extend(idxs[:per])
    rng.shuffle(out)
    out = out[:n_total]
    if len(out) < n_total:
        rest = [i for i in range(len(rows)) if i not in set(out)]
        rng.shuffle(rest)
        out.extend(rest[: n_total - len(out)])
    return out[:n_total]


def _segments_xy(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = np.column_stack([np.asarray(t, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _add_cam_colored_line(
    ax,
    t: np.ndarray,
    y: np.ndarray,
    cam: np.ndarray,
    *,
    cmap: Colormap,
    lw: float,
) -> LineCollection:
    """Draw ``y`` vs ``t`` as segment-colored line (``cam`` in [0,1], per-segment = mean of endpoints)."""
    t = np.asarray(t, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    cam = np.asarray(cam, dtype=np.float64).ravel()
    if not (len(t) == len(y) == len(cam) and len(t) >= 2):
        raise ValueError("t, y, cam must have same length >= 2")
    segs = _segments_xy(t, y)
    cseg = 0.5 * (cam[:-1] + cam[1:])
    norm = Normalize(vmin=0.0, vmax=1.0)
    lc = LineCollection(segs, cmap=cmap, norm=norm, array=cseg, linewidths=lw, capstyle="round")
    ax.add_collection(lc)
    ax.set_xlim(float(t.min()), float(t.max()))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    pad = 0.05 * (y_max - y_min + 1e-9)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(True, alpha=0.25)
    return lc


def save_gradcam_single_channel_figure(
    *,
    raw: np.ndarray,
    cam: np.ndarray,
    spasm_y: float,
    pred: float,
    out_path: Path,
    ch_id: int,
    sample_rate_hz: float,
    cmap: Colormap,
    title_zh: bool = False,
    line_width: float = 1.25,
) -> None:
    """One panel: raw EMG only; x = time (s) at ``sample_rate_hz``; line color = Grad-CAM."""
    n = len(raw)
    t = np.arange(n, dtype=np.float64) / float(sample_rate_hz)

    fig, ax0 = plt.subplots(1, 1, figsize=(10.0, 3.6), dpi=150, constrained_layout=True)

    _add_cam_colored_line(ax0, t, raw, cam, cmap=cmap, lw=line_width)
    ax0.set_ylabel(f"EMG c{ch_id} (raw)", fontsize=10)
    ax0.set_xlabel(f"Time (s)  @  {sample_rate_hz:g} Hz", fontsize=10)

    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax0, fraction=0.035, pad=0.02)
    cbar.set_label("Grad-CAM (norm.)", fontsize=9)

    if title_zh:
        fig.suptitle(
            f"痉挛等级（真值）: {spasm_y:.3f}  |  模型预测: {pred:.3f}  |  通道 c{ch_id}",
            fontsize=11,
            fontweight="bold",
        )
    else:
        fig.suptitle(
            f"Spasm level (ground truth): {spasm_y:.3f}  |  prediction: {pred:.3f}  |  channel c{ch_id}",
            fontsize=11,
            fontweight="bold",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1D Grad-CAM for 2-ch ResNet18 on train windows.")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--pickle_path", type=str, default="init_window_cache.pkl")
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/resnet_spasm_best.pt")
    p.add_argument("--out_dir", type=str, default="window_vis/gradcam_train")
    p.add_argument("--num_samples", type=int, default=1000, help="Number of windows to export (stratified by rounded label 0–4).")
    p.add_argument("--periods", type=str, default="0,1,7", help="Same as train_prediction2 (or 'all').")
    p.add_argument("--emg_channels", type=str, default="0,6", help="Exactly two indices; must match checkpoint for 2-ch models.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (default: config or cuda if available).")
    p.add_argument(
        "--sample_rate_hz",
        type=float,
        default=1000.0,
        help="Assumed EMG sampling rate (Hz); time axis = sample_index / rate.",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="research_blue",
        help="Colormap for Grad-CAM along the line. Default: research_blue (0=light blue, 1=dark blue). "
        "Also: blue_seq, pub_blue (aliases), or any matplotlib name (e.g. Blues, viridis).",
    )
    p.add_argument("--line_width", type=float, default=2.0, help="Linewidth for colored EMG traces.")
    p.add_argument(
        "--title_zh",
        action="store_true",
        help="Use Chinese suptitle (requires a CJK font, e.g. Noto Sans CJK SC); otherwise English.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    set_seed(int(args.seed))
    title_zh = bool(args.title_zh) and setup_cjk_matplotlib()
    if args.title_zh and not title_zh:
        print("[warn] --title_zh set but no CJK font found; using English suptitle.")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}

    pickle_path = Path(args.pickle_path)
    if not pickle_path.is_absolute():
        pickle_path = root / pickle_path

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    period_filter = parse_period_filter(args.periods)
    emg_set = parse_emg_channels(args.emg_channels)
    if emg_set is None or len(emg_set) != 2:
        raise ValueError("--emg_channels must specify exactly two indices (default 0,6) for this script.")
    emg_slice = tuple(sorted(emg_set))
    ch0, ch1 = int(emg_slice[0]), int(emg_slice[1])

    ck: dict[str, Any]
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    if ck.get("backend") == "sklearn":
        raise ValueError("Grad-CAM requires a PyTorch checkpoint (not sklearn).")

    model_type = str(ck.get("model_type", "resnet18_2ch"))
    hidden_dim = int(ck.get("hidden_dim", cfg.get("hidden_dim", 256)))
    resnet_dropout = float(ck.get("resnet_dropout", cfg.get("resnet_dropout", 0.15)))
    resnet_head_dropout = float(ck.get("resnet_head_dropout", cfg.get("resnet_head_dropout", 0.15)))
    ck_si = ck.get("emg_slice_indices")
    if ck_si is not None:
        ck_slice = tuple(int(i) for i in ck_si)
        if ck_slice != emg_slice:
            print(f"[warn] checkpoint emg_slice_indices {ck_slice} != CLI {emg_slice}; using CLI slice for plots/raw.")

    model = build_model(
        model_type=model_type,
        chunk_length=0,
        max_chunks=0,
        hidden_dim=hidden_dim,
        resnet_dropout=resnet_dropout,
        resnet_head_dropout=resnet_head_dropout,
        sklearn_cfg=cfg,
        seed=int(args.seed),
    )
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    device_s = args.device or str(cfg.get("device", "cuda"))
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds = WindowSpasmDataset(pickle_path, split="train", period_indices=period_filter)
    collate = collate_two_emg(emg_slice)
    indices = _pick_indices_stratified(ds.rows, int(args.num_samples), int(args.seed))

    print(f"[info] checkpoint={ckpt_path} | model_type={model_type} | emg_slice={emg_slice} | n={len(indices)} -> {out_dir}")

    cmap_resolved = resolve_gradcam_cmap(str(args.cmap))
    for k, idx in enumerate(indices):
        batch = [ds[idx]]
        x2, y = collate(batch)
        spasm_y = float(y.item())
        cam, pred = grad_cam_1d_layer4(model, x2, device=device)
        cam_np = cam.numpy()

        # Raw physical channels from original 9-ch window (aligned with model input order)
        x9 = batch[0]["x"].numpy()
        raw0 = x9[ch0].astype(np.float64, copy=False)
        raw6 = x9[ch1].astype(np.float64, copy=False)

        sid = int(batch[0]["subject_id"])
        eid = int(batch[0]["exp_id"])
        per = int(batch[0]["period"])
        stem = f"gradcam_train_i{idx:05d}_sub{sid}_exp{eid}_p{per}_spasm{spasm_y:.2f}"
        for ch, raw in ((ch0, raw0), (ch1, raw6)):
            fname = f"{stem}_c{ch}.png"
            save_gradcam_single_channel_figure(
                raw=raw,
                cam=cam_np,
                spasm_y=spasm_y,
                pred=pred,
                out_path=out_dir / fname,
                ch_id=ch,
                sample_rate_hz=float(args.sample_rate_hz),
                cmap=cmap_resolved,
                title_zh=title_zh,
                line_width=float(args.line_width),
            )
            print(f"  saved {fname}")

    print("[done]")


if __name__ == "__main__":
    main()
