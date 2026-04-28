from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _to_numpy_2d(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (C, T), got shape {arr.shape}")
    return arr


def plot_single_channel(emg, Fs=2000, title="EMG Signal", figsize=(12, 3), save_path=None):
    T = len(emg)
    t = np.arange(T) / Fs
    plt.figure(figsize=figsize)
    plt.plot(t, emg, color='b', linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("EMG")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    plt.show()

def plot_multichannel(X, Fs=4000, period=1, save_path=None, suptitle=None, *, show: bool = True):
    X = _to_numpy_2d(X)
    M, T = X.shape
    t = np.arange(T) / Fs

    fig, axes = plt.subplots(5, 2, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(M):
        axes[i].plot(t, X[i], color="b", linewidth=1)
        axes[i].set_title(f"Channel {i}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("EMG")
        axes[i].grid(True)

    for j in range(M, 10):
        axes[j].axis("off")

    plt.suptitle(suptitle if suptitle is not None else f"Period {period}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_init_sample(sample: dict[str, Any], period_index: int = 0, Fs: float = 2000, save_path=None) -> None:
    periods = sample["periods"]
    if period_index < 0 or period_index >= len(periods):
        raise IndexError(f"period_index {period_index} out of range [0, {len(periods)})")

    X = _to_numpy_2d(periods[period_index])
    lab = sample["label"]
    if hasattr(lab, "detach"):
        label = lab.detach().cpu().numpy()
    else:
        label = np.asarray(lab, dtype=np.float32)
    if label.ndim != 2 or label.shape != (8, 6):
        raise ValueError(f"Expected label shape (8, 6), got {label.shape}")

    subj = int(sample["subject_id"])
    exp = int(sample["exp_id"])
    mas_row = np.array2string(label[period_index], precision=2, separator=", ")
    sup = f"s{subj} exp{exp} — period {period_index + 1} / 8 | MAS: {mas_row}"

    plot_multichannel(X, Fs=Fs, period=period_index + 1, save_path=save_path, suptitle=sup, show=True)