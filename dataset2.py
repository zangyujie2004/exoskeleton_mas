from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from vis import plot_multichannel


def _pad_to_10_channels(emg: np.ndarray) -> np.ndarray:
    arr = np.asarray(emg, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected emg_data shape (C, T), got {arr.shape}")
    c, t = arr.shape
    if c >= 10:
        return arr[:10]
    out = np.zeros((10, t), dtype=np.float32)
    out[:c] = arr
    return out


def main() -> None:
    pickle_path = "/data/zyj/project_mas/init_window_cache.pkl"
    with Path(pickle_path).open("rb") as f:
        obj = pickle.load(f)

    train_data = obj["train"]
    test_data = obj["test"]
    out_root = Path("/data/zyj/project_mas/window_vis")
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for idx, data in enumerate(train_data):
        emg = _pad_to_10_channels(data["emg_data"])
        period = int(data["period"])
        sid = int(data["subject_id"])
        eid = int(data["exp_id"])
        save_path = train_dir / f"s{sid}_exp{eid}_p{period}_idx{idx}.png"
        title = f"train idx={idx} | s{sid} exp{eid} period={period} | shape={tuple(emg.shape)}"
        print(f"[info] {title}")
        plot_multichannel(
            emg,
            Fs=2000,
            period=period + 1,
            save_path=str(save_path),
            suptitle=title,
            show=False,
        )

    for idx, data in enumerate(test_data):
        emg = _pad_to_10_channels(data["emg_data"])
        period = int(data["period"])
        sid = int(data["subject_id"])
        eid = int(data["exp_id"])
        save_path = test_dir / f"s{sid}_exp{eid}_p{period}_idx{idx}.png"
        title = f"test idx={idx} | s{sid} exp{eid} period={period} | shape={tuple(emg.shape)}"
        print(f"[info] {title}")
        plot_multichannel(
            emg,
            Fs=2000,
            period=period + 1,
            save_path=str(save_path),
            suptitle=title,
            show=False,
        )

    print(f"[done] train saved to: {train_dir}")
    print(f"[done] test saved to: {test_dir}")


if __name__ == "__main__":
    main()