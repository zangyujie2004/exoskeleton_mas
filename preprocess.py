from __future__ import annotations

import argparse
import pickle
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset import NUM_PERIODS, SampleMeta, build_sample_index, mat_to_period_arrays


RAW_FS = 4000
TARGET_FS = 1000
WINDOW_SECONDS = 4
OVERLAP_SECONDS = 3
WINDOW_SIZE = TARGET_FS * WINDOW_SECONDS
OVERLAP_SIZE = TARGET_FS * OVERLAP_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess init EMG: 4k->2k interpolation, session 8:2 split, sliding windows. "
            "Labels align with EMG via dataset.build_sample_index (Excel col0 = subject_id; "
            "exp_id is the next unused exp for that subject in ascending order, not read from Excel)."
        )
    )
    parser.add_argument("--init_data_root", type=str, default="data/init_data")
    parser.add_argument(
        "--label_xlsx",
        type=str,
        default="data/init_data/output.xlsx",
        help="MAS/spasm label xlsx. Used to assign each overlap window a spasm level.",
    )
    parser.add_argument(
        "--out_pickle",
        type=str,
        default="init_window_cache.pkl",
        help="Output pickle cache path.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--overlap_size", type=int, default=OVERLAP_SIZE)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check Excel row0 vs (subject_id,exp_id,mat_path) and label matrix vs mat periods.",
    )
    return parser.parse_args()


def verify_label_emg_alignment(metas: list[SampleMeta], label_xlsx: Path) -> None:
    """
    与 ``dataset.build_sample_index`` 规则一致地核对：

    - Excel ``label`` 表第 0 列 = 该行对应的 ``subject_id``
    - ``exp_id`` 与 ``mat_path`` 文件名一致（``exp{n}_processed_segment.mat``）
    - 标签矩阵与从 ``mat_path`` 读出的 8 个 period 一一对应（仅做形状与有限数值检查）
    """
    df = pd.read_excel(label_xlsx, sheet_name="label", header=None)
    if len(metas) != len(df):
        raise RuntimeError(f"meta count {len(metas)} != excel rows {len(df)}")

    for i, m in enumerate(metas):
        sid_xlsx = int(df.iloc[i, 0])
        if sid_xlsx != int(m.subject_id):
            raise RuntimeError(
                f"Row {i}: Excel col0 subject {sid_xlsx} != meta.subject_id {m.subject_id}"
            )
        mm = re.search(r"exp(\d+)_processed_segment\.mat$", m.mat_path.name)
        if not mm or int(mm.group(1)) != int(m.exp_id):
            raise RuntimeError(f"Row {i}: mat filename exp vs meta.exp_id mismatch: {m.mat_path.name} vs {m.exp_id}")
        if m.mat_path.parent.name != f"s{m.subject_id}_segment":
            raise RuntimeError(
                f"Row {i}: mat folder {m.mat_path.parent.name!r} != s{m.subject_id}_segment"
            )
        lab = df.iloc[i, 1 : NUM_PERIODS * 6 + 1].to_numpy(dtype=np.float32).reshape(NUM_PERIODS, 6)
        if not np.allclose(lab, np.asarray(m.label, dtype=np.float32), rtol=0, atol=1e-4):
            raise RuntimeError(f"Row {i}: label matrix from xlsx != meta.label")

        periods = mat_to_period_arrays(m.mat_path)
        if len(periods) != NUM_PERIODS:
            raise RuntimeError(f"Row {i}: expected {NUM_PERIODS} periods from mat, got {len(periods)}")
        for p, seg in enumerate(periods):
            if seg.ndim != 2 or seg.shape[0] < 1:
                raise RuntimeError(f"Row {i} period {p}: bad emg shape {seg.shape}")

    print(f"[verify] OK: {len(metas)} sessions — Excel col0 ↔ (subject_id, exp_id, mat_path) and labels match.")


def resample_4000_to_2000(emg: np.ndarray) -> np.ndarray:
    """Channel-wise linear interpolation from 4000 Hz to 2000 Hz."""
    arr = np.asarray(emg, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected (C, T), got {arr.shape}")
    c, t = arr.shape
    if t <= 1:
        return arr.copy()

    dst_t = max(1, int(round(t * (TARGET_FS / RAW_FS))))
    src_x = np.linspace(0.0, 1.0, t, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, dst_t, dtype=np.float32)
    out = np.empty((c, dst_t), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(dst_x, src_x, arr[i]).astype(np.float32)
    return out


def split_windows_with_overlap(emg: np.ndarray, window_size: int, overlap_size: int) -> list[tuple[np.ndarray, int]]:
    """
    Sliding window split with overlap.
    Returns [(window_emg, start_index), ...].
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if overlap_size < 0 or overlap_size >= window_size:
        raise ValueError(f"overlap_size must satisfy 0 <= overlap_size < window_size, got {overlap_size}")

    arr = np.asarray(emg, dtype=np.float32)
    c, t = arr.shape
    stride = window_size - overlap_size

    if t < window_size:
        padded = np.zeros((c, window_size), dtype=np.float32)
        padded[:, :t] = arr
        return [(padded, 0)]

    starts = list(range(0, t - window_size + 1, stride))
    last_start = t - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    out: list[tuple[np.ndarray, int]] = []
    for s in starts:
        out.append((arr[:, s : s + window_size].copy(), s))
    return out


def interpolate_window_spasm(period_labels6: np.ndarray, period_length: int, start_idx: int, window_size: int) -> float:
    """
    Convert 6 MAS/spasm labels in one period to a continuous timeline, then
    assign each overlap window a scalar label by window-center interpolation.
    """
    labels = np.asarray(period_labels6, dtype=np.float32).reshape(-1)
    if labels.size != 6:
        raise ValueError(f"Expected 6 labels for one period, got {labels.size}")
    if period_length <= 1:
        return float(labels.mean())

    # 6 labels correspond to 6 ordered stretches in the period.
    # We map them uniformly along the period timeline for interpolation.
    anchor_x = np.linspace(0.0, float(period_length - 1), num=6, dtype=np.float32)
    center_x = float(start_idx) + 0.5 * float(window_size - 1)
    y = np.interp(center_x, anchor_x, labels, left=float(labels[0]), right=float(labels[-1]))
    return float(y)


def build_split_records(
    session_items: list[tuple[int, int, Path, np.ndarray]],
    *,
    window_size: int,
    overlap_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subject_id, exp_id, mat_path, period_labels in session_items:
        periods = mat_to_period_arrays(mat_path)
        if len(periods) != NUM_PERIODS:
            raise ValueError(
                f"Expected {NUM_PERIODS} periods, got {len(periods)} for session s{subject_id}/exp{exp_id}"
            )
        if np.asarray(period_labels).shape != (NUM_PERIODS, 6):
            raise ValueError(
                f"Expected period labels shape ({NUM_PERIODS}, 6), got {np.asarray(period_labels).shape}"
            )

        for period_idx, period_emg in enumerate(periods):
            emg_2k = resample_4000_to_2000(period_emg)
            windows = split_windows_with_overlap(emg_2k, window_size, overlap_size)
            for window_emg, start_idx in windows:
                spasm_level = interpolate_window_spasm(
                    period_labels6=period_labels[period_idx],
                    period_length=int(emg_2k.shape[1]),
                    start_idx=int(start_idx),
                    window_size=int(window_size),
                )
                records.append(
                    {
                        "emg_data": window_emg,  # (C, 8000), float32
                        "period": int(period_idx),
                        "subject_id": int(subject_id),
                        "exp_id": int(exp_id),
                        "spasm_level": float(spasm_level),
                    }
                )
    return records


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {args.train_ratio}")

    root = Path(__file__).resolve().parent
    init_data_root = Path(args.init_data_root)
    if not init_data_root.is_absolute():
        init_data_root = root / init_data_root

    label_xlsx = Path(args.label_xlsx)
    if not label_xlsx.is_absolute():
        label_xlsx = root / label_xlsx
    if not label_xlsx.is_file():
        raise FileNotFoundError(f"label_xlsx not found: {label_xlsx}")

    out_pickle = Path(args.out_pickle)
    if not out_pickle.is_absolute():
        out_pickle = root / out_pickle
    out_pickle.parent.mkdir(parents=True, exist_ok=True)

    metas = build_sample_index(init_data_root=init_data_root, label_xlsx=label_xlsx)
    if args.verify:
        verify_label_emg_alignment(metas, label_xlsx)

    sessions: list[tuple[int, int, Path, np.ndarray]] = [
        (int(m.subject_id), int(m.exp_id), m.mat_path, np.asarray(m.label, dtype=np.float32)) for m in metas
    ]
    n_sessions = len(sessions)
    all_idx = list(range(n_sessions))
    rng = random.Random(args.seed)
    rng.shuffle(all_idx)

    n_train = max(1, int(round(n_sessions * args.train_ratio)))
    n_train = min(n_train, n_sessions - 1) if n_sessions > 1 else n_train
    train_idx = all_idx[:n_train]
    test_idx = all_idx[n_train:]
    train_sessions = [sessions[i] for i in train_idx]
    test_sessions = [sessions[i] for i in test_idx]

    train_records = build_split_records(
        train_sessions,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
    )
    test_records = build_split_records(
        test_sessions,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
    )

    cache_obj = {
        # data format:
        # {
        #   "train": [{"emg_data", "period", "subject_id", "exp_id"}, ...],
        #   "test":  [{"emg_data", "period", "subject_id", "exp_id"}, ...],
        # }
        "train": train_records,
        "test": test_records,
    }

    with out_pickle.open("wb") as f:
        pickle.dump(cache_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] sessions total/train/test: {n_sessions}/{len(train_idx)}/{len(test_idx)}")
    print(f"[done] windows train/test: {len(train_records)}/{len(test_records)}")
    print(f"[done] cache saved to: {out_pickle}")


if __name__ == "__main__":
    main()