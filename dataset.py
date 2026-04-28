from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore[misc, assignment]

NUM_PERIODS = 8
SEGS_PER_PERIOD = 6
NUM_LABELS = NUM_PERIODS * SEGS_PER_PERIOD
NUM_EMG_CHANNELS = 9


def _default_init_data_root() -> Path:
    return Path(__file__).resolve().parent / "data" / "init_data"


def _subject_experiments(root: Path) -> dict[int, list[int]]:
    """``{subject_id: sorted [exp_id, …]}`` for every ``s{n}_segment`` folder."""
    folders: dict[int, list[int]] = {}
    for name in root.iterdir():
        if not name.is_dir():
            continue
        m = re.match(r"s(\d+)_segment$", name.name)
        if not m:
            continue
        sid = int(m.group(1))
        exps: list[int] = []
        for mat in name.glob("exp*_processed_segment.mat"):
            mm = re.search(r"exp(\d+)_processed", mat.name)
            if mm:
                exps.append(int(mm.group(1)))
        folders[sid] = sorted(exps)
    return folders


def to_nine_channels(seg: np.ndarray) -> np.ndarray:
    """``seg`` is ``(C, T)`` after transpose; keep nine sEMG channels only (no zero padding)."""
    c, _t = seg.shape
    if c < NUM_EMG_CHANNELS:
        raise ValueError(f"Expected at least {NUM_EMG_CHANNELS} channels, got {c}")
    return seg[:NUM_EMG_CHANNELS, :].astype(np.float32, copy=False)


def mat_to_period_arrays(mat_path: Path | str) -> List[np.ndarray]:
    """
    Load one ``*_processed_segment.mat`` and return eight ``float32`` arrays of shape
    ``(9, T_i)`` (time lengths differ per period).
    """
    mat_path = Path(mat_path)
    raw = sio.loadmat(mat_path.as_posix())
    segments: List[np.ndarray] = []
    for i in range(1, NUM_LABELS + 1):
        key = f"data_segment_{i}"
        if key not in raw:
            raise KeyError(f"{mat_path}: missing {key}")
        seg = np.asarray(raw[key], dtype=np.float32).T  # (T, C) -> (C, T)
        segments.append(to_nine_channels(seg))

    periods: List[np.ndarray] = []
    for p in range(NUM_PERIODS):
        start = p * SEGS_PER_PERIOD
        chunk = segments[start : start + SEGS_PER_PERIOD]
        periods.append(np.hstack(chunk))
    return periods


def split_period_emg_equal_parts(emg: np.ndarray, n_parts: int = SEGS_PER_PERIOD) -> List[np.ndarray]:
    """
    将单个治疗阶段整段肌电 ``(9, T)`` 沿时间**均分为 n_parts 份**（长度尽可能相等，余数由 ``numpy.array_split`` 分配），
    与同一阶段内 ``n_parts`` 个绝对 MAS 标签顺序对齐。

    注意：训练里的 ``chunk_length`` / ``max_chunks`` **不参与**本函数；它们只在之后对**每一段子肌电**单独做 token 化时使用。
    """
    c, t = emg.shape
    if t < n_parts:
        pad = np.zeros((c, n_parts - t), dtype=np.float32)
        emg = np.concatenate([emg.astype(np.float32, copy=False), pad], axis=1)
        t = n_parts
    ranges = np.array_split(np.arange(t, dtype=np.int64), n_parts)
    emg_f = emg.astype(np.float32, copy=False)
    return [emg_f[:, r].copy() for r in ranges]


@dataclass(frozen=True)
class SampleMeta:
    subject_id: int
    exp_id: int
    mat_path: Path
    label: np.ndarray  # (8, 6), float32


def build_sample_index(
    init_data_root: Path | str | None = None,
    label_xlsx: Path | str | None = None,
) -> List[SampleMeta]:
    """
    Pair every label row with the correct ``.mat`` using the FIFO rule described in the
    module docstring.
    """
    root = Path(init_data_root or _default_init_data_root())
    xlsx = Path(label_xlsx or (root / "output.xlsx"))
    df = pd.read_excel(xlsx, sheet_name="label", header=None)
    if df.shape[1] < NUM_LABELS + 1:
        raise ValueError(f"Expected ≥{NUM_LABELS + 1} columns in label sheet, got {df.shape[1]}")

    exps_by_subject = _subject_experiments(root)
    queues: dict[int, list[int]] = {k: copy.copy(v) for k, v in exps_by_subject.items()}
    metas: List[SampleMeta] = []

    for i in range(len(df)):
        sid = int(df.iloc[i, 0])
        if sid not in queues or not queues[sid]:
            raise RuntimeError(f"Label row {i}: no remaining experiments for subject {sid}")
        exp_id = queues[sid].pop(0)
        mat_path = root / f"s{sid}_segment" / f"exp{exp_id}_processed_segment.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        lab = df.iloc[i, 1 : NUM_LABELS + 1].to_numpy(dtype=np.float32).reshape(NUM_PERIODS, SEGS_PER_PERIOD)
        metas.append(SampleMeta(subject_id=sid, exp_id=exp_id, mat_path=mat_path, label=lab))

    remainder = {k: v for k, v in queues.items() if v}
    if remainder:
        raise RuntimeError(f"Some segment files were not assigned a label row: {remainder}")
    return metas


class InitSegmentDataset(Dataset):  # type: ignore[misc]
    """``__getitem__`` → ``dict`` with ``periods``, ``label``, ``subject_id``, ``exp_id``."""

    def __init__(
        self,
        init_data_root: Path | str | None = None,
        label_xlsx: Path | str | None = None,
        return_torch: bool = True,
    ) -> None:
        if return_torch and torch is None:
            raise ImportError("PyTorch is required when return_torch=True")
        self._return_torch = return_torch
        self.samples = build_sample_index(init_data_root, label_xlsx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        meta = self.samples[index]
        periods = mat_to_period_arrays(meta.mat_path)
        label = meta.label.copy()
        if self._return_torch:
            periods = [torch.from_numpy(p) for p in periods]
            label = torch.from_numpy(label)
        return {
            "periods": periods,
            "label": label,
            "subject_id": meta.subject_id,
            "exp_id": meta.exp_id,
            "path": str(meta.mat_path),
        }


def iter_metas(init_data_root: Path | str | None = None) -> Iterator[SampleMeta]:
    yield from build_sample_index(init_data_root)


# test
if __name__ == "__main__":
    path = "/data/zyj/project_mas/data/init_data"
    dataset = InitSegmentDataset(init_data_root=path)
    print(len(dataset))
    data_sample = dataset[0]
    print(data_sample.keys())
    print("channels", data_sample["periods"][0].shape[0], "n_periods", len(data_sample["periods"]))
    # 腕（wrist）
    # 四指（four fingers）
    # 拇指（thumb）
    # 食指（index）
    # 中指（middle）
    # 无名指（ring）
    # 小指（little finger）
    # 静态牵伸
    # FDS, FCU, FPL, FCR, EDM, EDC, ECU, EPB, ECR
    print(data_sample["label"].shape)
