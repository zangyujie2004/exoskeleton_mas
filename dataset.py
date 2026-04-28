from __future__ import annotations

import copy
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import scipy.io as sio

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None  # type: ignore[misc, assignment]
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
    c, _t = seg.shape
    if c < NUM_EMG_CHANNELS:
        raise ValueError(f"Expected at least {NUM_EMG_CHANNELS} channels, got {c}")
    return seg[:NUM_EMG_CHANNELS, :].astype(np.float32, copy=False)


def mat_to_period_arrays(mat_path: Path | str) -> List[np.ndarray]:
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
    c, t = emg.shape
    if t < n_parts:
        pad = np.zeros((c, n_parts), dtype=np.float32)
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
    label: np.ndarray


def build_sample_index(
    init_data_root: Path | str | None = None,
    label_xlsx: Path | str | None = None,
) -> List[SampleMeta]:
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
        raise RuntimeError(f"Unused experiments remain (check Excel vs folders): {remainder!r}")

    return metas


class InitSegmentDataset(Dataset):
    """One row per session: 8 concatenated-period EMG tensors + labels from Excel."""

    def __init__(
        self,
        init_data_root: Path | str | None = None,
        label_xlsx: Path | str | None = None,
        *,
        return_torch: bool = True,
    ) -> None:
        self.init_data_root = Path(init_data_root or _default_init_data_root())
        self.label_xlsx = label_xlsx
        self.return_torch = bool(return_torch)
        self.metas = build_sample_index(self.init_data_root, label_xlsx)

    def __len__(self) -> int:
        return len(self.metas)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.metas[idx]
        periods_np = mat_to_period_arrays(meta.mat_path)
        if self.return_torch and torch is not None:
            periods: Any = [torch.from_numpy(p.copy()) for p in periods_np]
        else:
            periods = [p.copy() for p in periods_np]
        return {
            "subject_id": int(meta.subject_id),
            "exp_id": int(meta.exp_id),
            "periods": periods,
        }


class PickledInitSegmentDataset(Dataset):
    """Loads ``save_init_segment_pickle`` output: list of dicts with ``subject_id``, ``exp_id``, ``periods``."""

    def __init__(self, pickle_path: Path | str, *, return_torch: bool = True) -> None:
        self.pickle_path = Path(pickle_path)
        with self.pickle_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"Expected list pickle from save_init_segment_pickle, got {type(obj)}")
        self.rows = obj
        self.return_torch = bool(return_torch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.rows[idx])
        periods = list(row["periods"])
        if self.return_torch and torch is not None:
            row["periods"] = [
                p if hasattr(p, "detach") else torch.from_numpy(np.asarray(p, dtype=np.float32)) for p in periods
            ]
        else:
            row["periods"] = [
                (p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p, dtype=np.float32)).astype(
                    np.float32, copy=False
                )
                for p in periods
            ]
        return row


def save_init_segment_pickle(
    out_path: Path | str,
    init_data_root: Path | str | None = None,
    label_xlsx: Path | str | None = None,
    *,
    return_torch: bool = False,
) -> None:
    """Materialize ``InitSegmentDataset`` to a pickle list for ``PickledInitSegmentDataset``."""
    ds = InitSegmentDataset(init_data_root=init_data_root, label_xlsx=label_xlsx, return_torch=return_torch)
    rows = [ds[i] for i in range(len(ds))]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
