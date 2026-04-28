from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from dataset import InitSegmentDataset, NUM_EMG_CHANNELS, PickledInitSegmentDataset

PERIOD_NAMES: list[str] = [
    "wrist",
    "four_fingers",
    "thumb",
    "index_finger",
    "middle_finger",
    "ring_finger",
    "little_finger",
    "static_stretch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use t-SNE to visualize 8 EMG periods in a 2D space."
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        default="init_segment_cache.pkl",
        help="Cached pickle built by dataset.save_init_segment_pickle.",
    )
    parser.add_argument(
        "--init_data_root",
        type=str,
        default="data/init_data",
        help="Raw init_data root, used when pickle is missing.",
    )
    parser.add_argument(
        "--label_xlsx",
        type=str,
        default=None,
        help="Optional label xlsx path for raw loading.",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=2048,
        help="Interpolate each period signal to this time length.",
    )
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=100,
        help="Limit number of sessions for speed. 0 means all sessions.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=50.0,
        help="Requested t-SNE perplexity (auto-clipped for small datasets).",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=2048,
        help="PCA components before t-SNE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/tsne",
        help="Output directory for figure and embedding csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Saved figure dpi.",
    )
    return parser.parse_args()


def to_numpy_period(period: object) -> np.ndarray:
    if hasattr(period, "detach"):
        period = period.detach().cpu().numpy()
    arr = np.asarray(period, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected period shape (C, T), got {arr.shape}")
    return arr


def resize_period(period: np.ndarray, target_length: int) -> np.ndarray:
    c, t = period.shape
    if target_length <= 0:
        raise ValueError(f"target_length must be > 0, got {target_length}")
    if t == target_length:
        return period.astype(np.float32, copy=False)
    if t == 1:
        return np.repeat(period, target_length, axis=1).astype(np.float32, copy=False)

    src_x = np.linspace(0.0, 1.0, t, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    out = np.empty((c, target_length), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(dst_x, src_x, period[i]).astype(np.float32)
    return out


def load_dataset(pickle_path: Path, init_data_root: Path, label_xlsx: str | None):
    if pickle_path.is_file():
        print(f"[info] Loading pickled dataset: {pickle_path}")
        return PickledInitSegmentDataset(pickle_path, return_torch=False)
    print(f"[info] Pickle not found, loading raw data: {init_data_root}")
    return InitSegmentDataset(
        init_data_root=init_data_root,
        label_xlsx=label_xlsx,
        return_torch=False,
    )


def build_feature_matrix(
    ds,
    target_length: int,
    max_sessions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    n_sessions = len(ds)
    if max_sessions > 0:
        n_sessions = min(n_sessions, max_sessions)

    features: list[np.ndarray] = []
    stage_ids: list[int] = []
    channel_ids: list[int] = []
    subject_ids: list[int] = []
    metas: list[tuple[int, int, int, int]] = []

    for idx in range(n_sessions):
        row = ds[idx]
        sid = int(row["subject_id"])
        eid = int(row["exp_id"])
        periods = row["periods"]
        if len(periods) != 8:
            raise ValueError(f"Expected 8 periods, got {len(periods)} for session {idx}")
        for stage, period in enumerate(periods):
            arr = to_numpy_period(period)
            arr = resize_period(arr, target_length)
            if arr.shape[0] < NUM_EMG_CHANNELS:
                raise ValueError(f"Expected at least {NUM_EMG_CHANNELS} channels, got {arr.shape[0]}")
            arr = arr[:NUM_EMG_CHANNELS]
            for ch in range(NUM_EMG_CHANNELS):
                features.append(arr[ch])
                stage_ids.append(stage)
                channel_ids.append(ch)
                subject_ids.append(sid)
                metas.append((sid, eid, stage, ch))

    x = np.vstack(features).astype(np.float32)
    stage_arr = np.asarray(stage_ids, dtype=np.int64)
    channel_arr = np.asarray(channel_ids, dtype=np.int64)
    subject_arr = np.asarray(subject_ids, dtype=np.int64)
    return x, stage_arr, channel_arr, subject_arr, metas


def run_tsne(
    features: np.ndarray,
    perplexity: float,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    if features.shape[0] < 4:
        raise ValueError("Need at least 4 samples to run t-SNE.")

    scaler = StandardScaler()
    x = scaler.fit_transform(features)

    pca_k = min(pca_dim, x.shape[1], max(2, x.shape[0] - 1))
    x_pca = PCA(n_components=pca_k, random_state=seed).fit_transform(x)

    max_perplexity = max(2.0, (x_pca.shape[0] - 1) / 3.0)
    p = min(perplexity, max_perplexity)
    if p < 2.0:
        p = 2.0
    print(f"[info] Running t-SNE with perplexity={p:.2f}, pca_dim={pca_k}")

    emb = TSNE(
        n_components=2,
        perplexity=p,
        init="pca",
        learning_rate="auto",
        random_state=seed
    ).fit_transform(x_pca)
    return emb.astype(np.float32)


def save_embedding_csv(
    emb: np.ndarray,
    stage_ids: np.ndarray,
    channel_ids: np.ndarray,
    subject_ids: np.ndarray,
    metas: list[tuple[int, int, int, int]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x",
                "y",
                "stage_id",
                "stage_name",
                "channel_id",
                "channel_name",
                "subject_id",
                "exp_id",
            ]
        )
        for i in range(emb.shape[0]):
            sid, eid, stage, ch = metas[i]
            writer.writerow(
                [
                    float(emb[i, 0]),
                    float(emb[i, 1]),
                    int(stage_ids[i]),
                    PERIOD_NAMES[stage],
                    int(channel_ids[i]),
                    f"ch_{int(channel_ids[i])}",
                    int(subject_ids[i]),
                    eid,
                ]
            )


def plot_embedding(
    emb: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
    title: str,
    out_path: Path,
    dpi: int,
    legend_ncol: int = 2,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    uniq = sorted(np.unique(label_ids).tolist())
    cmap = plt.get_cmap("tab20", max(20, len(uniq)))
    for i, lb in enumerate(uniq):
        mask = label_ids == lb
        if not np.any(mask):
            continue
        name = label_names[lb] if 0 <= lb < len(label_names) else str(lb)
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=18,
            alpha=0.70,
            color=cmap(i),
            label=f"{lb}: {name}",
        )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=8, ncol=legend_ncol)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _safe_global_cluster_metrics(emb: np.ndarray, label_ids: np.ndarray) -> dict[str, float]:
    uniq = np.unique(label_ids)
    out: dict[str, float] = {
        "n_samples": float(emb.shape[0]),
        "n_clusters": float(uniq.size),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }
    if uniq.size < 2 or uniq.size >= emb.shape[0]:
        return out

    counts = np.bincount(label_ids.astype(np.int64))
    has_singleton = np.any(counts[counts > 0] < 2)

    if not has_singleton:
        out["silhouette"] = float(silhouette_score(emb, label_ids))
    out["davies_bouldin"] = float(davies_bouldin_score(emb, label_ids))
    out["calinski_harabasz"] = float(calinski_harabasz_score(emb, label_ids))
    return out


def _cluster_distribution_rows(
    emb: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    uniq = sorted(np.unique(label_ids).tolist())

    centroids: dict[int, np.ndarray] = {}
    for lb in uniq:
        pts = emb[label_ids == lb]
        centroids[lb] = pts.mean(axis=0)

    for lb in uniq:
        pts = emb[label_ids == lb]
        centroid = centroids[lb]
        diffs = pts - centroid
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        cov = np.cov(pts.T) if pts.shape[0] >= 2 else np.zeros((2, 2), dtype=np.float64)
        spread_trace = float(np.trace(cov))
        spread_det = float(np.linalg.det(cov))

        intra_pair_mean = float("nan")
        if pts.shape[0] > 1:
            diff_mat = pts[:, None, :] - pts[None, :, :]
            dmat = np.sqrt(np.sum(diff_mat * diff_mat, axis=2))
            iu = np.triu_indices(pts.shape[0], k=1)
            intra_pair_mean = float(dmat[iu].mean())

        nearest_centroid_dist = float("nan")
        other_dists = []
        for other_lb in uniq:
            if other_lb == lb:
                continue
            other_dists.append(float(np.linalg.norm(centroid - centroids[other_lb])))
        if other_dists:
            nearest_centroid_dist = float(min(other_dists))

        name = label_names[lb] if 0 <= lb < len(label_names) else str(lb)
        rows.append(
            {
                "label_id": int(lb),
                "label_name": name,
                "n_samples": int(pts.shape[0]),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "radius_mean": float(dists.mean()),
                "radius_std": float(dists.std()),
                "radius_rms": float(np.sqrt(np.mean(dists * dists))),
                "radius_p90": float(np.percentile(dists, 90)),
                "intra_pair_mean_dist": intra_pair_mean,
                "spread_trace": spread_trace,
                "spread_det": spread_det,
                "nearest_centroid_dist": nearest_centroid_dist,
                "separation_compactness_ratio": (
                    nearest_centroid_dist / float(dists.mean()) if float(dists.mean()) > 1e-12 else float("nan")
                ),
            }
        )
    return rows


def _save_rows_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_global_metrics_csv(metrics: dict[str, float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def save_distribution_metrics(
    emb: np.ndarray,
    stage_ids: np.ndarray,
    channel_ids: np.ndarray,
    subject_ids: np.ndarray,
    out_dir: Path,
) -> None:
    channel_names = [f"ch_{i}" for i in range(NUM_EMG_CHANNELS)]
    uniq_subjects = sorted(np.unique(subject_ids).tolist())
    max_sub = max(uniq_subjects) if uniq_subjects else 0
    subject_names = [f"s{i}" for i in range(max_sub + 1)]

    by_period_rows = _cluster_distribution_rows(emb, stage_ids, PERIOD_NAMES)
    by_channel_rows = _cluster_distribution_rows(emb, channel_ids, channel_names)
    by_subject_rows = _cluster_distribution_rows(emb, subject_ids, subject_names)

    _save_rows_csv(by_period_rows, out_dir / "metrics_by_period.csv")
    _save_rows_csv(by_channel_rows, out_dir / "metrics_by_channel.csv")
    _save_rows_csv(by_subject_rows, out_dir / "metrics_by_subject.csv")

    _save_global_metrics_csv(
        _safe_global_cluster_metrics(emb, stage_ids),
        out_dir / "metrics_global_period.csv",
    )
    _save_global_metrics_csv(
        _safe_global_cluster_metrics(emb, channel_ids),
        out_dir / "metrics_global_channel.csv",
    )
    _save_global_metrics_csv(
        _safe_global_cluster_metrics(emb, subject_ids),
        out_dir / "metrics_global_subject.csv",
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    root = Path(__file__).resolve().parent
    pickle_path = Path(args.pickle_path)
    if not pickle_path.is_absolute():
        pickle_path = root / pickle_path

    init_data_root = Path(args.init_data_root)
    if not init_data_root.is_absolute():
        init_data_root = root / init_data_root

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(pickle_path, init_data_root, args.label_xlsx)
    x, stage_ids, channel_ids, subject_ids, metas = build_feature_matrix(
        ds,
        target_length=args.target_length,
        max_sessions=args.max_sessions,
    )
    print(f"[info] Feature matrix shape: {x.shape}")
    emb = run_tsne(
        x,
        perplexity=args.perplexity,
        pca_dim=args.pca_dim,
        seed=args.seed,
    )

    fig_period = out_dir / "tsne_by_period.png"
    fig_channel = out_dir / "tsne_by_channel.png"
    fig_subject = out_dir / "tsne_by_subject.png"
    csv_path = out_dir / "tsne_channel_decoupled.csv"

    plot_embedding(
        emb,
        stage_ids,
        PERIOD_NAMES,
        title="t-SNE (channel-decoupled) colored by period label",
        out_path=fig_period,
        dpi=args.dpi,
        legend_ncol=2,
    )
    channel_names = [f"ch_{i}" for i in range(NUM_EMG_CHANNELS)]
    plot_embedding(
        emb,
        channel_ids,
        channel_names,
        title="t-SNE (channel-decoupled) colored by channel label",
        out_path=fig_channel,
        dpi=args.dpi,
        legend_ncol=3,
    )
    uniq_subjects = sorted(np.unique(subject_ids).tolist())
    max_sub = max(uniq_subjects) if uniq_subjects else 0
    subject_names = [f"s{i}" for i in range(max_sub + 1)]
    legend_col = 3 if len(uniq_subjects) > 10 else 2
    plot_embedding(
        emb,
        subject_ids,
        subject_names,
        title="t-SNE (channel-decoupled) colored by subject label",
        out_path=fig_subject,
        dpi=args.dpi,
        legend_ncol=legend_col,
    )
    save_embedding_csv(emb, stage_ids, channel_ids, subject_ids, metas, csv_path)
    save_distribution_metrics(emb, stage_ids, channel_ids, subject_ids, out_dir)

    print(f"[done] Figure saved to: {fig_period}")
    print(f"[done] Figure saved to: {fig_channel}")
    print(f"[done] Figure saved to: {fig_subject}")
    print(f"[done] Embedding csv saved to: {csv_path}")
    print(f"[done] Distribution metrics saved under: {out_dir}")


if __name__ == "__main__":
    main()
