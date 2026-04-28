from __future__ import annotations

import torch
import torch.nn as nn

from typing import Any

from models.cnn_base import build_resnet1d18, build_resnet1d34


def build_model(
    model_type: str,
    chunk_length: int,
    max_chunks: int,
    hidden_dim: int,
    *,
    out_dim: int = 1,
    resnet_dropout: float = 0.1,
    resnet_head_dropout: float = 0.0,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int | None = None,
    transformer_dropout: float = 0.1,
    sklearn_cfg: dict[str, Any] | None = None,
    seed: int = 42,
) -> nn.Module:
    from models.ml_method import build_sklearn_from_cfg, is_sklearn_model_type

    if is_sklearn_model_type(model_type):
        return build_sklearn_from_cfg(model_type, sklearn_cfg or {}, random_state=int(seed))

    bw = hidden_dim // 4 if hidden_dim >= 64 else 64
    if model_type in ("resnet18", "resnet1d18", "res18"):
        return build_resnet1d18(
            in_channels=9,
            out_dim=out_dim,
            base_width=bw,
            dropout=resnet_dropout,
            head_dropout=resnet_head_dropout,
        )
    if model_type in ("resnet34", "resnet1d34", "res34"):
        return build_resnet1d34(
            in_channels=9,
            out_dim=out_dim,
            base_width=bw,
            dropout=resnet_dropout,
            head_dropout=resnet_head_dropout,
        )
    # Backward-compatible aliases: now route to 1D ResNet versions.
    if model_type in ("temporal_spatial_resnet18", "ts_resnet18", "ts18"):
        return build_resnet1d18(
            in_channels=9,
            out_dim=out_dim,
            base_width=bw,
            dropout=resnet_dropout,
            head_dropout=resnet_head_dropout,
        )
    if model_type in ("temporal_spatial_resnet34", "ts_resnet34", "ts34"):
        return build_resnet1d34(
            in_channels=9,
            out_dim=out_dim,
            base_width=bw,
            dropout=resnet_dropout,
            head_dropout=resnet_head_dropout,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
