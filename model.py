from __future__ import annotations

import torch
import torch.nn as nn


class ChunkTokenRegressor(nn.Module):
    """CNN per chunk + masked mean pool + linear（保留作对照或旧 checkpoint）。"""

    def __init__(
        self,
        chunk_length: int,
        in_channels: int = 9,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.chunk_length = chunk_length
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, m, c, l = x.shape
        u = x.reshape(b * m, c, l)
        h = self.encoder(u)
        z = self.time_pool(h).squeeze(-1).view(b, m, -1)
        w = mask.unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1e-6)
        pooled = (z * w).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


class ChunkCNNTransformerRegressor(nn.Module):
    """
    CNN 将每个时间 chunk 编码为 token 向量，再在 chunk 序列上做 Transformer，
    最后按 mask 做时序 token 聚合后回归标量（阶段 MAS 均值）。
    """

    def __init__(
        self,
        chunk_length: int,
        max_chunks: int,
        in_channels: int = 9,
        hidden_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})")
        self.chunk_length = chunk_length
        self.max_chunks = max_chunks
        self.hidden_dim = hidden_dim

        self.chunk_cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.time_pool = nn.AdaptiveAvgPool1d(1)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_chunks, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        d_ff = dim_feedforward if dim_feedforward is not None else 4 * hidden_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, C, L), mask: (B, M) float 1=valid chunk, 0=padding.
        """
        b, m, c, l = x.shape
        if m > self.max_chunks:
            raise ValueError(f"M={m} exceeds max_chunks={self.max_chunks}")

        u = x.reshape(b * m, c, l)
        tok = self.time_pool(self.chunk_cnn(u)).squeeze(-1)
        tok = tok.view(b, m, self.hidden_dim)
        tok = tok + self.pos_embed[:, :m, :]

        key_padding_mask = mask == 0
        h = self.transformer(tok, src_key_padding_mask=key_padding_mask)

        w = mask.unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1e-6)
        pooled = (h * w).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


def build_model(
    model_type: str,
    chunk_length: int,
    max_chunks: int,
    hidden_dim: int,
    *,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int | None = None,
    transformer_dropout: float = 0.1,
) -> nn.Module:
    if model_type in ("cnn_transformer", "cnn-transformer", "transformer"):
        return ChunkCNNTransformerRegressor(
            chunk_length=chunk_length,
            max_chunks=max_chunks,
            in_channels=9,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
        )
    if model_type in ("cnn_mlp", "cnn", "mlp"):
        return ChunkTokenRegressor(
            chunk_length=chunk_length,
            in_channels=9,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
