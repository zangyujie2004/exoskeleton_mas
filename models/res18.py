
from __future__ import annotations
from models.cnn_base import build_resnet1d18

def build_resnet18_emg_2ch(
    *,
    out_dim: int = 1,
    base_width: int = 64,
    dropout: float = 0.1,
    head_dropout: float = 0.1,
):
    return build_resnet1d18(
        in_channels=2,
        out_dim=out_dim,
        base_width=base_width,
        dropout=dropout,
        head_dropout=head_dropout,
    )
