from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        out = out + identity
        out = self.act(out)
        return out


class ResNet1DBase(nn.Module):
    """1D ResNet for EMG ``(B, 9, T)``: stacked convs + BN + pool + linear head only — no self-attention (low latency)."""

    def __init__(
        self,
        in_channels: int = 9,
        base_width: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.inplanes = base_width
        hd = float(head_dropout)
        self.head_dropout = nn.Dropout(hd) if hd > 0 else nn.Identity()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_width, layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = base_width * 8
        self.head = nn.Linear(self.feature_dim, out_dim)

    def _make_layer(self, out_ch: int, blocks: int, stride: int, dropout: float) -> nn.Sequential:
        layers = [BasicBlock1D(self.inplanes, out_ch, stride=stride, dropout=dropout)]
        self.inplanes = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, out_ch, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}")
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        feat = self.head_dropout(feat)
        out = self.head(feat)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def build_resnet1d18(
    in_channels: int = 9,
    out_dim: int = 1,
    base_width: int = 64,
    dropout: float = 0.1,
    head_dropout: float = 0.0,
) -> ResNet1DBase:
    return ResNet1DBase(
        in_channels=in_channels,
        base_width=base_width,
        layers=(2, 2, 2, 2),
        dropout=dropout,
        head_dropout=head_dropout,
        out_dim=out_dim,
    )


def build_resnet1d34(
    in_channels: int = 9,
    out_dim: int = 1,
    base_width: int = 64,
    dropout: float = 0.1,
    head_dropout: float = 0.0,
) -> ResNet1DBase:
    return ResNet1DBase(
        in_channels=in_channels,
        base_width=base_width,
        layers=(3, 4, 6, 3),
        dropout=dropout,
        head_dropout=head_dropout,
        out_dim=out_dim,
    )
