"""A compact 1D CNN baseline for raw EEG windows."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.25),
        )


class SimpleEEGCNN(nn.Module):
    """Baseline mapping `[batch, 32, samples]` to two class logits."""

    def __init__(self, eeg_channels: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(eeg_channels, 64, kernel_size=7),
            ConvBlock(64, 128, kernel_size=5),
            ConvBlock(128, 128, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).squeeze(-1)
        return self.classifier(features)

