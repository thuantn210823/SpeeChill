"""
Adapter Module for Speech-LLM Integration

Handles dimension transformation between audio encoder output
and LLM input dimension. Multiple adapter types supported.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


class LyzConvAdapter(nn.Module):
    """
    LyzConv1dSubsampling Adapter

    CNN-based adapter with 2-layer structure:
    - Layer 1: Conv1d + BatchNorm + ReLU (expands to 2*encoder_dim)
    - Layer 2: Conv1d + BatchNorm + ReLU (expands to 4*encoder_dim, stride=2)
    - Projection: Linear (4*encoder_dim -> llm_dim)
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        kernel_size: int = 5,
        activation_func: str = 'relu',
        norm: str = 'batch',
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        if encoder_dim * 4 < llm_dim:
            self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d1 = nn.Conv1d(encoder_dim, 2 * encoder_dim, kernel_size, 1, 0)
            self.bn1 = nn.BatchNorm1d(2 * encoder_dim, eps=1e-3, momentum=0.99)
            self.relu1 = nn.ReLU()

            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(2 * encoder_dim, 4 * encoder_dim, kernel_size, 2, 0)
            self.bn2 = nn.BatchNorm1d(4 * encoder_dim, eps=1e-3, momentum=0.99)
            self.relu2 = nn.ReLU()

            self.project = nn.Linear(4 * encoder_dim, llm_dim)
            self.cnn_num = 2
        else:
            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(encoder_dim, 2 * encoder_dim, kernel_size, 2, 0)
            if norm == 'batch':
                self.bn2 = nn.BatchNorm1d(2 * encoder_dim, eps=1e-3, momentum=0.99)
            elif norm == 'layer':
                self.bn2 = nn.LayerNorm(2 * encoder_dim, eps=1e-3)

            if activation_func == 'gelu':
                self.relu2 = nn.GELU()
            else:
                self.relu2 = nn.ReLU()

            self.project = nn.Linear(2 * encoder_dim, llm_dim)
            self.cnn_num = 1

    def forward(self, x: torch.Tensor, mask_pad: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.transpose(1, 2)

        if mask_pad is not None and mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        if self.cnn_num == 2:
            x = self.left_padding1(x)
            x = self.conv1d1(x)
            x = self.bn1(x)
            x = self.relu1(x)

        x = self.left_padding2(x)
        x = self.conv1d2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.bn2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.relu2(x)

        x = x.transpose(1, 2)
        x = self.project(x)

        if mask_pad is not None:
            mask_pad = mask_pad[:, :, 0::2]

        return x, mask_pad


class GxlConvAdapter(nn.Module):
    """
    GxlConv1dSubsampling Adapter (4x downsampling)

    Alternative CNN-based adapter with 3-layer structure.
    Requires additional linear projection after convolution.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        downsample_rate: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        if downsample_rate == 4:
            self.conv = nn.Sequential(
                nn.ConstantPad1d((2, 0), 0.0),
                nn.Conv1d(encoder_dim, encoder_dim, 3, 1),
                nn.GELU(),
                nn.ConstantPad1d((2, 0), 0.0),
                nn.Conv1d(encoder_dim, encoder_dim, 3, 2),
                nn.GELU(),
                nn.ConstantPad1d((2, 0), 0.0),
                nn.Conv1d(encoder_dim, encoder_dim, 3, 2),
                nn.GELU(),
            )
        elif downsample_rate == 2:
            self.conv = nn.Sequential(
                nn.Conv1d(encoder_dim, encoder_dim, 3, 1),
                nn.GELU(),
                nn.Conv1d(encoder_dim, encoder_dim, 3, 2),
                nn.GELU(),
            )
        elif downsample_rate == 8:
            self.conv = nn.Sequential(
                nn.Conv1d(encoder_dim, encoder_dim, 3, 1),
                nn.GELU(),
                nn.Conv1d(encoder_dim, encoder_dim, 3, 2),
                nn.GELU(),
                nn.Conv1d(encoder_dim, encoder_dim, 3, 2),
                nn.GELU(),
                nn.Conv1d(encoder_dim, encoder_dim, 3, 2),
                nn.GELU(),
            )
        else:
            self.conv = nn.Identity()

        self.proj = nn.Linear(encoder_dim, llm_dim)

    def forward(self, x: torch.Tensor, mask_pad: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x, mask_pad


class LinearAdapter(nn.Module):
    """
    Simple Linear Adapter

    Basic linear projection from encoder_dim to llm_dim.
    No temporal downsampling.
    """

    def __init__(self, encoder_dim: int, llm_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, llm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_pad: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.dropout(x)
        return self.proj(x), mask_pad
