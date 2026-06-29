"""CBAMDecoder — ResNet + CBAM 2D CNN for circuit-level decoding.

Ported from a standalone Jung-style decoder. Pairs with the existing
`spatial_grid` preprocessor: input is (B, n_rounds, H, W) with detector-less
cells filled with -0.5 (incoherent value).

Use with `coset_mode: true` in YAML so labels reach the trainer as (B,) long
indices in {0, ..., 2^k - 1} — matching CrossEntropy. For the standard
1-observable surface code that yields a 2-class head.

Aux head: when `aux_head: true`, training-time forward returns a (main, aux)
tuple; combine with the `cbam_aux_ce` criterion to add an aux CE term against
a fixed-zero target (regularization from the original training script).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model


class _ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attention = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attention.view(b, c, 1, 1)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attention


class _CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16,
                 spatial_kernel_size: int = 3):
        super().__init__()
        self.channel = _ChannelAttention(in_channels, reduction_ratio)
        self.spatial = _SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class _ResBlockCBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16,
                 spatial_kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = _CBAM(channels, reduction_ratio, spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return F.relu(out + residual)


@register_model("cbam")
class CBAMDecoder(BaseQECModel):
    """ResNet + CBAM circuit-level decoder (Jung-style).

    Requires `coset_mode: true` in YAML so num_observables is broadcast to
    2^k classes by the factory — the head emits CE logits.
    """
    REQUIRED_PREPROCESSOR = "spatial_grid"

    INCOHERENT_VALUE = -0.5

    def __init__(
        self,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        num_observables: int,
        code_distance: int,
        n_blocks: int = 4,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        input_dropout_apply: float = 0.0,
        aux_head: bool = False,
        nf_override: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(num_observables)

        if num_observables < 2:
            raise ValueError(
                "CBAMDecoder needs num_observables >= 2 (a multi-class head). "
                "Set `model.coset_mode: true` in YAML so the factory expands "
                "num_observables -> 2^k."
            )
        if min(int(grid_h), int(grid_w)) < 2:
            raise ValueError(f"Detector grid {grid_h}x{grid_w} too small.")

        self.n_rounds = int(in_channels)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.d = int(code_distance)
        self.input_dropout_p = float(input_dropout)
        self.input_dropout_apply_p = float(input_dropout_apply)
        self.aux_head_enabled = bool(aux_head)

        if nf_override is not None:
            self.nf = int(nf_override)
        else:
            target = self.d * self.d - 1
            u = math.ceil(math.log2(target)) if target > 1 else 1
            self.nf = 2 ** u

        self.stem = nn.Sequential(
            nn.Conv2d(self.n_rounds, self.nf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(inplace=True),
        )

        reduction = min(16, max(1, self.nf // 2))
        self.res_blocks = nn.Sequential(*[
            _ResBlockCBAM(self.nf, reduction_ratio=reduction, spatial_kernel_size=3)
            for _ in range(n_blocks)
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.nf, self.nf, kernel_size=2, padding=0, bias=False),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(inplace=True),
        )

        flatten_dim = self.nf * (self.grid_h - 1) * (self.grid_w - 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(50, self.num_observables),
        )

        if self.aux_head_enabled:
            self.aux_classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flatten_dim, 50),
                nn.ReLU(inplace=True),
                nn.Linear(50, self.num_observables),
            )
        else:
            self.aux_classifier = None

    def _apply_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training
                and self.input_dropout_p > 0.0
                and self.input_dropout_apply_p > 0.0):
            return x
        B = x.size(0)
        device = x.device
        sample_apply = (torch.rand(B, 1, 1, 1, device=device)
                        < self.input_dropout_apply_p)
        keep = (torch.rand_like(x) >= self.input_dropout_p)
        valid = (~sample_apply) | keep
        return torch.where(valid, x, torch.full_like(x, self.INCOHERENT_VALUE))

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_input_mask(x)
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.final_conv(x)

    def forward(self, x: torch.Tensor):
        feats = self._backbone(x)
        main = self.classifier(feats)
        if self.training and self.aux_classifier is not None:
            return main, self.aux_classifier(feats)
        return main
