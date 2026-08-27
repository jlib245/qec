"""Edge-scatter CNN family.

Architecture pattern (shared across variants):
  1. Per-edge MLP: edge feature (3-dim) → embedding (c_edge-dim)
  2. Spatial scatter: edge_emb → endpoint detectors → (T, H, W) grid
  3. Lightweight 3D CNN backbone (variant-specific blocks)
  4. GAP + head → logical bit

Variants:
  - edge_scatter_mbconv_se   (F3): EfficientNet-style MBConv + SE
  - edge_scatter_dw_cbam     (F1): Depthwise separable + CBAM3D
  - edge_scatter_inv_cbam    (F2): MobileNetV2 inverted residual + CBAM3D
  - edge_scatter_plain_cbam  (A):  Plain conv + CBAM3D

All variants share the same input layout from CachedEdgeScatterCNNPreprocessor:
  x (B, n_undir_e * 3) — flat edge features [w_e_norm, 1[E_0], 1[E_1]].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model


# ─────────────────────────────────────────────────────────────────────────
# Reusable attention blocks
# ─────────────────────────────────────────────────────────────────────────

class _ChannelAttention3D(nn.Module):
    """CBAM channel attention (3D version)."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg = F.adaptive_avg_pool3d(x, 1).view(b, c)
        mx = F.adaptive_max_pool3d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn.view(b, c, 1, 1, 1)


class _SpatialAttention3D(nn.Module):
    """CBAM spatial attention (3D version)."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class _CBAM3D(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16,
                 spatial_kernel_size: int = 3):
        super().__init__()
        self.channel = _ChannelAttention3D(in_channels, reduction_ratio)
        self.spatial = _SpatialAttention3D(spatial_kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


class _SE3D(nn.Module):
    """Squeeze-Excitation (channel attention only, lighter than CBAM)."""

    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        reduced = max(1, int(in_channels * se_ratio))
        self.fc1 = nn.Conv3d(in_channels, reduced, 1)
        self.fc2 = nn.Conv3d(reduced, in_channels, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        se = F.adaptive_avg_pool3d(x, 1)
        se = self.act(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


# ─────────────────────────────────────────────────────────────────────────
# Backbone blocks (one per variant)
# ─────────────────────────────────────────────────────────────────────────

class _PlainConvCBAMBlock(nn.Module):
    """A: plain 3D conv + CBAM."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.bn = nn.BatchNorm3d(out_c)
        self.cbam = _CBAM3D(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn(self.conv(x)))
        h = self.cbam(h)
        return h


class _DWConvCBAMBlock(nn.Module):
    """F1: depthwise separable + CBAM."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = nn.Conv3d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm3d(in_c)
        self.pw = nn.Conv3d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.cbam = _CBAM3D(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.dw(x)))
        h = self.act(self.bn2(self.pw(h)))
        h = self.cbam(h)
        return h


class _InvertedResidualCBAMBlock(nn.Module):
    """F2: MobileNetV2 inverted residual + CBAM."""

    def __init__(self, in_c, out_c, expansion=4):
        super().__init__()
        mid_c = in_c * expansion
        self.expand = nn.Conv3d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_c)
        self.dw = nn.Conv3d(mid_c, mid_c, 3, padding=1, groups=mid_c, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_c)
        self.cbam = _CBAM3D(mid_c)
        self.project = nn.Conv3d(mid_c, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.use_skip = (in_c == out_c)

    def forward(self, x):
        h = self.act(self.bn1(self.expand(x)))
        h = self.act(self.bn2(self.dw(h)))
        h = self.cbam(h)
        h = self.bn3(self.project(h))
        return x + h if self.use_skip else h


class _MBConvSEBlock(nn.Module):
    """F3: EfficientNet-style Mobile Bottleneck Conv with SE."""

    def __init__(self, in_c, out_c, expansion=4, se_ratio=0.25):
        super().__init__()
        mid_c = in_c * expansion
        self.expand = nn.Conv3d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_c)
        self.dw = nn.Conv3d(mid_c, mid_c, 3, padding=1, groups=mid_c, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_c)
        self.se = _SE3D(mid_c, se_ratio)
        self.project = nn.Conv3d(mid_c, out_c, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.expand(x)))
        h = self.act(self.bn2(self.dw(h)))
        h = self.se(h)
        h = self.bn3(self.project(h))
        return h


# ─────────────────────────────────────────────────────────────────────────
# Base class: edge embedding + scatter to grid + backbone + head
# ─────────────────────────────────────────────────────────────────────────

class _EdgeScatterCNNBase(BaseQECModel):
    """Shared infrastructure for edge-scatter CNN family.

    Subclasses define `_make_backbone(in_c, channels)` returning an
    nn.Sequential that maps (B, in_c, T, H, W) → (B, out_c, T, H, W),
    plus pooled (B, last_channels).
    """

    def __init__(
        self,
        num_observables: int,
        T: int, H: int, W: int,
        n_det: int,
        n_undir_e: int,
        d_edge: int = 3,
        n_scalars: int = 0,
        edge_endpoint_a_np=None,
        edge_endpoint_b_np=None,
        det_flat_idx_np=None,
        c_edge: int = 16,
        channels=(24, 40, 80),
        head_hidden: int = 64,
        dropout: float = 0.2,
        max_batch: int = 1024,
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        assert edge_endpoint_a_np is not None
        assert edge_endpoint_b_np is not None
        assert det_flat_idx_np is not None

        self.T, self.H, self.W = T, H, W
        self.n_det = n_det
        self.n_undir_e = n_undir_e
        self.d_edge = d_edge
        self.c_edge = c_edge
        self.max_batch = max_batch

        # Pre-compute valid (non-virtual) endpoint masks and indices once,
        # so the forward path avoids dynamic boolean masking / index ops.
        import numpy as _np
        a_np = _np.asarray(edge_endpoint_a_np, dtype=_np.int64)
        b_np = _np.asarray(edge_endpoint_b_np, dtype=_np.int64)
        valid_a_mask = a_np < n_det
        valid_b_mask = b_np < n_det
        # Indices INTO edge dim (length = n_valid_a / n_valid_b)
        valid_a_edge_idx = _np.nonzero(valid_a_mask)[0]
        valid_b_edge_idx = _np.nonzero(valid_b_mask)[0]
        a_valid = a_np[valid_a_mask]   # detector idx for each valid_a edge
        b_valid = b_np[valid_b_mask]

        self.register_buffer(
            "_valid_a_edge_idx",
            torch.from_numpy(valid_a_edge_idx).long(),
            persistent=False,
        )
        self.register_buffer(
            "_valid_b_edge_idx",
            torch.from_numpy(valid_b_edge_idx).long(),
            persistent=False,
        )
        # Pre-expanded scatter index targets — (max_batch, n_valid_*, c_edge)
        # so the forward doesn't allocate new index tensors each call.
        self.register_buffer(
            "_idx_a_full",
            torch.from_numpy(a_valid).long()
                .view(1, -1, 1).expand(max_batch, -1, c_edge).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_idx_b_full",
            torch.from_numpy(b_valid).long()
                .view(1, -1, 1).expand(max_batch, -1, c_edge).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_det_flat_idx",
            torch.from_numpy(_np.asarray(det_flat_idx_np, dtype=_np.int64)).long(),
            persistent=False,
        )
        # Pre-expanded target_idx for the (det → grid) scatter.
        det_flat = torch.from_numpy(_np.asarray(det_flat_idx_np, dtype=_np.int64)).long()
        self.register_buffer(
            "_target_idx_full",
            det_flat.view(1, 1, -1).expand(max_batch, c_edge, -1).contiguous(),
            persistent=False,
        )

        # Static work buffers (pre-allocated once at init).
        # These are persistent=False so they don't end up in state_dict
        # (best_model.pth from the old model loads cleanly with strict=False).
        self.register_buffer(
            "_voxel_buf",
            torch.zeros(max_batch, n_det, c_edge),
            persistent=False,
        )
        self.register_buffer(
            "_grid_buf",
            torch.zeros(max_batch, c_edge, T * H * W),
            persistent=False,
        )

        # Per-edge MLP: raw edge feature (d_edge) → embedding (c_edge)
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_edge, c_edge),
            nn.SiLU(inplace=True),
            nn.Linear(c_edge, c_edge),
        )

        # Backbone (variant-specific, built by subclass)
        self.backbone = self._make_backbone(c_edge, channels)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_observables),
        )

    # ─── subclass hook ───
    def _make_backbone(self, in_c, channels):
        raise NotImplementedError

    # ─── shared forward ───
    # In *eval* (inference) mode, uses pre-allocated static buffers so the
    # forward path is CUDA-graph capturable (no new allocations).
    # In *train* mode, autograd cannot in-place-modify leaf buffers reused
    # across iterations (raises "backward through the graph a second time")
    # → falls back to fresh allocations per forward (same behavior as the
    # original implementation; no perf loss since training is anyway dominated
    # by backward).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        edge_feat = x.view(B, self.n_undir_e, self.d_edge)          # (B, n_edge, d_edge)

        # Per-edge embedding
        edge_emb = self.edge_mlp(edge_feat)                         # (B, n_edge, c_edge)

        c = self.c_edge
        n_det = self.n_det

        # Pre-computed valid-edge index gather (avoids dynamic boolean masking)
        idx_a_full = self._idx_a_full
        idx_b_full = self._idx_b_full
        target_idx_full = self._target_idx_full

        if self.training:
            # ─── training path: fresh allocations (autograd-safe) ─────────
            voxel = torch.zeros(B, n_det, c, device=x.device, dtype=edge_emb.dtype)
            idx_a = idx_a_full[:B]
            idx_b = idx_b_full[:B]
            edge_emb_a = edge_emb.index_select(1, self._valid_a_edge_idx)
            edge_emb_b = edge_emb.index_select(1, self._valid_b_edge_idx)
            voxel.scatter_add_(1, idx_a, edge_emb_a)
            voxel.scatter_add_(1, idx_b, edge_emb_b)

            grid_flat = torch.zeros(B, c, self.T * self.H * self.W,
                                    device=x.device, dtype=voxel.dtype)
            target_idx = target_idx_full[:B]
            grid_flat.scatter_(2, target_idx, voxel.transpose(1, 2))
            grid = grid_flat.view(B, c, self.T, self.H, self.W)
        else:
            # ─── eval / inference path: static buffers (CUDA-graph friendly)
            if B > self.max_batch:
                raise RuntimeError(
                    f"batch={B} > max_batch={self.max_batch}; "
                    f"rebuild model with larger max_batch."
                )
            voxel = self._voxel_buf[:B]
            voxel.zero_()
            idx_a = idx_a_full[:B]
            idx_b = idx_b_full[:B]
            edge_emb_a = edge_emb.index_select(1, self._valid_a_edge_idx)
            edge_emb_b = edge_emb.index_select(1, self._valid_b_edge_idx)
            voxel.scatter_add_(1, idx_a, edge_emb_a)
            voxel.scatter_add_(1, idx_b, edge_emb_b)

            grid_flat = self._grid_buf[:B]
            grid_flat.zero_()
            target_idx = target_idx_full[:B]
            grid_flat.scatter_(2, target_idx, voxel.transpose(1, 2))
            grid = grid_flat.view(B, c, self.T, self.H, self.W)

        # Backbone + head
        feat = self.backbone(grid)
        return self.head(feat)


# ─────────────────────────────────────────────────────────────────────────
# Variant models
# ─────────────────────────────────────────────────────────────────────────

@register_model("edge_scatter_mbconv_se")
class EdgeScatterMBConvSE(_EdgeScatterCNNBase):
    """F3: EfficientNet-style MBConv + SE backbone."""
    REQUIRED_PREPROCESSOR = "cached_edge_scatter_cnn"

    def _make_backbone(self, in_c, channels):
        layers = []
        prev = in_c
        for ch in channels:
            layers.append(_MBConvSEBlock(prev, ch, expansion=4, se_ratio=0.25))
            prev = ch
        return nn.Sequential(*layers)


@register_model("edge_scatter_dw_cbam")
class EdgeScatterDWCBAM(_EdgeScatterCNNBase):
    """F1: depthwise separable + CBAM3D backbone."""
    REQUIRED_PREPROCESSOR = "cached_edge_scatter_cnn"

    def _make_backbone(self, in_c, channels):
        layers = []
        prev = in_c
        for ch in channels:
            layers.append(_DWConvCBAMBlock(prev, ch))
            prev = ch
        return nn.Sequential(*layers)


@register_model("edge_scatter_inv_cbam")
class EdgeScatterInvCBAM(_EdgeScatterCNNBase):
    """F2: MobileNetV2 inverted residual + CBAM3D backbone."""
    REQUIRED_PREPROCESSOR = "cached_edge_scatter_cnn"

    def _make_backbone(self, in_c, channels):
        layers = []
        prev = in_c
        for ch in channels:
            layers.append(_InvertedResidualCBAMBlock(prev, ch, expansion=4))
            prev = ch
        return nn.Sequential(*layers)


@register_model("edge_scatter_plain_cbam")
class EdgeScatterPlainCBAM(_EdgeScatterCNNBase):
    """A: plain 3D conv + CBAM3D backbone."""
    REQUIRED_PREPROCESSOR = "cached_edge_scatter_cnn"

    def _make_backbone(self, in_c, channels):
        layers = []
        prev = in_c
        for ch in channels:
            layers.append(_PlainConvCBAMBlock(prev, ch))
            prev = ch
        return nn.Sequential(*layers)
