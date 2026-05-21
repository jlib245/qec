"""GapRoutedCNN — 3D CNN that consumes (syndrome + MWPM edge-diff + per-detector
edge weight masks) as 4 spatial channels, plus scalar features (gap, w_0, w_1,
weight_sum, n_matched_norm, mwpm_pred) in the dense head.

Designed as the "strong stage" of a confidence router: at inference time the
caller selects between MWPM-only (gap ≥ τ) and this CNN (gap < τ). At training
time we still train on the full distribution and let the model learn to imitate
MWPM where gap is large.

Input tensor layout (produced by GapRoutedCNNPreprocessor):
  x[:, :grid_dim]                  flat 4-channel grid  (4 * T * H * W)
  x[:, grid_dim:grid_dim+6]        scalar features
"""
import torch
import torch.nn as nn

from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model


@register_model("gap_routed_cnn")
class GapRoutedCNN(BaseQECModel):
    REQUIRED_PREPROCESSOR = "gap_routed_cnn"

    def __init__(
        self,
        num_observables: int,
        T: int, H: int, W: int,
        n_grid_channels: int = 4,
        n_scalars: int = 6,
        conv_channels=(32, 64, 128),
        hidden: int = 128,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(num_observables=num_observables)
        self.T, self.H, self.W = T, H, W
        self.n_grid_channels = n_grid_channels
        self.n_scalars = n_scalars
        self.grid_dim = n_grid_channels * T * H * W

        conv_layers = []
        in_c = n_grid_channels
        for out_c in conv_channels:
            conv_layers += [
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        conv_layers += [nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()]
        self.conv = nn.Sequential(*conv_layers)

        feat_dim = conv_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(feat_dim + n_scalars, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_observables),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid_flat = x[:, : self.grid_dim]
        scalars = x[:, self.grid_dim : self.grid_dim + self.n_scalars]
        grid = grid_flat.view(-1, self.n_grid_channels, self.T, self.H, self.W)
        feat = self.conv(grid)
        return self.head(torch.cat([feat, scalars], dim=1))


@register_model("gap_routed_cnn_filtered")
class GapRoutedCNNFiltered(GapRoutedCNN):
    """Same as GapRoutedCNN but trained on gap-filtered data via
    GapRoutedCNNFilteredPreprocessor (gap < threshold sub-distribution)."""
    REQUIRED_PREPROCESSOR = "gap_routed_cnn_filtered"


@register_model("syndrome_3d_cnn")
class Syndrome3DCNN(GapRoutedCNN):
    """Syndrome-only 3D CNN — same architecture as GapRoutedCNN but consumes
    only the syndrome volume (no MWPM byproducts). Pairs with the
    `syndrome_3d` preprocessor (1 grid channel, 0 scalars)."""
    REQUIRED_PREPROCESSOR = "syndrome_3d"

    def __init__(self, num_observables: int, T: int, H: int, W: int,
                 n_grid_channels: int = 1, n_scalars: int = 0,
                 **kwargs):
        super().__init__(num_observables=num_observables,
                         T=T, H=H, W=W,
                         n_grid_channels=n_grid_channels,
                         n_scalars=n_scalars,
                         **kwargs)


@register_model("syndrome_3d_cnn_filtered")
class Syndrome3DCNNFiltered(Syndrome3DCNN):
    """Syndrome-only 3D CNN with gap-filtered fine-tuning preprocessor."""
    REQUIRED_PREPROCESSOR = "syndrome_3d_filtered"
