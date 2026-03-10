# qec_sim/models/baseline.py
import torch.nn as nn
from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model


@register_model("erasure_mlp")
class ErasureAwareMLP(BaseQECModel):
    REQUIRED_PREPROCESSOR = "flat"

    def __init__(self, input_dim: int, num_observables: int, hidden_dim: int = 256, **kwargs):
        super().__init__(num_observables)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_observables),
        )

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))
