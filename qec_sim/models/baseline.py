# qec_sim/models/baseline.py
import torch.nn as nn
from qec_sim.models.registry import register_model  # 정확한 파일명(.registry) 명시!
from qec_sim.core.interfaces import BaseQECModel

@register_model("erasure_mlp")
class ErasureAwareMLP(nn.Module):
    def __init__(self, input_dim: int, num_observables: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_observables)
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1) 
        return self.network(x_flat)