# qec_sim/models/baseline.py
import torch.nn as nn
from .registry import register_model

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