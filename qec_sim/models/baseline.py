# qec_sim/models/baseline.py
import torch
import torch.nn as nn

# 레지스트리에서 데코레이터 가져오기
from .registry import register_model

# "erasure_mlp"라는 이름표로 이 클래스를 시스템에 등록
@register_model("erasure_mlp")
class ErasureAwareMLP(nn.Module):
    def __init__(self, num_detectors: int, num_observables: int, hidden_dim: int = 512):
        super().__init__()
        
        # 입력 데이터 형태: (Batch, 2, num_detectors)
        input_dim = 2 * num_detectors
        
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
        logits = self.network(x_flat)
        return logits