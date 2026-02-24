import torch
import torch.nn as nn

class ErasureAwareMLP(nn.Module):
    def __init__(self, num_detectors: int, num_observables: int, hidden_dim: int = 512):
        super().__init__()
        
        # 입력 데이터 형태: (Batch, 2, num_detectors)
        # 2채널(신드롬, 누설)을 평탄화(Flatten)하여 입력으로 받으므로 입력 크기는 2 * num_detectors가 됩니다.
        input_dim = 2 * num_detectors
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            # 최종 출력: 논리적 큐비트(Observable)가 플립되었는지 여부
            nn.Linear(hidden_dim // 2, num_observables)
        )

    def forward(self, x):
        # x.shape: (Batch, Channels=2, Detectors)
        # 평탄화: (Batch, 2 * Detectors)
        x_flat = x.view(x.size(0), -1) 
        
        # 모델 통과 (Logits 반환)
        logits = self.network(x_flat)
        return logits