import torch
import torch.nn as nn
from qec_sim.core.interfaces import BaseQECModel
from qec_sim.models.registry import register_model

@register_model("jung_cnn_low")
class JungCNN_Low(BaseQECModel):
    REQUIRED_PREPROCESSOR = "spatial_grid" 
    
    def __init__(self, in_channels: int, grid_h: int, grid_w: int, num_observables: int, code_distance: int, **kwargs):
        super().__init__(num_observables)
        
        n_filters = 8 if code_distance <= 3 else (32 if code_distance == 5 else 64)
            
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(n_filters, n_filters, kernel_size=2, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )
        
        # Conv2d(kernel_size=2, padding=1)은 출력 크기를 H+1로 바꾸므로
        # 더미 텐서로 실제 flatten 크기를 동적으로 계산합니다.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_h, grid_w)
            flatten_dim = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, num_observables)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)