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
        
        flatten_dim = n_filters * grid_h * grid_w
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, num_observables)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)