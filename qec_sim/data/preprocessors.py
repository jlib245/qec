# qec_sim/data/preprocessors.py
from abc import ABC, abstractmethod
import torch
from .registry import register_preprocessor

class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, syndromes: torch.Tensor, erasures: torch.Tensor = None) -> torch.Tensor:
        pass

@register_preprocessor("flat_preprocessor")
class FlatPreprocessor(BasePreprocessor):
    def __init__(self, num_detectors: int, use_erasures: bool):
        self.use_erasures = use_erasures
        self.output_dim = (num_detectors * 2) if use_erasures else num_detectors

    def process(self, syndromes, erasures=None) -> torch.Tensor:
        # numpy.ndarray로 들어올 경우 Tensor로 변환
        if not isinstance(syndromes, torch.Tensor):
            syndromes = torch.tensor(syndromes, dtype=torch.float32)
        if erasures is not None and not isinstance(erasures, torch.Tensor):
            erasures = torch.tensor(erasures, dtype=torch.float32)

        if self.use_erasures:
            if erasures is None:
                erasures = torch.zeros_like(syndromes)
            x = torch.stack([syndromes, erasures], dim=1)
        else:
            x = syndromes.unsqueeze(1) # (Batch, 1, Detectors)
            
        return x.float()

@register_preprocessor("grid_preprocessor")
class GridPreprocessor(BasePreprocessor):
    def __init__(self, detector_coords: dict, use_erasures: bool, x_step: float, y_step: float):
        self.use_erasures = use_erasures
        # (중략: 기존 H, W, 채널 수 계산 및 매핑 로직 유지)

    def process(self, syndromes, erasures=None) -> torch.Tensor:
        if not isinstance(syndromes, torch.Tensor):
            syndromes = torch.tensor(syndromes, dtype=torch.float32)
            
        batch_size = syndromes.shape[0]
        device = syndromes.device
        
        grid = torch.full((batch_size, self.out_channels, self.grid_h, self.grid_w), 
                          -0.5, dtype=torch.float32, device=device)
        
        if self.use_erasures and erasures is None:
            erasures = torch.zeros_like(syndromes)
        elif erasures is not None and not isinstance(erasures, torch.Tensor):
            erasures = torch.tensor(erasures, dtype=torch.float32, device=device)
            
        for (det_idx, t_idx, row, col) in self.mapping:
            grid[:, t_idx, row, col] = syndromes[:, det_idx]
            if self.use_erasures:
                grid[:, self.input_depth + t_idx, row, col] = erasures[:, det_idx]
                
        return grid