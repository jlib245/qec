import torch
from typing import Dict, Any, List
from qec_sim.core.interfaces import BasePreprocessor
from qec_sim.data.registry import register_preprocessor

@register_preprocessor("spatial_grid")
class SpatialGridPreprocessor(BasePreprocessor):
    def __init__(self, detector_coords: dict, num_detectors: int, use_erasures: bool = True):
        self.num_detectors = num_detectors
        self.use_erasures = use_erasures
        
        # 1. 상위 계층(데이터셋)에 요구할 데이터 키 명시
        self._required_keys = ["syndromes"]
        if self.use_erasures:
            self._required_keys.append("erasures")

        # 2. Grid 차원 계산
        all_x = [c[0] for c in detector_coords.values()]
        all_y = [c[1] for c in detector_coords.values()]
        all_t = [c[2] for c in detector_coords.values()] if len(list(detector_coords.values())[0]) > 2 else [0]
        
        min_x, min_y = min(all_x), min(all_y)
        self.grid_w = int((max(all_x) - min_x) // 2.0) + 1
        self.grid_h = int((max(all_y) - min_y) // 2.0) + 1
        
        unique_t = sorted(list(set(all_t)))
        self.input_depth = len(unique_t)
        self.out_channels = self.input_depth * (2 if self.use_erasures else 1)
        
        # 3. GPU 인덱싱용 텐서 캐싱
        det_indices, c_indices, h_indices, w_indices = [], [], [], []
        t_map = {t: i for i, t in enumerate(unique_t)}
        for det_idx in range(num_detectors):
            if det_idx in detector_coords:
                x, y, t = detector_coords[det_idx][0], detector_coords[det_idx][1], (detector_coords[det_idx][2] if len(detector_coords[det_idx]) > 2 else 0)
                det_indices.append(det_idx)
                c_indices.append(t_map[t])
                h_indices.append(int((y - min_y) // 2.0))
                w_indices.append(int((x - min_x) // 2.0))
                
        self.det_idx = torch.tensor(det_indices, dtype=torch.long)
        self.c_idx = torch.tensor(c_indices, dtype=torch.long)
        self.h_idx = torch.tensor(h_indices, dtype=torch.long)
        self.w_idx = torch.tensor(w_indices, dtype=torch.long)

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"in_channels": self.out_channels, "grid_h": self.grid_h, "grid_w": self.grid_w}

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        # CNN 전처리는 가벼우므로 CPU 통과
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_syn = batch_data["syndromes"]
        batch_size = batch_syn.size(0)
        device = batch_syn.device
        
        if self.det_idx.device != device:
            self.det_idx, self.c_idx, self.h_idx, self.w_idx = (
                self.det_idx.to(device), self.c_idx.to(device), 
                self.h_idx.to(device), self.w_idx.to(device)
            )

        grid = torch.full((batch_size, self.out_channels, self.grid_h, self.grid_w), -0.5, device=device)
        grid[:, self.c_idx, self.h_idx, self.w_idx] = batch_syn[:, self.det_idx]
        
        if self.use_erasures and "erasures" in batch_data:
            grid[:, self.c_idx + self.input_depth, self.h_idx, self.w_idx] = batch_data["erasures"][:, self.det_idx]
            
        return grid