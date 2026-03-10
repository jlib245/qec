# qec_sim/data/preprocessors.py
import torch
from typing import Dict, Any, List
from qec_sim.core.interfaces import BasePreprocessor
from qec_sim.data.registry import register_preprocessor


@register_preprocessor("spatial_grid")
class SpatialGridPreprocessor(BasePreprocessor):
    def __init__(self, detector_coords: dict, num_detectors: int, use_erasures: bool = True, **kwargs):
        self.num_detectors = num_detectors
        self.use_erasures = use_erasures

        self._required_keys = ["syndromes"]
        if self.use_erasures:
            self._required_keys.append("erasures")

        all_x = [c[0] for c in detector_coords.values()]
        all_y = [c[1] for c in detector_coords.values()]
        all_t = [c[2] for c in detector_coords.values()] if len(list(detector_coords.values())[0]) > 2 else [0]

        min_x, min_y = min(all_x), min(all_y)
        self.grid_w = int((max(all_x) - min_x) // 2.0) + 1
        self.grid_h = int((max(all_y) - min_y) // 2.0) + 1

        unique_t = sorted(list(set(all_t)))
        self.input_depth = len(unique_t)
        self.out_channels = self.input_depth * (2 if self.use_erasures else 1)

        det_indices, c_indices, h_indices, w_indices = [], [], [], []
        t_map = {t: i for i, t in enumerate(unique_t)}
        for det_idx in range(num_detectors):
            if det_idx in detector_coords:
                coords = detector_coords[det_idx]
                x, y = coords[0], coords[1]
                t = coords[2] if len(coords) > 2 else 0
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
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_syn = batch_data["syndromes"]
        batch_size = batch_syn.size(0)
        device = batch_syn.device

        if self.det_idx.device != device:
            self.det_idx = self.det_idx.to(device)
            self.c_idx = self.c_idx.to(device)
            self.h_idx = self.h_idx.to(device)
            self.w_idx = self.w_idx.to(device)

        grid = torch.full((batch_size, self.out_channels, self.grid_h, self.grid_w), -0.5, device=device)
        grid[:, self.c_idx, self.h_idx, self.w_idx] = batch_syn[:, self.det_idx]

        if self.use_erasures and "erasures" in batch_data:
            grid[:, self.c_idx + self.input_depth, self.h_idx, self.w_idx] = batch_data["erasures"][:, self.det_idx]

        return grid


@register_preprocessor("flat")
class FlatPreprocessor(BasePreprocessor):
    """MLP 계열 모델용 flat 전처리기. syndromes과 erasures를 이어붙여 1D 텐서로 반환합니다."""

    def __init__(self, num_detectors: int, use_erasures: bool = True, **kwargs):
        self.num_detectors = num_detectors
        self.use_erasures = use_erasures
        self._required_keys = ["syndromes"]
        if use_erasures:
            self._required_keys.append("erasures")

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        multiplier = 2 if self.use_erasures else 1
        return {"input_dim": self.num_detectors * multiplier}

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = [batch_data["syndromes"].float()]
        if self.use_erasures and "erasures" in batch_data:
            tensors.append(batch_data["erasures"].float())
        return torch.cat(tensors, dim=-1)
