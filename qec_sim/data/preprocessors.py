# qec_sim/data/preprocessors.py
import torch
from typing import Dict, Any, List
from qec_sim.core.interfaces import BasePreprocessor
from qec_sim.data.registry import register_preprocessor


@register_preprocessor("spatial_grid")
class SpatialGridPreprocessor(BasePreprocessor):
    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(
            detector_coords=circuit.get_detector_coordinates(),
            num_detectors=circuit.num_detectors,
        )

    def __init__(self, detector_coords: dict, num_detectors: int, **kwargs):
        self.num_detectors = num_detectors

        self._required_keys = ["syndromes"]

        all_x = [c[0] for c in detector_coords.values()]
        all_y = [c[1] for c in detector_coords.values()]
        all_t = [c[2] for c in detector_coords.values()] if len(list(detector_coords.values())[0]) > 2 else [0]

        min_x, min_y = min(all_x), min(all_y)
        self.grid_w = int((max(all_x) - min_x) // 2.0) + 1
        self.grid_h = int((max(all_y) - min_y) // 2.0) + 1

        unique_t = sorted(list(set(all_t)))
        self.input_depth = len(unique_t)
        self.out_channels = self.input_depth

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

        return grid


@register_preprocessor("soft_grid")
class SoftGridPreprocessor(BasePreprocessor):
    """
    IQ soft measurement 기반 spatial grid 전처리기.
    soft_measurements (shots, rounds, n_ancilla) → (shots, rounds, H, W) float tensor.

    각 채널은 한 라운드의 soft detection event:
      - 채널 0: soft_meas[0]  (첫 라운드 posterior, 완벽한 |0⟩ 참조 대비)
      - 채널 r>0: soft_xor(soft_meas[r], soft_meas[r-1])  (연속 라운드 XOR)
    """

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        if simulator_pool is None:
            raise ValueError("SoftGridPreprocessor requires simulator_pool (pauli_plus backend)")
        sim = simulator_pool.get_random_simulator()
        return cls(
            ancilla_coords=sim.get_ancilla_coordinates(),
            rounds=config.code.rounds,
        )

    def __init__(self, ancilla_coords: dict, rounds: int, **kwargs):
        self.rounds = rounds
        n_ancilla = len(ancilla_coords)

        all_x = [c[0] for c in ancilla_coords.values()]
        all_y = [c[1] for c in ancilla_coords.values()]
        min_x, min_y = min(all_x), min(all_y)
        self.grid_w = int((max(all_x) - min_x) / 2) + 1
        self.grid_h = int((max(all_y) - min_y) / 2) + 1

        h_indices = [int((ancilla_coords[i][1] - min_y) / 2) for i in range(n_ancilla)]
        w_indices = [int((ancilla_coords[i][0] - min_x) / 2) for i in range(n_ancilla)]

        self.h_idx = torch.tensor(h_indices, dtype=torch.long)
        self.w_idx = torch.tensor(w_indices, dtype=torch.long)

    @property
    def required_data_keys(self) -> List[str]:
        return ["soft_measurements"]

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"in_channels": self.rounds, "grid_h": self.grid_h, "grid_w": self.grid_w}

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # soft_meas: (batch, rounds, n_ancilla)
        soft_meas = batch_data["soft_measurements"]
        batch_size = soft_meas.size(0)
        device = soft_meas.device

        if self.h_idx.device != device:
            self.h_idx = self.h_idx.to(device)
            self.w_idx = self.w_idx.to(device)

        # soft detection events: round 0 그대로, round 1+ 는 soft XOR
        soft_r0 = soft_meas[:, 0:1, :]                                          # (B, 1, A)
        m_curr = soft_meas[:, 1:, :]
        m_prev = soft_meas[:, :-1, :]
        soft_rest = m_curr + m_prev - 2 * m_curr * m_prev                       # (B, R-1, A)
        soft_det = torch.cat([soft_r0, soft_rest], dim=1)                        # (B, R, A)

        grid = torch.zeros(batch_size, self.rounds, self.grid_h, self.grid_w, device=device)
        grid[:, :, self.h_idx, self.w_idx] = soft_det

        return grid


@register_preprocessor("flat")
class FlatPreprocessor(BasePreprocessor):
    """MLP 계열 모델용 flat 전처리기. syndromes을 1D 텐서로 반환합니다."""

    @classmethod
    def from_config(cls, config, circuit, simulator_pool=None):
        return cls(num_detectors=circuit.num_detectors)

    def __init__(self, num_detectors: int, **kwargs):
        self.num_detectors = num_detectors
        self._required_keys = ["syndromes"]

    @property
    def required_data_keys(self) -> List[str]:
        return self._required_keys

    def get_model_kwargs(self) -> Dict[str, Any]:
        return {"input_dim": self.num_detectors}

    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        return raw_sample

    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch_data["syndromes"].float()
