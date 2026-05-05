# qec_sim/decoders/base.py
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

try:
    import stim
except ImportError:
    stim = None


class BaseDecoder(ABC):
    """
    모든 QEC 디코더가 상속받아야 하는 추상 기본 클래스입니다.
    """

    def __init__(self, circuit: "stim.Circuit" = None, **kwargs):
        # fault_id → flipped qubit 좌표 매핑 (시각화용, circuit 있을 때만 빌드)
        self._fault_id_to_qubits: dict = {}
        if circuit is not None and stim is not None:
            dem = circuit.detector_error_model(decompose_errors=True)
            explained = circuit.explain_detector_error_model_errors(
                dem_filter=dem,
                reduce_to_one_representative_error=True,
            )
            for fault_id, err in enumerate(explained):
                coords = set()
                for loc in err.circuit_error_locations:
                    for gt in loc.flipped_pauli_product:
                        c = gt.coords
                        if c:
                            coords.add((float(c[0]), float(c[1])))
                self._fault_id_to_qubits[fault_id] = list(coords)

    @abstractmethod
    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """
        주어진 신드롬을 기반으로 논리적 에러를 예측합니다.

        Args:
            syndromes (np.ndarray): (Batch, Num_Detectors) 형태의 에러 신드롬 배열

        Returns:
            np.ndarray: (Batch, Num_Observables) 형태의 논리적 에러 예측값
        """
        pass

    @abstractmethod
    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        """
        단일 샷 디코딩 + correction이 적용된 fault_ids 반환 (시각화용).

        Returns:
            {
                "logical_error": (n_observables,) bool list,
                "corrected_fault_ids": list[int],
            }
        """
        pass

    def get_corrected_qubits(self, fault_ids: list[int]) -> list[dict]:
        """fault_id 목록 → correction이 적용된 data qubit 좌표 목록 반환 (시각화용)."""
        coords = set()
        for fid in fault_ids:
            for coord in self._fault_id_to_qubits.get(fid, []):
                coords.add(coord)
        return [{"x": x, "y": y} for x, y in coords]