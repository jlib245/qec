# qec_sim/decoders/mwpm.py
import os
import pymatching
import stim
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .base import BaseDecoder
from .registry import register_decoder


@register_decoder("mwpm")
class ErasureMWPM(BaseDecoder):
    def __init__(self, error_model: stim.DetectorErrorModel, circuit: stim.Circuit = None, **kwargs):
        super().__init__(circuit=circuit)
        self.error_model = error_model
        self.base_matching = pymatching.Matching.from_detector_error_model(error_model)

    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None) -> np.ndarray:
        # erasure 없는 경우 → pymatching 배치 디코딩 (빠름)
        if erasures is None or not np.any(erasures):
            return self.base_matching.decode_batch(syndromes)

        has_erasure = np.any(erasures, axis=1)  # (n_shots,) bool

        # 전체 결과 배열 초기화
        sample_pred = self.base_matching.decode(syndromes[0])
        predictions = np.zeros((len(syndromes), len(sample_pred)), dtype=bool)

        # erasure 없는 샷: 배치 디코딩
        no_era_idx = np.where(~has_erasure)[0]
        if len(no_era_idx) > 0:
            predictions[no_era_idx] = self.base_matching.decode_batch(syndromes[no_era_idx])

        # erasure 있는 샷: ThreadPoolExecutor로 병렬 처리
        # (pymatching은 C++ 라이브러리로 GIL을 해제하므로 스레드 병렬화 효과 있음)
        era_indices = np.where(has_erasure)[0]
        if len(era_indices) == 0:
            return predictions

        error_model = self.error_model

        def _decode_one(i):
            tmp = pymatching.Matching.from_detector_error_model(error_model)
            for d in np.where(erasures[i])[0]:
                tmp.add_boundary_edge(
                    int(d), fault_ids=set(), weight=0.0,
                    error_probability=1.0, merge_strategy="replace",
                )
            return tmp.decode(syndromes[i])

        n_workers = min(os.cpu_count() or 4, len(era_indices))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for i, pred in zip(era_indices, ex.map(_decode_one, era_indices)):
                predictions[i] = pred

        return predictions

    def decode_single_with_correction(self, syndrome: np.ndarray) -> dict:
        logical_error = self.base_matching.decode(syndrome)
        edges = self.base_matching.decode_to_edges_array(syndrome)
        corrected_fault_ids = list({int(fid) for edge in edges for fid in edge if fid != -1})
        return {
            "logical_error": logical_error.tolist(),
            "corrected_fault_ids": corrected_fault_ids,
        }
