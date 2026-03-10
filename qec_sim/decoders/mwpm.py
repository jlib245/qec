# qec_sim/decoders/mwpm.py
import pymatching
import stim
import numpy as np

from .base import BaseDecoder
from .registry import register_decoder


@register_decoder("mwpm")
class ErasureMWPM(BaseDecoder):
    def __init__(self, error_model: stim.DetectorErrorModel, **kwargs):
        self.error_model = error_model
        self.base_matching = pymatching.Matching.from_detector_error_model(error_model)

    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None) -> np.ndarray:
        # erasure 없는 경우 → pymatching 배치 디코딩 (빠름)
        if erasures is None or not np.any(erasures):
            return self.base_matching.decode_batch(syndromes)

        has_erasure = np.any(erasures, axis=1)  # (n_shots,) bool

        # 전체 결과 배열 초기화 (shape 추론: 한 샘플 디코딩으로 확인)
        sample_pred = self.base_matching.decode(syndromes[0])
        predictions = np.zeros((len(syndromes), len(sample_pred)), dtype=bool)

        # erasure 없는 샷: 배치 디코딩
        no_era_idx = np.where(~has_erasure)[0]
        if len(no_era_idx) > 0:
            predictions[no_era_idx] = self.base_matching.decode_batch(syndromes[no_era_idx])

        # erasure 있는 샷: 개별 처리 (가중치 수정 필요)
        for i in np.where(has_erasure)[0]:
            temp_matching = pymatching.Matching.from_detector_error_model(self.error_model)
            for d in np.where(erasures[i])[0]:
                temp_matching.add_boundary_edge(
                    int(d),
                    fault_ids=set(),
                    weight=0.0,
                    error_probability=1.0,
                    merge_strategy="replace",
                )
            predictions[i] = temp_matching.decode(syndromes[i])

        return predictions
