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
        # 기본 매칭 그래프 생성
        self.base_matching = pymatching.Matching.from_detector_error_model(error_model)

    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None) -> np.ndarray:
        """
        배치 단위로 신드롬을 디코딩합니다. erasures 맵이 주어지면 가중치를 수정합니다.
        """
        num_shots = syndromes.shape[0]
        predictions = []

        for i in range(num_shots):
            syndrome = syndromes[i]
            
            # [주의] if와 else 구조가 하나로 연결되어야 모든 경우에 prediction이 할당됩니다!
            if erasures is not None and np.any(erasures[i]):
                temp_matching = pymatching.Matching.from_detector_error_model(self.error_model)
                erased_detectors = np.where(erasures[i])[0]
                
                for d in erased_detectors:
                    temp_matching.add_boundary_edge(
                        int(d), 
                        fault_ids=set(), 
                        weight=0.0, 
                        error_probability=1.0, 
                        merge_strategy="replace"
                    )
                
                # 누설이 있을 때의 예측값 할당
                prediction = temp_matching.decode(syndrome)
            else:
                # 누설 정보가 없거나, 해당 샷에서 누설이 없을 때의 예측값 할당
                prediction = self.base_matching.decode(syndrome)
                
            # 위 if-else를 무조건 거치므로 prediction은 항상 존재합니다.
            predictions.append(prediction)

        return np.array(predictions)