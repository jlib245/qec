import pymatching
import stim
import numpy as np

class ErasureMWPM:
    def __init__(self, error_model: stim.DetectorErrorModel):
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
                
                # [핵심] 수정한 그래프로 디코딩을 수행하여 prediction에 저장합니다.
                prediction = temp_matching.decode(syndrome)
            else:
                # 누설 정보가 없거나 누설이 발생하지 않은 경우 기본 디코딩
                prediction = self.base_matching.decode(syndrome)
                
            predictions.append(prediction)

        return np.array(predictions)