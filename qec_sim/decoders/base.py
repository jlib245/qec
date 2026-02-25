# qec_sim/decoders/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseDecoder(ABC):
    """
    모든 QEC 디코더가 상속받아야 하는 추상 기본 클래스입니다.
    """
    @abstractmethod
    def decode_batch(self, syndromes: np.ndarray, erasures: np.ndarray = None) -> np.ndarray:
        """
        주어진 신드롬과 누설(erasure) 맵을 기반으로 논리적 에러를 예측합니다.
        
        Args:
            syndromes (np.ndarray): (Batch, Num_Detectors) 형태의 에러 신드롬 배열
            erasures (np.ndarray, optional): (Batch, Num_Detectors) 형태의 누설 맵. Defaults to None.
            
        Returns:
            np.ndarray: (Batch, Num_Observables) 형태의 논리적 에러 예측값
        """
        pass