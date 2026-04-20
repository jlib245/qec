import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from torch.utils.data import DataLoader


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class BaseQECModel(nn.Module, ABC):
    """[Level 1] 순수 코어 모델 (수학적 연산만 수행)"""
    def __init__(self, num_observables: int, **kwargs):
        super().__init__()
        self.num_observables = num_observables

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BasePreprocessor(ABC):
    """[Level 2] 전처리기 (데이터 공급과 가공 정책을 통제)"""

    @classmethod
    @abstractmethod
    def from_config(cls, config, circuit, simulator_pool=None) -> "BasePreprocessor":
        """팩토리가 전처리기를 생성할 때 호출. 각 서브클래스가 필요한 정보를 직접 추출."""
        pass

    @property
    @abstractmethod
    def required_data_keys(self) -> List[str]:
        """데이터셋이 하드디스크에서 읽어와야 할 필수 키 목록"""
        pass

    @abstractmethod
    def get_model_kwargs(self) -> Dict[str, Any]:
        """팩토리가 코어 모델을 생성할 때 주입할 파라미터 규격"""
        pass

    @abstractmethod
    def cpu_transform(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """[CPU] 데이터 로더 내부에서 수행될 무거운 연산"""
        pass

    @abstractmethod
    def gpu_transform(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[GPU] 래퍼 내부에서 수행될 초고속 텐서 변환. 반환값은 코어 모델의 입력이 됨."""
        pass


class BaseSimulator(ABC):
    """시뮬레이터 공통 인터페이스. CircuitNoiseSimulator와 PauliPlusSimulator가 구현."""

    @property
    @abstractmethod
    def num_detectors(self) -> int:
        pass

    @property
    @abstractmethod
    def num_observables(self) -> int:
        pass

    @abstractmethod
    def generate_data(self, shots: int) -> Dict[str, np.ndarray]:
        """{'syndromes', 'observables', 'erasures', ...} 형태의 dict 반환"""
        pass


class BaseDataStrategy(ABC):
    """[Level 3] 데이터 공급기"""
    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        pass