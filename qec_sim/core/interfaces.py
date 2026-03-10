import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from torch.utils.data import DataLoader

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


class BaseDataStrategy(ABC):
    """[Level 3] 데이터 공급기"""
    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        pass