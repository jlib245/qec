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


class BasePreprocessor(nn.Module, ABC):
    """[Level 2] 전처리기 (데이터 공급과 가공 정책을 통제).

    nn.Module 상속 — 텐서 인덱스를 register_buffer(persistent=False)로
    등록하면 wrapper.to(device) 시 자동 이동, state_dict에는 미포함.
    """

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
        """{'syndromes', 'observables', ...} 형태의 dict 반환"""
        pass

    def reseed(self, seed) -> None:
        """워커별 RNG 재시딩. DataLoader 멀티워커 환경에서 fork된 워커들이
        동일한 RNG 상태를 공유하지 않도록 호출 (OnlineQECDataset.__iter__).

        기본 구현은 no-op — global numpy random만 쓰는 시뮬레이터(예: stim 기반)는
        worker_init_fn에서 이미 시드되므로 추가 작업 불필요.
        자체 rng 인스턴스를 가진 시뮬레이터는 오버라이드.
        """
        pass


class BaseDataStrategy(ABC):
    """[Level 3] 데이터 공급기"""
    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        pass