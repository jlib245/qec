# qec_sim/data/datamodule.py
import numpy as np
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from qec_sim.data.dataset import OfflineQECDataset

# 1. 전략(Strategy) 인터페이스 정의
class DataStrategy(ABC):
    @abstractmethod
    def get_loaders(self) -> tuple[DataLoader, DataLoader]:
        pass

    @property
    @abstractmethod
    def num_detectors(self) -> int:
        pass

    @property
    @abstractmethod
    def num_observables(self) -> int:
        pass

# 2. 오프라인 파일 로드 전략
class OfflineDataStrategy(DataStrategy):
    def __init__(self, train_path: str, val_path: str, batch_size: int):
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        
        # 메타데이터 파악을 위해 mmap_mode='r'을 사용하여 메모리 효율적으로 헤더만 로드
        sample = np.load(train_path, mmap_mode='r')
        self._num_detectors = sample['syndromes'].shape[1]
        self._num_observables = sample['observables'].shape[1]

    @property
    def num_detectors(self): return self._num_detectors
    
    @property
    def num_observables(self): return self._num_observables

    def _create_dataset(self, path: str) -> OfflineQECDataset:
        # dataset.py의 OfflineQECDataset에 책임을 위임.
        return OfflineQECDataset(path)

    def get_loaders(self):
        train_ds = self._create_dataset(self.train_path)
        val_ds = self._create_dataset(self.val_path)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

# 3. 데이터 모듈 (Context)
class QECDataModule:
    def __init__(self, strategy: DataStrategy):
        self.strategy = strategy

    @property
    def num_detectors(self): return self.strategy.num_detectors
    
    @property
    def num_observables(self): return self.strategy.num_observables

    def get_loaders(self):
        return self.strategy.get_loaders()