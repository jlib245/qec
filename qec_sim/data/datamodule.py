# qec_sim/data/datamodule.py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod

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
        
        # 메타데이터 파악을 위해 첫 파일의 헤더만 미리 로드
        sample = np.load(train_path)
        self._num_detectors = sample['syndromes'].shape[1]
        self._num_observables = sample['observables'].shape[1]

    @property
    def num_detectors(self): return self._num_detectors
    
    @property
    def num_observables(self): return self._num_observables

    def _create_dataset(self, path: str) -> TensorDataset:
        data = np.load(path)
        syndromes = data['syndromes']
        erasures = data.get('erasures', np.zeros_like(syndromes))
        observables = data['observables']
        
        # (Batch, 2채널, Detectors) 형태로 결합
        x = torch.tensor(np.stack([syndromes, erasures], axis=1), dtype=torch.float32)
        y = torch.tensor(observables, dtype=torch.float32)
        return TensorDataset(x, y)

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