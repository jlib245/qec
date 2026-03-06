# qec_sim/data/datamodule.py
import numpy as np
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from qec_sim.data.dataset import OfflineQECDataset, OnlineQECDataset
from qec_sim.circuit.simulator import SimulatorPool

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
    def __init__(self, config):
        self.config = config
        self.train_path = config.training.train_path
        self.val_path = config.training.val_path
        self.batch_size = config.training.batch_size
        
        # 메타데이터 파악을 위해 mmap_mode='r'을 사용하여 메모리 효율적으로 헤더만 로드
        sample = np.load(self.train_path, mmap_mode='r')
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
        
        nw = self.config.training.num_workers
        pm = self.config.training.pin_memory
        # PyTorch : num_workers가 0일 때는 prefetch_factor를 넘기면 에러 발생.
        pf = self.config.training.prefetch_factor if nw > 0 else None
        
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=nw, prefetch_factor=pf, pin_memory=pm
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=nw, prefetch_factor=pf, pin_memory=pm
        )
        return train_loader, val_loader

# 3. 온라인 시뮬레이터 생성 전략
class OnlineDataStrategy(DataStrategy):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.training.batch_size
        
        noise_configs = self.config.get_expanded_noise_configs()
        pool = SimulatorPool(self.config.code, noise_configs)
        self._num_detectors = pool.num_detectors
        self._num_observables = pool.num_observables

    @property
    def num_detectors(self): return self._num_detectors
    @property
    def num_observables(self): return self._num_observables

    def get_loaders(self):
        noise_configs = self.config.get_expanded_noise_configs()
        
        train_ds = OnlineQECDataset(
            self.config.code, noise_configs, 
            epoch_size=self.config.training.train_steps * self.batch_size, 
            chunk_size=self.config.training.chunk_size
        )
        
        val_ds = OnlineQECDataset(
            self.config.code, noise_configs, 
            epoch_size=self.config.training.val_steps * self.batch_size, 
            chunk_size=self.config.training.chunk_size
        )
        
        nw = self.config.training.num_workers
        pm = self.config.training.pin_memory
        pf = self.config.training.prefetch_factor if nw > 0 else None
        
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=nw, prefetch_factor=pf, pin_memory=pm
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=nw, prefetch_factor=pf, pin_memory=pm
        )
        return train_loader, val_loader

# 4. 데이터 모듈 (Context)
class QECDataModule:
    def __init__(self, strategy: DataStrategy):
        self.strategy = strategy

    @property
    def num_detectors(self): return self.strategy.num_detectors
    
    @property
    def num_observables(self): return self.strategy.num_observables

    def get_loaders(self):
        return self.strategy.get_loaders()