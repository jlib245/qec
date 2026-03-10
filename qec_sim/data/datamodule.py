# qec_sim/data/datamodule.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Callable, Optional, List, Tuple, Union


def _worker_init_fn(worker_id: int):
    """DataLoader 워커마다 독립적이고 재현 가능한 seed를 설정합니다."""
    # 메인 프로세스 seed + worker_id → 워커별 고유 seed
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    import random
    random.seed(seed)


class QECRawDataset(Dataset):
    """전처리기가 명시한 데이터만 하드디스크에서 꺼내어 cpu_transform을 거쳐 반환합니다."""

    def __init__(self, npz_path: str, required_keys: List[str], cpu_transform: Optional[Callable] = None):
        data = np.load(npz_path)

        self.data_dict = {}
        for key in required_keys:
            if key not in data:
                raise ValueError(
                    f"전처리기가 '{key}'를 요구했으나 데이터셋에 없습니다. "
                    f"파일 내 키: {list(data.keys())}"
                )
            self.data_dict[key] = torch.tensor(data[key])

        # 'observables' 또는 'logical_outcomes' 키 모두 지원
        if 'observables' in data:
            self.labels = torch.tensor(data['observables'])
        elif 'logical_outcomes' in data:
            self.labels = torch.tensor(data['logical_outcomes'])
        else:
            raise ValueError(f"라벨 키('observables' 또는 'logical_outcomes')가 데이터셋에 없습니다. 파일 내 키: {list(data.keys())}")

        self.cpu_transform = cpu_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {key: self.data_dict[key][idx] for key in self.data_dict}
        if self.cpu_transform:
            sample = self.cpu_transform(sample)
        return sample, self.labels[idx]


class OfflineDataStrategy:
    """오프라인 .npz 파일에서 데이터를 로드하는 전략."""

    def __init__(self, config, required_keys: List[str], cpu_transform: Optional[Callable] = None):
        self.config = config
        self.required_keys = required_keys
        self.cpu_transform = cpu_transform
        self._train_dataset: Optional[QECRawDataset] = None
        self._val_dataset: Optional[QECRawDataset] = None

    def prepare(self) -> None:
        tc = self.config.training
        self._train_dataset = QECRawDataset(
            npz_path=tc.train_path,
            required_keys=self.required_keys,
            cpu_transform=self.cpu_transform,
        )
        self._val_dataset = QECRawDataset(
            npz_path=tc.val_path,
            required_keys=self.required_keys,
            cpu_transform=self.cpu_transform,
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self._train_dataset is None or self._val_dataset is None:
            raise RuntimeError("prepare()를 먼저 호출하세요.")

        tc = self.config.training
        # num_workers=0 이면 prefetch_factor 사용 불가
        pf = tc.prefetch_factor if tc.num_workers > 0 else None

        train_loader = DataLoader(
            self._train_dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            self._val_dataset,
            batch_size=tc.batch_size,
            shuffle=False,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
        )
        return train_loader, val_loader


class OnlineQECDataset(IterableDataset):
    """시뮬레이터에서 on-the-fly로 데이터를 생성하는 IterableDataset."""

    def __init__(
        self,
        simulator_pool,
        required_keys: List[str],
        epoch_samples: int,
        chunk_size: int,
        cpu_transform: Optional[Callable] = None,
    ):
        self.simulator_pool = simulator_pool
        self.required_keys = required_keys
        self.epoch_samples = epoch_samples
        self.chunk_size = chunk_size
        self.cpu_transform = cpu_transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers, worker_id = 1, 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # 각 워커는 epoch_samples / num_workers 만큼 생성
        per_worker = self.epoch_samples // num_workers
        # 나머지는 마지막 워커에게
        if worker_id == num_workers - 1:
            per_worker += self.epoch_samples - per_worker * num_workers

        generated = 0
        while generated < per_worker:
            sim = self.simulator_pool.get_random_simulator()
            batch = min(self.chunk_size, per_worker - generated)
            raw = sim.generate_data(batch)

            observables = raw['observables']
            for i in range(len(observables)):
                sample = {
                    k: torch.tensor(raw[k][i], dtype=torch.float32)
                    for k in self.required_keys
                    if k in raw
                }

                if self.cpu_transform:
                    sample = self.cpu_transform(sample)

                label = torch.tensor(observables[i], dtype=torch.float32)
                yield sample, label
                generated += 1
                if generated >= per_worker:
                    break


class OnlineDataStrategy:
    """시뮬레이터에서 실시간으로 데이터를 생성하는 전략."""

    def __init__(self, config, simulator_pool, required_keys: List[str], cpu_transform: Optional[Callable] = None):
        self.config = config
        self.simulator_pool = simulator_pool
        self.required_keys = required_keys
        self.cpu_transform = cpu_transform
        self._train_dataset: Optional[OnlineQECDataset] = None
        self._val_dataset: Optional[OnlineQECDataset] = None

    def prepare(self) -> None:
        tc = self.config.training
        train_samples = tc.train_steps or 10000
        val_samples = tc.val_steps or 2000

        self._train_dataset = OnlineQECDataset(
            simulator_pool=self.simulator_pool,
            required_keys=self.required_keys,
            epoch_samples=train_samples,
            chunk_size=tc.chunk_size,
            cpu_transform=self.cpu_transform,
        )
        self._val_dataset = OnlineQECDataset(
            simulator_pool=self.simulator_pool,
            required_keys=self.required_keys,
            epoch_samples=val_samples,
            chunk_size=tc.chunk_size,
            cpu_transform=self.cpu_transform,
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self._train_dataset is None or self._val_dataset is None:
            raise RuntimeError("prepare()를 먼저 호출하세요.")

        tc = self.config.training
        pf = tc.prefetch_factor if tc.num_workers > 0 else None

        train_loader = DataLoader(
            self._train_dataset,
            batch_size=tc.batch_size,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            self._val_dataset,
            batch_size=tc.batch_size,
            num_workers=tc.num_workers,
            pin_memory=tc.pin_memory,
            prefetch_factor=pf,
            worker_init_fn=_worker_init_fn if tc.num_workers > 0 else None,
        )
        return train_loader, val_loader


class QECDataModule:
    """데이터 전략을 감싸는 모듈."""

    def __init__(self, strategy: Union[OfflineDataStrategy, OnlineDataStrategy]):
        self.strategy = strategy

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.strategy.get_loaders()
