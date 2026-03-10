import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, List

class QECRawDataset(Dataset):
    """전처리기가 명시한 데이터만 하드디스크에서 꺼내어 cpu_transform을 거쳐 반환합니다."""
    def __init__(self, npz_path: str, required_keys: List[str], cpu_transform: Optional[Callable] = None):
        data = np.load(npz_path)
        
        self.data_dict = {}
        for key in required_keys:
            if key not in data:
                raise ValueError(f"전처리기가 '{key}'를 요구했으나 데이터셋에 없습니다!")
            self.data_dict[key] = torch.tensor(data[key])
            
        self.labels = torch.tensor(data['logical_outcomes'])
        self.cpu_transform = cpu_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. 1D Raw 데이터 딕셔너리 생성
        sample = {key: self.data_dict[key][idx] for key in self.data_dict}
        
        # 2. 전처리기의 CPU 정책 적용
        if self.cpu_transform:
            sample = self.cpu_transform(sample)
            
        return sample, self.labels[idx]

# (QECDataModule 및 OfflineDataStrategy 구현체는 위 QECRawDataset을 사용하여 로더를 생성하도록 구현)