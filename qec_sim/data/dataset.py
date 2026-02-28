# qec_sim/data/datset.py

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np

# 1. 오프라인 데이터셋 (미리 생성된 .npz 파일 로드)
class OfflineQECDataset(Dataset):
    def __init__(self, filepath: str):
        """저장된 npz 파일을 RAM에 한 번에 올려두고 사용합니다."""
        data = np.load(filepath)
        self.syndromes = data['syndromes']
        self.erasures = data['erasures']
        self.observables = data['observables']

    def __len__(self):
        return len(self.syndromes)

    def __getitem__(self, idx):
        # 1. 각각의 데이터 가져오기
        s = self.syndromes[idx]
        e = self.erasures[idx]
        
        # 2. 채널 병합: 딥러닝 모델이 두 정보를 모두 볼 수 있도록 (2, num_detectors) 형태로 쌓음
        x = np.stack([s, e], axis=0)
        
        # 3. 정답 라벨
        y = self.observables[idx]

        # PyTorch 텐서로 변환하여 반환
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 2. 온라인 데이터셋 (실시간 무한 생성기)
class OnlineQECDataset(IterableDataset):
    def __init__(self, code_config, noise_config, epoch_size=100000, chunk_size=10000):
        """
        학습이 돌아가는 동안 실시간으로 Stim을 돌려 데이터를 생성합니다.
        - epoch_size: 1 에포크당 생성할 총 데이터 개수
        - chunk_size: 한 번 Stim을 부를 때 몇 개씩 찍어낼지 (너무 작으면 느려짐)
        """
        self.code_config = code_config
        self.noise_config = noise_config
        self.epoch_size = epoch_size
        self.chunk_size = chunk_size

    def __len__(self):
        # DataLoader가 epoch_size를 알 수 있도록 합니다. 실제로는 무한 생성이지만, 1 epoch당 생성할 데이터 개수를 제한합니다.
        return self.epoch_size
    
    def __iter__(self):
        # 멀티프로세싱 충돌을 막기 위해 시뮬레이터는 __iter__ 안에서 초기화합니다.
        from qec_sim.core.builder import CustomCircuitBuilder
        from qec_sim.core.simulator import ComplexNoiseSimulator
        
        builder = CustomCircuitBuilder(self.code_config, self.noise_config)
        simulator = ComplexNoiseSimulator(builder.build(), self.noise_config)

        shots_generated = 0
        while shots_generated < self.epoch_size:
            current_chunk = min(self.chunk_size, self.epoch_size - shots_generated)
            
            # C++ 백엔드로 한 번에 뭉텅이(chunk) 데이터 생성
            syndromes, observables, erasures = simulator.generate_data(shots=current_chunk)
            shots_generated += current_chunk

            # 생성된 뭉텅이 안에서 하나씩 꺼내어 PyTorch 포맷으로 변환 후 전달(yield)
            for i in range(current_chunk):
                x = np.stack([syndromes[i], erasures[i]], axis=0)
                y = observables[i]
                yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)