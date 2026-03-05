# qec_sim/data/dataset.py
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random

from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator

class OfflineQECDataset(Dataset):
    def __init__(self, filepath: str):
        data = np.load(filepath, mmap_mode='r')
        self.syndromes = data['syndromes']
        self.erasures = data['erasures']
        self.observables = data['observables']

    def __len__(self):
        return len(self.syndromes)

    def __getitem__(self, idx):
        s, e = self.syndromes[idx], self.erasures[idx]
        x = np.stack([s, e], axis=0)
        y = self.observables[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class OnlineQECDataset(IterableDataset):
    def __init__(self, code_config, noise_configs, epoch_size, chunk_size):
        self.code_config = code_config
        self.noise_configs = noise_configs if isinstance(noise_configs, list) else [noise_configs]
        self.epoch_size = epoch_size
        self.chunk_size = chunk_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # 각 워커마다 서로 다른 시드를 갖게 함
            np.random.seed(worker_info.id + int(torch.initial_seed()) % 2**32)
            random.seed(worker_info.id + 100)
        
        # 모든 노이즈 환경에 대해 시뮬레이터를 미리 준비
        simulators = []
        for n_config in self.noise_configs:
            # 설정 파일의 code.name (예: "surface_code")을 사용
            builder = build_circuit(self.code_config.name, self.code_config, n_config)
            simulators.append(CircuitNoiseSimulator(builder.build(), n_config))

        shots_generated = 0
        while shots_generated < self.epoch_size:
            current_chunk = min(self.chunk_size, self.epoch_size - shots_generated)
            
            # 청크를 뽑을 때마다 랜덤한 노이즈 환경 시뮬레이터를 골라서 실행
            simulator = random.choice(simulators)
            syndromes, observables, erasures = simulator.generate_data(shots=current_chunk)
            shots_generated += current_chunk

            for i in range(current_chunk):
                x = np.stack([syndromes[i], erasures[i]], axis=0)
                y = observables[i]
                yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)