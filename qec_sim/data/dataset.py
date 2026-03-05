# qec_sim/data/dataset.py
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import random

from qec_sim.circuit.simulator import SimulatorPool

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
        self.epoch_size = epoch_size
        self.chunk_size = chunk_size
        self.pool = SimulatorPool(code_config, noise_configs)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.id + int(torch.initial_seed()) % 2**32)
            random.seed(worker_info.id + 100)
        
        shots_generated = 0
        while shots_generated < self.epoch_size:
            current_chunk = min(self.chunk_size, self.epoch_size - shots_generated)
            
            # Pool에서 랜덤 시뮬레이터를 꺼내서 사용.
            simulator = self.pool.get_random_simulator()
            syndromes, observables, erasures = simulator.generate_data(shots=current_chunk)
            shots_generated += current_chunk

            for i in range(current_chunk):
                x = np.stack([syndromes[i], erasures[i]], axis=0)
                y = observables[i]
                yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)