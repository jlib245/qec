# qec_sim/data/datset.py
import random
import torch
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from qec_sim.core.parameters import CodeParams, NoiseParams
# (시뮬레이터나 generator 등 기존 회로 생성 모듈 임포트)
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.core.simulator import ComplexNoiseSimulator
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
    def __init__(self, code_config_dict: dict, noise_config_dict: dict, size: int, chunk_size: int = 1000):
        self.size = size             # 1 에포크당 생성할 총 데이터 개수
        self.chunk_size = chunk_size # 한 노이즈 환경에서 한 번에 뽑아낼 샷 수 (속도 최적화의 핵심)
        
        self.code_config_dict = code_config_dict
        
        # 노이즈 값들을 리스트로 정규화
        self.noise_config_lists = {}
        required_noise_keys = ['p_gate', 'p_meas', 'p_corr', 'p_leak']
        for key in required_noise_keys:
            value = noise_config_dict[key]
            self.noise_config_lists[key] = value if isinstance(value, list) else [value]

    def __len__(self):
        return self.size

    def __iter__(self):
        # 멀티프로세싱 시, 워커별로 생성할 데이터 할당량을 나눕니다.
        worker_info = get_worker_info()
        if worker_info is None:
            # 단일 프로세스 실행 시
            worker_size = self.size
            seed = random.randint(0, 10000)
        else:
            # 멀티 프로세스 실행 시: 워커별로 할당량을 나누고 시드를 다르게 설정
            worker_size = self.size // worker_info.num_workers
            # 워커 ID를 시드에 더해 각 워커가 다른 난수 시퀀스를 갖도록 함
            seed = torch.initial_seed() % 2**32 + worker_info.id
        
        # 난수 시드 고정 (중복 방지)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        yielded_count = 0
        
        while yielded_count < worker_size:
            # 1. 이번 청크(Chunk)에 사용할 노이즈를 랜덤으로 샘플링!
            # (나중에 연속형 랜덤 값을 원하시면 이 부분만 random.uniform 등으로 바꾸면 끝입니다)
            sampled_noise_kwargs = {
                k: random.choice(v) for k, v in self.noise_config_lists.items()
            }
            noise_config = NoiseParams(**sampled_noise_kwargs)
            code_config = CodeParams(**self.code_config_dict)
            
            # 2. 회로와 시뮬레이터 빌드 (청크당 딱 1번만 실행됨 -> 초고속)
            builder = CustomCircuitBuilder(code_config, noise_config)
            circuit = builder.build()
            simulator = ComplexNoiseSimulator(circuit, noise_config)
            
            # 3. 한 번에 왕창(chunk_size만큼) 뽑아냅니다.
            current_chunk = min(self.chunk_size, worker_size - yielded_count)
            syndromes, observables, erasures = simulator.generate_data(shots=current_chunk)
            
            # 4. 뽑아낸 뭉텅이 안에서 하나씩 PyTorch 포맷으로 넘겨줍니다(yield).
            for i in range(current_chunk):
                x = np.stack([syndromes[i], erasures[i]], axis=0)
                y = observables[i]
                yield torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
                
            yielded_count += current_chunk