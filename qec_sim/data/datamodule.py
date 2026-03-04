# qec_sim/data/datamodule.py
import os
from torch.utils.data import DataLoader
from .dataset import OfflineQECDataset, OnlineQECDataset
from qec_sim.core.parameters import CodeParams, get_noise_combinations
from qec_sim.core.builder import CustomCircuitBuilder
from qec_sim.data.generator import DatasetGenerator  # 제너레이터 임포트

class QECDataModule:
    """
    데이터 로드, 전처리, DataLoader 생성을 전담하는 모듈입니다.
    """
    def __init__(self, config: dict):
        self.config = config
        self.train_config = config['training']
        self.data_mode = self.train_config['data_mode']
        self.batch_size = self.train_config['batch_size']
        self.chunk_size = self.train_config['chunk_size'] 
        
        self.num_detectors = 0
        self.num_observables = 0
        self.train_dataset = None
        self.val_dataset = None
        
        self._setup()

    def _setup(self):
        print(f"\n데이터 로드 모드: {self.data_mode}")
        if self.data_mode == 'offline':
            self._setup_offline()
        elif self.data_mode == 'online':
            self._setup_online()
        else:
            raise ValueError("data_mode는 'offline' 또는 'online'이어야 합니다.")

    def _setup_offline(self):
        train_path = self.train_config['train_path']
        val_path = self.train_config['val_path']
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("⚠️ 지정된 경로에 오프라인 데이터가 없습니다. 자동 생성을 시작합니다.")
            self._generate_missing_data(train_path, val_path)
            
        self.train_dataset = OfflineQECDataset(train_path)
        self.val_dataset = OfflineQECDataset(val_path)
        
        sample_x, sample_y = self.train_dataset[0]
        self.num_detectors = sample_x.shape[1]
        self.num_observables = sample_y.shape[0]

    def _generate_missing_data(self, train_path: str, val_path: str):
        code_config = CodeParams(**self.config['code'])
        # ⭐️ 딕셔너리에 있는 모든 노이즈 조합(18가지 등)을 리스트로 뽑아냄
        noise_configs = get_noise_combinations(self.config['noise'])
        generator = DatasetGenerator(code_config, noise_configs)
        
        if not os.path.exists(train_path):
            train_dir, train_file = os.path.split(train_path)
            train_name, _ = os.path.splitext(train_file)
            train_shots = self.train_config['train_steps']               
            generator.generate_and_save(shots=train_shots, save_dir=train_dir, filename=train_name)
        if not os.path.exists(val_path):
            val_dir, val_file = os.path.split(val_path)
            val_name, _ = os.path.splitext(val_file)
            val_shots = self.train_config['val_steps']
            generator.generate_and_save(shots=val_shots, save_dir=val_dir, filename=val_name)

    def _setup_online(self):
        """온라인 학습을 위한 설정: 학습은 실시간 생성, 검증은 메모리 고정"""
        code_config = CodeParams(**self.config['code'])
        noise_configs = get_noise_combinations(self.config['noise'])
        
        # 1. 학습 데이터 세팅 (Online)
        self.train_dataset = OnlineQECDataset(
            code_config, noise_configs, 
            epoch_size=self.train_config['train_steps']
        )
        
        # 2. 검증 데이터 세팅 (In-Memory 고정)
        self.val_dataset = self._create_in_memory_val_dataset(code_config, noise_configs)
        
        # 3. 모델 빌드를 위한 메타데이터 파악
        temp_circuit = CustomCircuitBuilder(code_config, noise_configs[0]).build()
        self.num_detectors = temp_circuit.num_detectors
        self.num_observables = temp_circuit.num_observables

    def _create_in_memory_val_dataset(self, code_config, noise_configs):
        """검증 데이터를 메모리에 생성하고 TensorDataset으로 반환"""
        import torch
        import numpy as np
        from torch.utils.data import TensorDataset
        from qec_sim.core.simulator import ComplexNoiseSimulator
        
        val_shots = self.train_config['val_steps']
        total_val_shots = val_shots * len(noise_configs)
        
        print(f"  ->  [In-Memory] 검증 데이터 {total_val_shots}개 생성 중 (RAM 상주)...")
        
        all_x, all_y = [], []
        
        for n_config in noise_configs:
            # 시뮬레이션 및 데이터 생성
            builder = CustomCircuitBuilder(code_config, n_config)
            simulator = ComplexNoiseSimulator(builder.build(), n_config)
            syndromes, observables, erasures = simulator.generate_data(shots=val_shots)
            
            # 전처리: (shots, 2, num_detectors) 형태로 채널 병합
            x = np.stack([syndromes, erasures], axis=1) 
            
            all_x.append(x)
            all_y.append(observables)
            
        # 데이터를 합치고 텐서로 변환
        final_x = np.concatenate(all_x, axis=0)
        final_y = np.concatenate(all_y, axis=0)
        
        dataset = TensorDataset(
            torch.tensor(final_x, dtype=torch.float32), 
            torch.tensor(final_y, dtype=torch.float32)
        )
        print("  -> 검증 데이터 생성 완료")
        return dataset

    def get_loaders(self):
        is_iterable = self.data_mode == 'online'
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=not is_iterable, 
                                  num_workers=8,
                                  pin_memory=True, # GPU 전송 속도 향상
                                  prefetch_factor=4  if is_iterable else None # 온라인 모드일 때 데이터 로딩과 모델 학습 병렬화
        )
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True
            )

        return train_loader, val_loader