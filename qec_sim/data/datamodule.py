import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from .dataset import OfflineQECDataset, OnlineQECDataset
from .generator import DatasetGenerator
from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.core.builder import CustomCircuitBuilder

class QECDataModule:
    def __init__(self, config: dict):
        self.config = config
        self.train_config = config['training']
        self.data_mode = self.train_config['data_mode']
        self.batch_size = self.train_config['batch_size']
        
        self.num_detectors = 0
        self.num_observables = 0
        self.train_dataset = None
        self.val_dataset = None
        
        self._setup()

    def _setup(self):
        print(f"\n데이터 로드 모드: {self.data_mode}")
        
        # [수동 입력 필수 체크] 값이 없으면 여기서 KeyError로 중단됨
        req_train_steps = self.train_config['train_steps']
        req_val_steps = self.train_config['val_steps']

        if self.data_mode == 'offline':
            train_path = self.train_config['train_path']
            val_path = self.train_config['val_path']
            
            # 파일 없으면 생성 (이때도 simulation.shots를 수동으로 적어야 함)
            if not os.path.exists(train_path):
                gen_shots = self.config['simulation']['shots']
                generator = DatasetGenerator(CodeParams(**self.config['code']), self.config['noise'])
                generator.generate_and_save(shots=gen_shots, save_dir=os.path.dirname(train_path), filename="train")
                generator.generate_and_save(shots=gen_shots // 10, save_dir=os.path.dirname(train_path), filename="val")

            # 파일 로드
            full_train_ds = OfflineQECDataset(train_path)
            full_val_ds = OfflineQECDataset(val_path)

            # [수동 설정 검증] 파일보다 더 큰 숫자를 적었는지 확인
            if req_train_steps > len(full_train_ds):
                raise ValueError(f"YAML의 train_steps({req_train_steps})가 실제 파일({len(full_train_ds)})보다 큽니다.")

            # 사용자가 적은 숫자만큼만 정확히 슬라이싱
            self.train_dataset = Subset(full_train_ds, range(req_train_steps))
            self.val_dataset = Subset(full_val_ds, range(req_val_steps))
            
            sample_x, _ = full_train_ds[0]
            self.num_detectors = sample_x.shape[1]

        elif self.data_mode == 'online':
            # 사용자님이 말씀하신 "이전 그 코드"와 동일한 로직입니다.
            # 다만 req_train_steps를 수동으로 위에서 받아왔음을 보장합니다.
            self.train_dataset = OnlineQECDataset(
                self.config['code'], self.config['noise'], size=req_train_steps
            )
            self.val_dataset = OnlineQECDataset(
                self.config['code'], self.config['noise'], size=req_val_steps
            )
            
            # 차원 파악용 (리스트 대응)
            temp_noise = {k: (v[0] if isinstance(v, list) else v) for k, v in self.config['noise'].items()}
            temp_circuit = CustomCircuitBuilder(CodeParams(**self.config['code']), NoiseParams(**temp_noise)).build()
            self.num_detectors = temp_circuit.num_detectors

            
    def get_loaders(self):
        # Online 모드(IterableDataset)는 shuffle=False여야 함
        is_online = (self.data_mode == 'online')
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=not is_online, 
            num_workers=4,
            pin_memory=True
        ), DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )