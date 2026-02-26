# qec_sim/data/datamodule.py

from torch.utils.data import DataLoader
from .dataset import OfflineQECDataset, OnlineQECDataset
from qec_sim.core.parameters import CodeParams, NoiseParams
from qec_sim.core.builder import CustomCircuitBuilder

class QECDataModule:
    """
    데이터 로드, 전처리, DataLoader 생성을 전담하는 모듈입니다.
    """
    def __init__(self, config: dict):
        self.config = config
        self.train_config = config.get('training', {})
        self.data_mode = self.train_config.get('data_mode', 'offline')
        self.batch_size = self.train_config.get('batch_size', 512)
        
        # 모델 생성에 필요한 정보들
        self.num_detectors = 0
        self.num_observables = 0
        
        # 데이터셋 객체
        self.train_dataset = None
        self.val_dataset = None
        
        # 초기화 시 바로 데이터셋을 세팅합니다.
        self._setup()

    def _setup(self):
        print(f"\n데이터 로드 모드: {self.data_mode}")
        if self.data_mode == 'offline':
            train_path = self.train_config.get('train_path', 'datasets/d5_complex_noise/train.npz')
            val_path = self.train_config.get('val_path', 'datasets/d5_complex_noise/val.npz')
            
            self.train_dataset = OfflineQECDataset(train_path)
            self.val_dataset = OfflineQECDataset(val_path)
            
            # 첫 번째 샘플을 뽑아 형태(shape)를 확인합니다.
            sample_x, sample_y = self.train_dataset[0]
            self.num_detectors = sample_x.shape[1]
            self.num_observables = sample_y.shape[0]
            
        elif self.data_mode == 'online':
            code_config = CodeParams(**self.config.get('code', {}))
            noise_config = NoiseParams(**self.config.get('noise', {}))
            
            self.train_dataset = OnlineQECDataset(code_config, noise_config, epoch_size=self.train_config.get('epoch_size', 100000))
            self.val_dataset = OnlineQECDataset(code_config, noise_config, epoch_size=self.train_config.get('val_size', 10000))
            
            # 임시 회로를 생성하여 디텍터 수를 파악합니다.
            temp_circuit = CustomCircuitBuilder(code_config, noise_config).build()
            self.num_detectors = temp_circuit.num_detectors
            self.num_observables = temp_circuit.num_observables
        else:
            raise ValueError("data_mode는 'offline' 또는 'online'이어야 합니다.")

    def get_loaders(self):
        """준비된 데이터셋을 묶어 Train, Val DataLoader를 반환합니다."""
        is_iterable = self.data_mode == 'online'
        # Online(Iterable) 모드일 때는 shuffle=False여야 합니다.
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=not is_iterable)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader