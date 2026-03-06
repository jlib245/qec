# qec_sim/config/schema.py
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from itertools import product # 경우의 수 확장을 위해 추가

@dataclass
class CodeParams:
    name: str
    distance: int
    rounds: int

@dataclass
class NoiseParams:
    p_gate: Union[float, List[float]]
    p_meas: Union[float, List[float]]
    p_corr: Union[float, List[float]]
    p_leak: Union[float, List[float]]

@dataclass
class TrainingConfig:
    data_mode: str
    train_path: str
    val_path: str
    train_steps: Optional[int]
    val_steps: Optional[int]
    epochs: int
    batch_size: int
    chunk_size: int

    num_workers: int
    prefetch_factor: int
    pin_memory: bool

    optimizer: Dict[str, Any]
    criterion: Dict[str, Any]
    output_dir: str 
    early_stopping: Dict[str, int]
    scheduler: Dict[str, Any]

@dataclass
class ModelConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecoderConfig:
    name: str
    weight_path: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    code: CodeParams
    noise: NoiseParams
    training: TrainingConfig
    model: ModelConfig
    decoder: DecoderConfig
    simulation: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls(
            code=CodeParams(**data['code']),
            noise=NoiseParams(**data['noise']),
            training=TrainingConfig(**data['training']),
            model=ModelConfig(**data['model']),
            decoder=DecoderConfig(**data['decoder']),
            simulation=data['simulation']
        )

    def get_expanded_noise_configs(self) -> List[NoiseParams]:
        """YAML의 리스트 형태 노이즈 설정을 모든 경우의 수(Cartesian Product)로 확장합니다."""
        n = self.noise
        keys = ['p_gate', 'p_meas', 'p_corr', 'p_leak']
        
        # 각 속성이 리스트면 그대로 사용, 단일값이면 리스트로 래핑
        values = [getattr(n, k) if isinstance(getattr(n, k), list) else [getattr(n, k)] for k in keys]
        
        combinations = product(*values)
        return [NoiseParams(**dict(zip(keys, v))) for v in combinations]