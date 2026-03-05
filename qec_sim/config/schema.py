# qec_sim/config/schema.py
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union

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
    train_steps: int
    val_steps: int 
    epochs: int
    batch_size: int
    chunk_size: int
    optimizer: Dict[str, Any]
    output_dir: str 
    save_path: str 
    early_stopping: int
    scheduler: Dict[str, Any] = field(default_factory=dict)
    

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
            
        # get() 없이 키 직접 접근으로 필수값 누락 시 조기 에러 발생 (Fail-Fast)
        return cls(
            code=CodeParams(**data['code']),
            noise=NoiseParams(**data['noise']),
            training=TrainingConfig(**data['training']),
            model=ModelConfig(**data['model']),
            decoder=DecoderConfig(**data['decoder']),
            simulation=data['simulation']
        )