# qec_sim/config/schema.py
import yaml
import dataclasses as dc
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from itertools import product


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
    # --- 필수 필드 ---
    data_mode: str
    train_path: str
    val_path: str
    epochs: int
    batch_size: int
    output_dir: str
    optimizer: Dict[str, Any]
    criterion: Dict[str, Any]
    early_stopping: Dict[str, int]

    # --- 선택 필드 (기본값 있음) ---
    train_steps: Optional[int] = None
    val_steps: Optional[int] = None
    chunk_size: int = 10000
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    use_erasures: bool = True
    scheduler: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    preprocessor: Dict[str, Any] = field(default_factory=dict)


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

        def filter_fields(datacls, raw: dict) -> dict:
            """데이터클래스에 없는 YAML 키를 무시하고 유효한 키만 반환합니다."""
            valid_keys = {f.name for f in dc.fields(datacls)}
            return {k: v for k, v in raw.items() if k in valid_keys}

        return cls(
            code=CodeParams(**filter_fields(CodeParams, data['code'])),
            noise=NoiseParams(**filter_fields(NoiseParams, data['noise'])),
            training=TrainingConfig(**filter_fields(TrainingConfig, data['training'])),
            model=ModelConfig(**filter_fields(ModelConfig, data['model'])),
            decoder=DecoderConfig(**filter_fields(DecoderConfig, data['decoder'])),
            simulation=data['simulation']
        )

    def get_expanded_noise_configs(self) -> List[NoiseParams]:
        """리스트 형태 노이즈 설정을 Cartesian Product로 확장합니다."""
        n = self.noise
        keys = ['p_gate', 'p_meas', 'p_corr', 'p_leak']
        values = [
            getattr(n, k) if isinstance(getattr(n, k), list) else [getattr(n, k)]
            for k in keys
        ]
        return [NoiseParams(**dict(zip(keys, v))) for v in product(*values)]
