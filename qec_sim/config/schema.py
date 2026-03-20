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
class PauliPlusNoiseParams:
    """
    Pauli+ 시뮬레이터용 노이즈 파라미터.

    SI1000 gate noise (단일 p 기반, Table S1):
      2q gate (CZ):      p
      1q gate (H/idle):  p / 10
      measurement:       5p
      reset:             2p
      resonator idle:    2p

    Leakage 파라미터 (p와 독립적인 절대값, 디바이스 특성화 데이터 기반):
      cz_dephasing_leakage: CZ 게이트 중 |2⟩ 전이 확률 (경로 A)
      heating_rate_per_us:  열적 여기율 [1/μs] (경로 B)
      passive_decay_T1_us:  Leaked qubit T1 [μs]

    I/Q 파라미터 (Phase 2, soft measurement용):
      snr:              신호 대 잡음비
      t_meas_over_T1:   측정 시간 / T1
    """
    # SI1000 gate noise
    p: Union[float, List[float]]

    # Leakage (Phase 2) — p와 독립적인 절대값, 기본값 0 (noiseless)
    cz_dephasing_leakage: float = 0.0        # 논문값: 2e-4 (경로 A: 8e-4 × 0.25)
    heating_rate_per_us: float = 0.0         # 논문값: 1/700 (경로 B)
    passive_decay_T1_us: float = 0.0         # 논문값: 20.0
    leakage_removal_after_reset: bool = True
    data_qubit_leakage_removal_per_cycle: bool = True

    # I/Q noise (Phase 2), 기본값 0 (noiseless)
    snr: float = 0.0                         # 논문값: 10.0
    t_meas_over_T1: float = 0.0             # 논문값: 0.01

    @property
    def p_2q(self) -> float:
        return float(self.p) if not isinstance(self.p, list) else self.p[0]

    @property
    def p_1q(self) -> float:
        return self.p_2q / 10

    @property
    def p_meas(self) -> float:
        return self.p_2q * 5

    @property
    def p_reset(self) -> float:
        return self.p_2q * 2

    @property
    def p_idle(self) -> float:
        return self.p_2q / 10

    @property
    def p_resonator_idle(self) -> float:
        return self.p_2q * 2


@dataclass
class TrainingConfig:
    # --- 필수 필드 ---
    data_mode: str       # "offline" | "online"
    epochs: int
    batch_size: int
    output_dir: str
    optimizer: Dict[str, Any]
    criterion: Dict[str, Any]
    early_stopping: Dict[str, int]

    # --- 선택 필드 (기본값 있음) ---
    # offline 전용
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    # online 전용 (epoch당 생성할 샘플 수)
    train_steps: Optional[int] = None
    val_steps: Optional[int] = None
    chunk_size: int = 10000          # 시뮬레이터 1회 호출당 생성 샷 수
    # 공통
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    scheduler: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


@dataclass
class ModelConfig:
    name: str
    use_erasures: bool = True
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
