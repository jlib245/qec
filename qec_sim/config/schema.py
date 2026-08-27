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
    # name=="stim_file"일 때만 사용. 디스크의 .stim 회로를 로드 (노이즈 baked-in).
    # glob 패턴 허용 — 여러 파일 매치 시 파일당 하나의 eval task 로 확장됨.
    stim_path: Optional[str] = None


@dataclass
class NoiseParams:
    p_gate: Union[float, List[float]]
    p_meas: Union[float, List[float]]
    p_corr: Union[float, List[float]]


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

    # Crosstalk (Phase 3), 기본값 0 (noiseless)
    p_crosstalk: float = 0.0                 # 논문값: 9.5e-4 (Nature 614 Table I)
    crosstalk_alpha: float = 0.25            # Bausch 스케일: 0.25, 원래: 1.0
    crosstalk_leakage_rate: float = 0.0      # crosstalk 유래 leakage 확률

    # 게이트 시간 [μs] — heating(rate × time) 및 측정 중 T1 decay 계산에 사용.
    # 기본값은 SI1000 표준 (CX 50 ns × 4 sub-layer = 200 ns/round, 측정 500 ns).
    t_cx_us: float = 0.05                    # CX sub-layer 1개 시간
    t_meas_us: float = 0.5                   # data qubit 측정 시간

    # 개별 강도 override (None이면 p 기반 derived 값 사용, 검증/디버깅용)
    p_idle_override: float | None = None
    p_resonator_idle_override: float | None = None

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
        return self.p_idle_override if self.p_idle_override is not None else self.p_2q / 10

    @property
    def p_resonator_idle(self) -> float:
        return self.p_resonator_idle_override if self.p_resonator_idle_override is not None else self.p_2q * 2


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
    epoch_fraction: Optional[float] = None  # epoch당 데이터셋의 몇 분의 1을 볼지 (0~1]
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

    def __post_init__(self):
        if self.data_mode == "offline":
            if self.epoch_fraction is None:
                raise ValueError(
                    "offline 모드는 training.epoch_fraction 명시 필수 "
                    "(예: 0.2 = 데이터셋의 1/5을 한 epoch로 본다, 1.0 = 전체)."
                )
            if not (0.0 < self.epoch_fraction <= 1.0):
                raise ValueError(
                    f"training.epoch_fraction은 (0, 1] 범위여야 합니다. 받은 값: {self.epoch_fraction}"
                )
            if self.train_steps is not None:
                raise ValueError(
                    "training.train_steps는 online 전용입니다. offline에선 epoch_fraction을 쓰세요."
                )
            if self.val_steps is not None:
                raise ValueError(
                    "training.val_steps는 online 전용입니다."
                )
        elif self.data_mode == "online":
            if self.train_steps is None or self.train_steps <= 0:
                raise ValueError(
                    f"online 모드는 training.train_steps 양의 정수 필수. 받은 값: {self.train_steps}"
                )
            if self.val_steps is None or self.val_steps <= 0:
                raise ValueError(
                    f"online 모드는 training.val_steps 양의 정수 필수. 받은 값: {self.val_steps}"
                )
            if self.epoch_fraction is not None:
                raise ValueError(
                    "training.epoch_fraction은 offline 전용입니다."
                )


@dataclass
class ModelConfig:
    name: str
    coset_mode: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)
    preprocessor: Dict[str, Any] = field(default_factory=dict)
    pretrained: Optional[str] = None  # path to .pth for fine-tuning


@dataclass
class DecoderConfig:
    name: str
    weight_path: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """`simulation:` yaml 블록의 타입 명시 표현.

    backend: 'stim' 또는 'pauli_plus' (필수)
    shots:   레거시 필드 (현재 코드에서는 미사용 — eval shots는 CLI 인자 사용)
    pauli_plus: backend='pauli_plus'일 때 노이즈 dict.
                list-valued 필드 cartesian 확장을 위해 raw dict 유지.
                키는 PauliPlusNoiseParams 필드와 일치해야 함 (from_yaml에서 검증).
    engine:  eval 실행 엔진. 'inprocess'(기본, per-shot 파이썬 루프) 또는
             'sinter'(multiprocess + adaptive stopping). 'sinter'는 stim backend
             + algo decoder(mwpm/belief_matching/belief_matching_fast/bp_osd)에서만
             지원 — neural/pauli_plus는 inprocess 전용.
    max_shots/max_errors: engine='sinter'일 때 필수. sinter adaptive stopping 기준.
    workers: sinter worker 프로세스 수.
    """
    backend: str
    shots: Optional[int] = None
    pauli_plus: Optional[Dict[str, Any]] = None
    engine: str = "inprocess"
    max_shots: Optional[int] = None
    max_errors: Optional[int] = None
    workers: int = 8


@dataclass
class MLflowConfig:
    """`mlflow:` yaml 블록. 블록이 없거나 enable=False 면 mlflow 로깅은 비활성.

    enable=True 일 때만 학습 중 mlflow run 이 열린다. 모든 필드는 schema 에
    노출 — 사용 지점에서 getattr 기본값 같은 silent fallback 을 두지 말 것.
    """
    enable: bool = False
    experiment_name: Optional[str] = None  # None → f"{code.name}_d{code.distance}" 자동 (per-code 컨벤션)
    run_name: Optional[str] = None  # None → mlflow auto-generate
    tags: Dict[str, str] = field(default_factory=dict)
    tracking_uri: str = "sqlite:///mlruns.db"
    artifact_location: Optional[str] = None  # None → ./mlartifacts/<exp_id>
    # Model Registry (opt-in). 활성화 시 학습 종료 후 best model 을 registry 에 등록.
    register_model: bool = False
    registered_model_name: Optional[str] = None  # None → f"{model.name}_d{code.distance}"
    register_alias: Optional[str] = None  # 새 version 에 박을 alias (예: "candidate")


@dataclass
class ExperimentConfig:
    code: CodeParams
    noise: NoiseParams
    training: TrainingConfig
    model: ModelConfig
    decoder: DecoderConfig
    simulation: SimulationConfig
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        def filter_fields(datacls, raw: dict) -> dict:
            """데이터클래스에 없는 YAML 키를 무시하고 유효한 키만 반환합니다."""
            valid_keys = {f.name for f in dc.fields(datacls)}
            return {k: v for k, v in raw.items() if k in valid_keys}

        sim_raw = data['simulation']
        sim = SimulationConfig(
            backend=sim_raw['backend'],
            shots=sim_raw.get('shots'),
            pauli_plus=sim_raw.get('pauli_plus'),
            engine=sim_raw.get('engine', 'inprocess'),
            max_shots=sim_raw.get('max_shots'),
            max_errors=sim_raw.get('max_errors'),
            workers=sim_raw.get('workers', 8),
        )
        if sim.engine not in ('inprocess', 'sinter'):
            raise ValueError(
                f"simulation.engine은 'inprocess' 또는 'sinter'여야 합니다. 받은 값: {sim.engine!r}"
            )
        if sim.engine == 'sinter':
            if sim.backend != 'stim':
                raise ValueError(
                    "simulation.engine='sinter'는 backend='stim'에서만 지원됩니다 "
                    f"(받은 backend={sim.backend!r})."
                )
            if sim.max_shots is None or sim.max_errors is None:
                raise ValueError(
                    "simulation.engine='sinter'는 max_shots와 max_errors 명시 필수 "
                    "(sinter adaptive stopping 기준)."
                )
        if sim.backend == 'pauli_plus':
            import warnings
            warnings.warn(
                "backend='pauli_plus'는 deprecated입니다 (2026-07-13~). stim backend로 "
                "이전을 권장하며, Pauli+ 경로는 더 이상 확장/유지되지 않습니다.",
                DeprecationWarning,
                stacklevel=2,
            )
        if sim.pauli_plus is not None:
            valid = {f.name for f in dc.fields(PauliPlusNoiseParams)}
            unknown = set(sim.pauli_plus) - valid
            if unknown:
                raise KeyError(
                    f"simulation.pauli_plus에 알 수 없는 키: {unknown}. "
                    f"PauliPlusNoiseParams 필드: {sorted(valid)}"
                )

        mlflow_raw = data.get('mlflow')
        if mlflow_raw is None:
            mlflow_cfg = MLflowConfig()
        else:
            valid = {f.name for f in dc.fields(MLflowConfig)}
            unknown = set(mlflow_raw) - valid
            if unknown:
                raise KeyError(
                    f"mlflow 블록에 알 수 없는 키: {unknown}. "
                    f"MLflowConfig 필드: {sorted(valid)}"
                )
            mlflow_cfg = MLflowConfig(**mlflow_raw)

        return cls(
            code=CodeParams(**filter_fields(CodeParams, data['code'])),
            noise=NoiseParams(**filter_fields(NoiseParams, data['noise'])),
            training=TrainingConfig(**filter_fields(TrainingConfig, data['training'])),
            model=ModelConfig(**filter_fields(ModelConfig, data['model'])),
            decoder=DecoderConfig(**filter_fields(DecoderConfig, data['decoder'])),
            simulation=sim,
            mlflow=mlflow_cfg,
        )

    def get_expanded_noise_configs(self) -> List[NoiseParams]:
        """리스트 형태 노이즈 설정을 Cartesian Product로 확장합니다."""
        n = self.noise
        keys = ['p_gate', 'p_meas', 'p_corr']
        values = [
            getattr(n, k) if isinstance(getattr(n, k), list) else [getattr(n, k)]
            for k in keys
        ]
        return [NoiseParams(**dict(zip(keys, v))) for v in product(*values)]

    def get_expanded_pauli_plus_configs(self) -> List["PauliPlusNoiseParams"]:
        """simulation.pauli_plus의 list-valued 필드를 Cartesian product로 확장.
        Stim 백엔드의 get_expanded_noise_configs와 동일한 컨벤션."""
        pp_cfg = self.simulation.pauli_plus
        if pp_cfg is None:
            raise KeyError("simulation.pauli_plus 블록이 yaml에 없습니다.")
        if not pp_cfg:
            raise ValueError("simulation.pauli_plus 블록이 비어있습니다.")

        list_fields = {k: v for k, v in pp_cfg.items() if isinstance(v, list)}
        scalar_fields = {k: v for k, v in pp_cfg.items() if not isinstance(v, list)}

        if not list_fields:
            return [PauliPlusNoiseParams(**pp_cfg)]

        keys = list(list_fields.keys())
        values = [list_fields[k] for k in keys]
        return [PauliPlusNoiseParams(**scalar_fields, **dict(zip(keys, combo)))
                for combo in product(*values)]
