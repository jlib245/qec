# qec_sim/trainer/factory.py
from qec_sim.config.schema import ExperimentConfig, PauliPlusNoiseParams
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import SimulatorPool
from qec_sim.models.wrapper import PreprocessorWrapper
from qec_sim.models.registry import get_model_class
from qec_sim.data.registry import get_preprocessor_class
from qec_sim.data.datamodule import QECDataModule, OfflineDataStrategy, OnlineDataStrategy


class ComponentFactory:
    @staticmethod
    def build_system(config: ExperimentConfig):
        noise_configs = config.get_expanded_noise_configs()

        # 회로 빌드 (detector 좌표 및 메타데이터 추출용)
        circuit = build_circuit(
            config.code.name,
            config.code,
            noise_configs[0]
        ).build()

        # 1. 모델 클래스 확인 → 필요한 전처리기 이름 조회
        core_model_cls = get_model_class(config.model.name)
        preprocessor_name = getattr(core_model_cls, 'REQUIRED_PREPROCESSOR', 'flat')

        # 2. simulator_pool 미리 빌드 (online 모드 또는 preprocessor가 필요로 할 때)
        data_mode = config.training.data_mode
        simulator_pool = _build_simulator_pool(config) if data_mode == "online" else None

        # 3. 전처리기 생성 (각 클래스가 from_config로 필요한 정보 직접 추출)
        preprocessor_cls = get_preprocessor_class(preprocessor_name)
        preprocessor = preprocessor_cls.from_config(config, circuit, simulator_pool)

        # 4. 데이터 모듈 생성 (data_mode에 따라 전략 선택)
        if data_mode == "online":
            strategy = OnlineDataStrategy(
                config=config,
                simulator_pool=simulator_pool,
                required_keys=preprocessor.required_data_keys,
                cpu_transform=preprocessor.cpu_transform,
            )
        elif data_mode == "offline":
            strategy = OfflineDataStrategy(
                config=config,
                required_keys=preprocessor.required_data_keys,
                cpu_transform=preprocessor.cpu_transform,
            )
        else:
            raise ValueError(f"알 수 없는 data_mode: '{data_mode}'. 'online' 또는 'offline'을 사용하세요.")

        datamodule = QECDataModule(strategy=strategy)

        # 5. 코어 모델 생성 (전처리기의 출력 규격 주입)
        model_kwargs = preprocessor.get_model_kwargs()
        model_kwargs["num_observables"] = circuit.num_observables
        model_kwargs["code_distance"] = config.code.distance
        model_kwargs.update(config.model.kwargs)

        core_model = core_model_cls(**model_kwargs)

        # 5. 래퍼로 조립 (코어 모델 + GPU 전처리 함수)
        wrapped_model = PreprocessorWrapper(core_model, preprocessor)

        return datamodule, wrapped_model


def _build_simulator_pool(config: ExperimentConfig) -> SimulatorPool:
    """
    simulation.backend 키에 따라 SimulatorPool 생성.
      stim (기본값): 기존 Stim 기반 CircuitNoiseSimulator
      pauli_plus:    PauliPlusSimulator (Phase 1+2)
    """
    backend = config.simulation.get('backend', 'stim')

    if backend == 'stim':
        noise_configs = config.get_expanded_noise_configs()
        return SimulatorPool.from_stim(config.code, noise_configs)

    elif backend == 'pauli_plus':
        from qec_sim.circuit.pauli_plus import PauliPlusSimulator
        pp_cfg = config.simulation.get('pauli_plus', {})
        noise = PauliPlusNoiseParams(**pp_cfg)
        sim = PauliPlusSimulator(config.code, noise)
        return SimulatorPool([sim])

    else:
        raise ValueError(f"알 수 없는 simulation backend: '{backend}'. 'stim' 또는 'pauli_plus'를 사용하세요.")
