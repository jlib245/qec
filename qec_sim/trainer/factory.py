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

        # 1. 모델 클래스 확인 → 필요한 전처리기 이름 조회 (모든 모델은 클래스 속성으로 명시).
        core_model_cls = get_model_class(config.model.name)
        if not hasattr(core_model_cls, 'REQUIRED_PREPROCESSOR'):
            raise AttributeError(
                f"모델 '{config.model.name}'에 REQUIRED_PREPROCESSOR 클래스 속성이 없습니다."
            )
        preprocessor_name = core_model_cls.REQUIRED_PREPROCESSOR

        # 2. simulator_pool 미리 빌드 (online 모드는 학습 데이터 생성용,
        # offline 모드는 preprocessor가 ancilla 좌표 같은 메타데이터를 풀에서 추출하므로 필요)
        data_mode = config.training.data_mode
        simulator_pool = _build_simulator_pool(config)

        # 3. 전처리기 생성 (각 클래스가 from_config로 필요한 정보 직접 추출)
        preprocessor_cls = get_preprocessor_class(preprocessor_name)
        preprocessor = preprocessor_cls.from_config(config, circuit, simulator_pool)

        # 3.5. Coset mode: DEM에서 LUT 생성
        coset_mode = config.model.coset_mode
        coset_lut = None
        if coset_mode:
            from qec_sim.decoders.lut import build_detector_lut
            coset_lut = build_detector_lut(circuit)

        # 4. 데이터 모듈 생성 (data_mode에 따라 전략 선택)
        if data_mode == "online":
            strategy = OnlineDataStrategy(
                config=config,
                simulator_pool=simulator_pool,
                required_keys=preprocessor.required_data_keys,
                cpu_transform=preprocessor.cpu_transform,
                coset_lut=coset_lut,
            )
        elif data_mode == "offline":
            strategy = OfflineDataStrategy(
                config=config,
                required_keys=preprocessor.required_data_keys,
                cpu_transform=preprocessor.cpu_transform,
                coset_lut=coset_lut,
            )
        else:
            raise ValueError(f"알 수 없는 data_mode: '{data_mode}'. 'online' 또는 'offline'을 사용하세요.")

        datamodule = QECDataModule(strategy=strategy)

        # 5. 코어 모델 생성 (전처리기의 출력 규격 주입)
        model_kwargs = preprocessor.get_model_kwargs()
        # coset mode: 2^num_observables class 출력, binary mode: num_observables 출력
        if coset_mode:
            model_kwargs["num_observables"] = 2 ** circuit.num_observables
        else:
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
    backend = config.simulation.backend

    if backend == 'stim':
        noise_configs = config.get_expanded_noise_configs()
        return SimulatorPool.from_stim(config.code, noise_configs)

    elif backend == 'pauli_plus':
        from qec_sim.circuit.pauli_plus import PauliPlusSimulator
        noise_configs = config.get_expanded_pauli_plus_configs()
        sims = [PauliPlusSimulator(config.code, n) for n in noise_configs]
        return SimulatorPool(sims)

    else:
        raise ValueError(f"알 수 없는 simulation backend: '{backend}'. 'stim' 또는 'pauli_plus'를 사용하세요.")
