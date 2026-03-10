from qec_sim.config.schema import ExperimentConfig
from qec_sim.circuit.registry import build_circuit
from qec_sim.models.wrapper import PreprocessorWrapper
from qec_sim.models.registry import get_model_class
from qec_sim.data.registry import get_preprocessor_class

class ComponentFactory:
    @staticmethod
    def build_system(config: ExperimentConfig):
        circuit = build_circuit(config.code.name, config.code, config.get_expanded_noise_configs()[0]).build()
        
        # 1. 모델 클래스 확인 및 전처리기 지명 확인
        core_model_cls = get_model_class(config.model.name)
        preprocessor_name = getattr(core_model_cls, 'REQUIRED_PREPROCESSOR', 'flat')
        
        # 2. 전처리기 생성 (정책서 발급)
        preprocessor_cls = get_preprocessor_class(preprocessor_name)
        preprocessor = preprocessor_cls(
            detector_coords=circuit.get_detector_coordinates(),
            num_detectors=circuit.num_detectors,
            use_erasures=getattr(config.training, 'use_erasures', True)
        )
        
        # 3. [상위 종속] 데이터 모듈 생성 (전처리기의 요구사항 주입)
        strategy = OfflineDataStrategy(
            config=config,
            required_keys=preprocessor.required_data_keys,  # 무조건 가져와야 할 데이터 명시
            cpu_transform=preprocessor.cpu_transform        # 데이터 로더에 꽂아줄 CPU 함수
        )
        datamodule = QECDataModule(strategy=strategy)
        
        # 4. [하위 종속] 코어 모델 생성 (전처리기의 출력 규격 주입)
        model_kwargs = preprocessor.get_model_kwargs()
        model_kwargs["num_observables"] = circuit.num_observables
        model_kwargs["code_distance"] = config.code.distance
        model_kwargs.update(config.model.kwargs)
        
        core_model = core_model_cls(**model_kwargs)
        
        # 5. [래핑] 팩토리 최종 산출물 완성 (코어 모델 + GPU 전처리 함수)
        wrapped_model = PreprocessorWrapper(core_model, preprocessor.gpu_transform)
        
        return datamodule, wrapped_model