# qec_sim/circuit/builder.py
from abc import ABC, abstractmethod
import stim

# 설정 스키마 및 레지스트리 임포트
from qec_sim.config.schema import CodeParams, NoiseParams
from qec_sim.circuit.registry import register_builder

# 1. 모든 빌더가 지켜야 할 공통 규격 (인터페이스)
class BaseCircuitBuilder(ABC):
    def __init__(self, code_params: CodeParams, noise_params: NoiseParams, **kwargs):
        self.code_params = code_params
        self.noise_params = noise_params
        self.kwargs = kwargs
        
    @abstractmethod
    def build(self) -> stim.Circuit:
        pass

# 2. 데코레이터로 "surface_code"라는 이름 등록.
@register_builder("surface_code")
class SurfaceCodeBuilder(BaseCircuitBuilder):
    def __init__(self, code_params: CodeParams, noise_params: NoiseParams, **kwargs):
        """
        양자 회로를 생성하는 빌더 클래스.
        노이즈 파라미터가 리스트인 경우 첫 번째 값을 기준으로 대표 회로를 구성.
        """
        super().__init__(code_params, noise_params, **kwargs)
        self.code = code_params
        
        self.p_gate = noise_params.p_gate[0] if isinstance(noise_params.p_gate, list) else noise_params.p_gate
        self.p_meas = noise_params.p_meas[0] if isinstance(noise_params.p_meas, list) else noise_params.p_meas
        self.p_corr = noise_params.p_corr[0] if isinstance(noise_params.p_corr, list) else noise_params.p_corr
        self.p_leak = noise_params.p_leak[0] if isinstance(noise_params.p_leak, list) else noise_params.p_leak

    def build(self) -> stim.Circuit:
        """
        Stim의 내장 표면 코드 생성기를 사용하여 회로를 빌드.
        모든 인자는 단일 float 값이 전달되어 경고가 발생하지 않습니다.
        """
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.code.distance,
            rounds=self.code.rounds,
            after_clifford_depolarization=self.p_gate,
            before_measure_flip_probability=self.p_meas,
            # p_leak은 simulator.py에서 별도 후처리, -> Herald Erasure로 처리 가능..? -> but 회로 다 뜯어야 할 것
            # p_corr은 현재 미구현.
        )
        return circuit