import stim
from .parameters import CodeParams, NoiseParams

class CustomCircuitBuilder:
    def __init__(self, code_params: CodeParams, noise_params: NoiseParams):
        """
        양자 회로를 생성하는 빌더 클래스.
        노이즈 파라미터가 리스트인 경우 첫 번째 값을 기준으로 기본 회로를 구성.
        """
        self.code = code_params
        
        # 각 노이즈 파라미터가 리스트일 경우를 대비해 단일 float 값으로 정규화
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
            # p_corr나 p_leak은 기본 생성기 인자에 없을 수 있으나, 
            # 커스텀 회로 구성 시 위에서 정규화한 속성들을 사용하면 됩니다.
        )
        return circuit