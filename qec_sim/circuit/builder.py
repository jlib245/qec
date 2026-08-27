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
@register_builder("surface_code_basic")
class SurfaceCodeBuilder(BaseCircuitBuilder):
    def __init__(self, code_params: CodeParams, noise_params: NoiseParams, **kwargs):
        """
        양자 회로를 생성하는 빌더 클래스.
        노이즈 파라미터가 리스트인 경우 첫 번째 값을 기준으로 대표 회로를 구성.
        """
        super().__init__(code_params, noise_params, **kwargs)

        self.p_gate = noise_params.p_gate[0] if isinstance(noise_params.p_gate, list) else noise_params.p_gate
        self.p_meas = noise_params.p_meas[0] if isinstance(noise_params.p_meas, list) else noise_params.p_meas
        self.p_corr = noise_params.p_corr[0] if isinstance(noise_params.p_corr, list) else noise_params.p_corr

    def build(self) -> stim.Circuit:
        """
        Stim의 내장 표면 코드 생성기를 사용하여 회로를 빌드.
        모든 인자는 단일 float 값이 전달되어 경고가 발생하지 않습니다.
        """
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.code_params.distance,
            rounds=self.code_params.rounds,
            after_clifford_depolarization=self.p_gate,
            before_measure_flip_probability=self.p_meas,
            # p_corr은 현재 미구현.
        )
        return circuit


@register_builder("surface_code_si1000")
class SurfaceCodeSI1000Builder(BaseCircuitBuilder):
    """
    Canonical SI1000 (Gidney 2021) on top of stim's rotated_memory_z generator.

    Step 1: zero-noise base circuit from stim.Circuit.generated.
    Step 2: apply NoiseModel.SI1000(p) — 1q gate p/10, 2q gate p, idle p/10,
            measure_reset_idle 2p, R 2p, M 5p — exactly matching the
            honeycomb_threshold reference implementation.
    """

    def __init__(self, code_params: CodeParams, noise_params: NoiseParams, **kwargs):
        super().__init__(code_params, noise_params, **kwargs)
        self.p = noise_params.p_gate[0] if isinstance(noise_params.p_gate, list) else noise_params.p_gate

    def build(self) -> stim.Circuit:
        from qec_sim.circuit.noise_model import NoiseModel
        base = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.code_params.distance,
            rounds=self.code_params.rounds,
        )
        return NoiseModel.SI1000(self.p).noisy_circuit(base)


@register_builder("stim_file")
class StimFileBuilder(BaseCircuitBuilder):
    """디스크의 .stim 회로를 그대로 로드 (노이즈는 파일에 baked-in).

    surface code가 아닌 외부 회로(BB/qLDPC 등) eval용. noise_params는 무시 —
    회로에 이미 노이즈가 적용돼 있음. code_params.stim_path 필수.
    glob 패턴이면 정확히 하나만 매치해야 함 (다중 파일 sweep은 eval_pipeline이
    파일별로 개별 처리; 이 빌더는 build_circuit 단일 회로 계약을 지킴).
    """

    def build(self) -> stim.Circuit:
        import glob as _glob
        path = self.code_params.stim_path
        if not path:
            raise ValueError("stim_file 빌더는 code.stim_path 명시 필수.")
        matches = sorted(_glob.glob(path))
        if len(matches) == 0:
            raise FileNotFoundError(f"stim_path에 매치되는 파일 없음: {path}")
        if len(matches) > 1:
            raise ValueError(
                f"stim_path glob이 {len(matches)}개 매치 — 단일 회로 빌더엔 하나만 허용. "
                f"다중 파일 sweep은 eval_pipeline(engine=sinter)을 사용하세요. 매치: {matches[:3]}..."
            )
        return stim.Circuit.from_file(matches[0])


def build_si1000_canonical(distance: int, rounds: int, p: float) -> stim.Circuit:
    """Convenience helper: build a canonical-SI1000 rotated_memory_z circuit."""
    from qec_sim.circuit.noise_model import NoiseModel
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance, rounds=rounds,
    )
    return NoiseModel.SI1000(p).noisy_circuit(base)