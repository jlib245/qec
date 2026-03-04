# qec_sim/core/parameters.py
import itertools
from dataclasses import dataclass

@dataclass
class NoiseParams:
    p_gate: float = 0.0      # 기본 게이트 노이즈
    p_meas: float = 0.0      # 측정 노이즈
    p_corr: float = 0.0      # 상관 관계 노이즈
    p_leak: float = 0.0      # 누설 확률

def get_noise_combinations(noise_dict: dict) -> list[NoiseParams]:
    """YAML 설정의 리스트들을 조합하여 모든 경우의 수(Cartesian Product)를 생성합니다."""
    # 스칼라 값도 단일 요소 리스트로 변환
    lists = {k: (v if isinstance(v, list) else [v]) for k, v in noise_dict.items()}
    keys, values = zip(*lists.items())
    
    # 모든 가능한 조합 생성 (예: p_gate 3개 x p_leak 3개 = 9개 조합)
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return [NoiseParams(**kwargs) for kwargs in combinations]

@dataclass
class CodeParams:
    distance: int
    rounds: int