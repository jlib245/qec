# qec_sim/core/parameters.py

from dataclasses import dataclass

@dataclass
class NoiseParams:
    p_gate: float = 0.0      # 기본 게이트 노이즈
    p_meas: float = 0.0      # 측정 노이즈
    p_corr: float = 0.0      # 상관 관계 노이즈 (예: CNOT 직후 XX)
    p_leak: float = 0.0      # 누설(Leakage) 확률

@dataclass
class CodeParams:
    distance: int            # Surface code distance (d)
    rounds: int              # 신드롬 측정 반복 횟수