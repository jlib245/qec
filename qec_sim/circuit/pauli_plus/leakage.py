# qec_sim/circuit/pauli_plus/leakage.py
"""
Leakage 전이 모델 (Phase 2).

Bausch et al. 2023 Appendix A.1.9 기반.
세 가지 leakage 경로:
  A. CZ dephasing leakage: CZ 게이트 중 |2⟩ 전이
  B. Heating rate: 열적 여기
  C. Crosstalk leakage: Phase 3에서 처리

Leakage 제거:
  - 측정 후 multi-level reset → leakage removal
  - 매 cycle data qubit leakage removal (Miao et al. 2022)
  - T1 passive decay

States: I=0, X=1, Y=2, Z=3, L2=4, L3=5
"""

import numpy as np
from .frame import PauliFrame, I, X, Y, Z, L2, L3
from .channels import depolarize1


def apply_cz_leakage(frame: PauliFrame, ctrl: int, tgt: int,
                     p_leakage: float, rng: np.random.Generator):
    """
    경로 A: CZ 게이트 중 dephasing leakage.
    각 큐빗이 독립적으로 확률 p_leakage로 L2로 전이.
    leaked → non-leaked 큐빗에는 1q depolarizing 적용 (Pauli+ spec).
    """
    if p_leakage <= 0:
        return

    fc = frame.frame[:, ctrl]
    ft = frame.frame[:, tgt]

    ctrl_normal = fc < 4
    tgt_normal  = ft < 4

    # 전이 발생 여부
    ctrl_leaks = ctrl_normal & (rng.random(frame.n_shots) < p_leakage)
    tgt_leaks  = tgt_normal  & (rng.random(frame.n_shots) < p_leakage)

    # leaked → non-leaked: non-leaked 큐빗에 1q depolarizing
    # (이미 apply_cx에서 Clifford 적용 완료됐으므로 noise만 추가)
    depolarize1(frame, tgt,  p=0.5, rng=rng,
                mask=ctrl_leaks & tgt_normal & ~tgt_leaks)
    depolarize1(frame, ctrl, p=0.5, rng=rng,
                mask=tgt_leaks & ctrl_normal & ~ctrl_leaks)

    # 전이 적용: leaked 큐빗은 L2로, frame의 Pauli 정보는 소실
    frame.frame[ctrl_leaks, ctrl] = L2
    frame.frame[tgt_leaks,  tgt]  = L2


def apply_heating(frame: PauliFrame, qubit: int,
                  heating_rate_per_us: float, gate_time_us: float,
                  rng: np.random.Generator):
    """
    경로 B: 열적 여기 (heating).
    확률 rate * gate_time 으로 computational 큐빗이 L2로 전이.
    """
    p = heating_rate_per_us * gate_time_us
    if p <= 0:
        return

    f = frame.frame[:, qubit]
    normal = f < 4
    leaks  = normal & (rng.random(frame.n_shots) < p)
    frame.frame[leaks, qubit] = L2


def remove_leakage(frame: PauliFrame, qubits: list,
                   rng: np.random.Generator):
    """
    Leakage removal (multi-level reset).
    Leaked 큐빗을 |0⟩ (I)으로 되돌림.
    Non-leaked 큐빗은 그대로.
    """
    for q in qubits:
        f = frame.frame[:, q]
        leaked = f >= 4
        frame.frame[leaked, q] = I


def apply_passive_decay(frame: PauliFrame, qubits: list,
                        T1_us: float, t_us: float,
                        rng: np.random.Generator):
    """
    T1 passive decay: 확률 1 - exp(-t/T1) 로 L2 → I (amplitude damping).
    Computational 큐빗의 passive leakage는 heating_rate로 처리하므로,
    여기서는 leaked 큐빗의 decay만 처리.
    """
    if T1_us <= 0 or t_us <= 0:
        return

    p_decay = 1.0 - np.exp(-t_us / T1_us)

    for q in qubits:
        f = frame.frame[:, q]
        leaked  = f >= 4
        decays  = leaked & (rng.random(frame.n_shots) < p_decay)
        frame.frame[decays, q] = I


def leaked_mask(frame: PauliFrame, qubit: int) -> np.ndarray:
    """(n_shots,) bool: 해당 qubit이 leaked 상태인 shot."""
    return frame.frame[:, qubit] >= 4
