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
import numba
from .frame import PauliFrame


# ------------------------------------------------------------------ #
# Numba-JIT batch leakage operations                                  #
# ------------------------------------------------------------------ #

@numba.njit(cache=True)
def _remove_leakage_nb(frame, qubits):
    """Leaked 큐빗(≥4)을 I(0)으로 일괄 리셋."""
    n_shots = frame.shape[0]
    for qi in range(len(qubits)):
        q = qubits[qi]
        for i in range(n_shots):
            if frame[i, q] >= 4:
                frame[i, q] = 0  # I


@numba.njit(cache=True)
def _apply_passive_decay_nb(frame, qubits, p_decay, rand):
    """
    T1 passive decay: leaked 큐빗이 확률 p_decay로 I로 돌아옴.
    rand: (shots, n_qubits) float
    """
    n_shots = frame.shape[0]
    for qi in range(len(qubits)):
        q = qubits[qi]
        for i in range(n_shots):
            if frame[i, q] >= 4 and rand[i, qi] < p_decay:
                frame[i, q] = 0  # I


@numba.njit(cache=True)
def _apply_cz_leakage_layer_nb(frame, ctrls, tgts, p_leakage,
                                 rand_ctrl, rand_tgt,
                                 rand_dep_ctrl, rand_dep_tgt, pm):
    """
    CZ dephasing leakage를 한 sub-layer의 모든 쌍에 일괄 적용.

    rand_ctrl, rand_tgt:         (shots, n_pairs) — leakage 발생 여부
    rand_dep_ctrl, rand_dep_tgt: (shots, n_pairs) — crossed 경우 depolarizing용
    """
    n_shots = frame.shape[0]
    p3 = 0.5 / 3.0  # depolarize1(p=0.5) → 각 Pauli p/3

    for pi in range(len(ctrls)):
        c = ctrls[pi]
        t = tgts[pi]
        for i in range(n_shots):
            fc = frame[i, c]
            ft = frame[i, t]
            ctrl_normal = fc < 4
            tgt_normal  = ft < 4

            ctrl_leaks = ctrl_normal and (rand_ctrl[i, pi] < p_leakage)
            tgt_leaks  = tgt_normal  and (rand_tgt[i,  pi] < p_leakage)

            # crossed: ctrl leaked → tgt에 1q depolarizing
            if ctrl_leaks and tgt_normal and not tgt_leaks:
                r = rand_dep_tgt[i, pi]
                if   r < p3:       frame[i, t] = pm[ft, np.int8(1)]
                elif r < 2.0*p3:   frame[i, t] = pm[ft, np.int8(2)]
                elif r < 0.5:      frame[i, t] = pm[ft, np.int8(3)]

            # crossed: tgt leaked → ctrl에 1q depolarizing
            if tgt_leaks and ctrl_normal and not ctrl_leaks:
                r = rand_dep_ctrl[i, pi]
                if   r < p3:       frame[i, c] = pm[fc, np.int8(1)]
                elif r < 2.0*p3:   frame[i, c] = pm[fc, np.int8(2)]
                elif r < 0.5:      frame[i, c] = pm[fc, np.int8(3)]

            # 전이 적용
            if ctrl_leaks:
                frame[i, c] = np.int8(4)  # L2
            if tgt_leaks:
                frame[i, t] = np.int8(4)  # L2


@numba.njit(cache=True)
def _apply_heating_nb(frame, qubits, p, rand):
    """
    Heating: computational 큐빗이 확률 p로 L2로 전이.
    rand: (shots, n_qubits) float
    """
    n_shots = frame.shape[0]
    for qi in range(len(qubits)):
        q = qubits[qi]
        for i in range(n_shots):
            if frame[i, q] < 4 and rand[i, qi] < p:
                frame[i, q] = np.int8(4)  # L2


def remove_leakage(frame: PauliFrame, qubits,
                   rng: np.random.Generator):
    """
    Leakage removal (multi-level reset).
    Leaked 큐빗을 |0⟩ (I)으로 되돌림.
    Non-leaked 큐빗은 그대로.
    """
    q_arr = qubits if isinstance(qubits, np.ndarray) else np.array(qubits, dtype=np.int32)
    _remove_leakage_nb(frame.frame, q_arr)


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
    q_arr = qubits if isinstance(qubits, np.ndarray) else np.array(qubits, dtype=np.int32)
    _apply_passive_decay_nb(frame.frame, q_arr, p_decay,
                            rng.random((frame.n_shots, len(q_arr))))
