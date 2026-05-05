# qec_sim/circuit/pauli_plus/iq_noise.py
"""
IQ 측정 노이즈 모델 (Phase 2, Bausch et al. 2023).

noiseless 측정 결과(Pauli frame의 true state)에 사후 적용.
같은 Pauli+ 시뮬레이션 결과에 다양한 SNR / t_meas 파라미터를 효율적으로 적용 가능.

IQ 클러스터 (1D 투영, normalized σ=1):
  |0⟩  → z ~ N(0,   1)
  |1⟩  → z ~ N(snr, 1)  (T1 decay 보정 포함)
  |2⟩  → z ~ N(snr/2, 1)  (leakage 클러스터, |0⟩과 |1⟩ 사이)

T1 decay during measurement:
  p_decay = 1 - exp(-t_meas_over_T1)
  true |1⟩이라도 측정 중 |0⟩으로 decay할 수 있음.
  → P(z | true=1) = (1-p_decay)*N(z; snr, 1) + p_decay*N(z; 0, 1)

posterior:
  post1[i] = P(state=1 | z[i])
  post2[i] = P(state=2 | z[i])   (leakage)

soft detection event:
  soft_xor(a, b) = a + b - 2*a*b   (확률적 XOR)
"""

import numpy as np
import numba

_INV_SQRT_2PI = 0.3989422804014327  # 1 / sqrt(2π)


# ------------------------------------------------------------------ #
# Numba-JIT IQ batch operations                                       #
# ------------------------------------------------------------------ #

@numba.njit(cache=True)
def _get_true_z_state_batch_nb(frame, qubits):
    """
    모든 ancilla의 true Z-basis state를 일괄 추출.
    반환: (shots, n_qubits) int8  — 0=|0⟩, 1=|1⟩, 2=leaked
    """
    n_shots = frame.shape[0]
    n_q = len(qubits)
    out = np.zeros((n_shots, n_q), dtype=np.int8)
    for qi in range(n_q):
        q = qubits[qi]
        for i in range(n_shots):
            f = frame[i, q]
            if f >= 4:
                out[i, qi] = np.int8(2)
            elif f == 1 or f == 2:   # X or Y → |1⟩
                out[i, qi] = np.int8(1)
    return out


@numba.njit(cache=True)
def _sample_iq_batch_nb(true_states, snr, p_decay, noise, rand_decay):
    """
    모든 ancilla의 IQ 값을 일괄 샘플링.
    true_states: (shots, n_qubits) int8
    noise:       (shots, n_qubits) float — standard normal
    rand_decay:  (shots, n_qubits) float — T1 decay 여부 결정
    반환: z (shots, n_qubits) float
    """
    n_shots = true_states.shape[0]
    n_q     = true_states.shape[1]
    z = noise.copy()
    for qi in range(n_q):
        for i in range(n_shots):
            s = true_states[i, qi]
            if s == 1:
                if rand_decay[i, qi] >= p_decay:   # not decayed → |1⟩ 클러스터
                    z[i, qi] += snr
                # decayed → |0⟩ 클러스터, z 그대로
            elif s == 2:
                z[i, qi] += snr * 0.5              # leakage 클러스터
    return z


@numba.njit(cache=True)
def _compute_posterior_batch_nb(z, snr, p_decay):
    """
    scipy.stats.norm.pdf 대신 inline exp 계산.
    z: (shots, n_qubits) float
    반환: post1, post2 각각 (shots, n_qubits) float
    """
    n_shots = z.shape[0]
    n_q     = z.shape[1]
    post1 = np.empty((n_shots, n_q))
    post2 = np.empty((n_shots, n_q))

    for qi in range(n_q):
        for i in range(n_shots):
            zi = z[i, qi]
            # norm.pdf(zi, loc, scale=1) = INV_SQRT_2PI * exp(-0.5*(zi-loc)^2)
            # INV_SQRT_2PI는 세 likelihood 모두에 곱해지므로 정규화 시 상쇄 → 생략
            l0 = np.exp(-0.5 * zi * zi)
            l1 = ((1.0 - p_decay) * np.exp(-0.5 * (zi - snr) ** 2)
                  + p_decay       * np.exp(-0.5 * zi * zi))
            l2 = np.exp(-0.5 * (zi - snr * 0.5) ** 2)

            total = l0 + l1 + l2   # uniform prior
            if total == 0.0:
                total = 1.0
            post1[i, qi] = l1 / total
            post2[i, qi] = l2 / total

    return post1, post2


