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
from scipy.stats import norm
from .frame import PauliFrame, X, Y


# ------------------------------------------------------------------ #
# True state extraction                                               #
# ------------------------------------------------------------------ #

def get_true_z_state(frame: PauliFrame, qubit: int) -> np.ndarray:
    """
    측정 전 qubit의 true Z-basis state를 반환.
    Returns (n_shots,) int array: 0=|0⟩, 1=|1⟩, 2=leaked
    frame을 변경하지 않음 (read-only).
    """
    f = frame.frame[:, qubit]
    return np.where(f >= 4, 2,
           np.where((f == X) | (f == Y), 1, 0)).astype(np.int8)


# ------------------------------------------------------------------ #
# IQ sampling                                                         #
# ------------------------------------------------------------------ #

def sample_iq(true_state: np.ndarray, snr: float,
              t_meas_over_T1: float,
              rng: np.random.Generator) -> np.ndarray:
    """
    true_state: (n_shots,) int {0, 1, 2}
    Returns z: (n_shots,) float — IQ 투영값
    """
    n = len(true_state)
    z = rng.standard_normal(n)  # 기본 노이즈 (σ=1)

    # |0⟩: center 0 (z 그대로)
    # |1⟩: center snr, T1 decay 보정
    p_decay = 1.0 - np.exp(-t_meas_over_T1) if t_meas_over_T1 > 0 else 0.0

    is_1 = true_state == 1
    is_2 = true_state == 2

    # |1⟩인 shot 중 decay된 것은 |0⟩ 클러스터에서 샘플
    decayed = is_1 & (rng.random(n) < p_decay)
    not_decayed = is_1 & ~decayed

    z[not_decayed] += snr        # |1⟩ 클러스터
    # decayed는 |0⟩ 클러스터 → z 그대로
    z[is_2] += snr / 2           # leakage 클러스터 (|0⟩과 |1⟩ 사이)

    return z


# ------------------------------------------------------------------ #
# Posterior computation                                               #
# ------------------------------------------------------------------ #

def compute_posterior(z: np.ndarray, snr: float,
                      t_meas_over_T1: float,
                      prior: np.ndarray | None = None) -> tuple:
    """
    z: (n_shots,) IQ 값
    prior: (n_shots, 3) — [P(0), P(1), P(2)], None이면 uniform
    Returns:
      post1: (n_shots,) P(state=1 | z)
      post2: (n_shots,) P(state=2 | z)
    """
    n = len(z)
    p_decay = 1.0 - np.exp(-t_meas_over_T1) if t_meas_over_T1 > 0 else 0.0

    # Likelihood P(z | state)
    like_0 = norm.pdf(z, loc=0,       scale=1)
    like_1 = ((1 - p_decay) * norm.pdf(z, loc=snr,     scale=1)
              + p_decay      * norm.pdf(z, loc=0,       scale=1))
    like_2 = norm.pdf(z, loc=snr / 2, scale=1)

    if prior is None:
        prior = np.ones((n, 3)) / 3

    unnorm = np.stack([
        prior[:, 0] * like_0,
        prior[:, 1] * like_1,
        prior[:, 2] * like_2,
    ], axis=1)

    total = unnorm.sum(axis=1, keepdims=True)
    total = np.where(total == 0, 1.0, total)  # div-by-zero 방지
    posterior = unnorm / total

    return posterior[:, 1], posterior[:, 2]  # post1, post2


# ------------------------------------------------------------------ #
# Soft detection event                                                #
# ------------------------------------------------------------------ #

def soft_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    두 soft bit (확률)의 XOR 확률.
    soft_xor(a, b) = a + b - 2*a*b
    """
    return a + b - 2 * a * b


def soft_xor_consecutive(post1_rounds: np.ndarray) -> np.ndarray:
    """
    post1_rounds: (shots, rounds, n_ancilla)
    Returns soft_det: (shots, rounds-1, n_ancilla) soft detection events
    """
    return soft_xor(post1_rounds[:, 1:, :], post1_rounds[:, :-1, :])


# ------------------------------------------------------------------ #
# Hard threshold                                                       #
# ------------------------------------------------------------------ #

def threshold(post1: np.ndarray) -> np.ndarray:
    """soft → binary. 최종 data qubit 측정에 반드시 적용."""
    return post1 > 0.5
