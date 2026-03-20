# qec_sim/circuit/pauli_plus/crosstalk.py
"""
Phenomenological crosstalk model (Phase 3).

Nature 614 Supplementary, Section XIV.A.4:
- 물리적 원인: 비활성 커플러의 불완전 상쇄 + 대각선 capacitive coupling (~200 kHz)
- 원래 모델: 동시 실행 게이트 쌍에 대해 4-level qubit 시뮬레이션 → U_I → Generalized Pauli Twirling
- 구체적 U_I 값 미공개 ("deferred to future work") → 페놀로지컬 근사 사용

페놀로지컬 모델:
- 총 에러율: p_crosstalk = 9.5e-4 (Nature 614 Table I) × alpha (Bausch: 0.25)
- 에러 분포: ZZ-dominant (잔여 ZZ coupling + capacitive swap이 주 원인)
- Crosstalk leakage: 별도 확률로 L2 전이

대상 쌍:
- 같은 CX sub-layer에서 동시 실행되는 게이트 중 nearest/diagonal neighbor 쌍
  (CZ-CZ, CZ-idle도 포함하나 여기서는 CZ-CZ만 처리)
"""

import numpy as np
from .frame import PauliFrame, PAULI_MUL, I, X, Y, Z, L2

# ---------------------------------------------------------------------------
# ZZ-dominant 2-qubit Pauli 분포
# ---------------------------------------------------------------------------

# 15개 non-identity 2-qubit Pauli 목록 (각각 (p0, p1) 쌍)
_PAULIS_2Q = [(a, b) for a in range(4) for b in range(4) if not (a == 0 and b == 0)]

# ZZ-dominant 가중치:
# - ZZ: 가장 높음 (잔여 ZZ coupling)
# - ZI, IZ: 중간 (단일 큐빗 Z 오류)
# - ZX, XZ, ZY, YZ: 낮음 (ZZ + 교차)
# - 나머지: 매우 낮음
_ZZ_WEIGHTS_RAW = {
    (Z, Z): 5.0,   # ZZ coupling 주 원인
    (Z, I): 2.0,   # residual ZZ → single-qubit Z
    (I, Z): 2.0,
    (Z, X): 0.5,
    (X, Z): 0.5,
    (Z, Y): 0.5,
    (Y, Z): 0.5,
    (X, X): 0.3,   # capacitive swap 기여
    (Y, Y): 0.3,
    (X, Y): 0.1,
    (Y, X): 0.1,
    (X, I): 0.2,
    (I, X): 0.2,
    (Y, I): 0.1,
    (I, Y): 0.1,
}

_weights_arr = np.array(
    [_ZZ_WEIGHTS_RAW.get(p, 0.05) for p in _PAULIS_2Q], dtype=float
)
_weights_arr /= _weights_arr.sum()
_CUMWEIGHTS = np.cumsum(_weights_arr)

# ---------------------------------------------------------------------------
# 동시 실행 CZ 쌍 식별
# ---------------------------------------------------------------------------

def get_simultaneous_pairs(cx_layer: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """
    하나의 CX sub-layer 내에서 인접(nearest/diagonal) CZ-CZ 쌍을 찾는다.

    반환: [(ctrl_a, tgt_a, ctrl_b, tgt_b), ...] — 공유 큐빗 없는 쌍만
    """
    pairs = []
    n = len(cx_layer)
    for i in range(n):
        for j in range(i + 1, n):
            ca, ta = cx_layer[i]
            cb, tb = cx_layer[j]
            # 큐빗이 겹치면 skip (같은 큐빗에 두 게이트 동시 불가)
            if len({ca, ta, cb, tb}) < 4:
                continue
            pairs.append((ca, ta, cb, tb))
    return pairs


# ---------------------------------------------------------------------------
# Crosstalk 채널
# ---------------------------------------------------------------------------

def apply_crosstalk(
    frame: PauliFrame,
    cx_layer: list[tuple[int, int]],
    p_crosstalk: float,
    alpha: float = 0.25,
    crosstalk_leakage_rate: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """
    하나의 CX sub-layer에 대해 동시 실행 쌍마다 crosstalk 채널 적용.

    파라미터
    --------
    cx_layer : 이번 sub-layer의 (ctrl, tgt) 쌍 목록
    p_crosstalk : 쌍당 총 crosstalk 에러율 (기본값 9.5e-4, Nature 614 Table I)
    alpha : 스케일 팩터 (1.0=원래, 0.25=Bausch 논문)
    crosstalk_leakage_rate : 한 게이트 당 crosstalk 유래 leakage 확률
    """
    if rng is None:
        rng = np.random.default_rng()

    eff_p = p_crosstalk * alpha
    if eff_p <= 0 and crosstalk_leakage_rate <= 0:
        return

    sim_pairs = get_simultaneous_pairs(cx_layer)
    for (ca, ta, cb, tb) in sim_pairs:
        # --- correlated Pauli 에러 (2개 게이트 큐빗에 각각 적용) ---
        if eff_p > 0:
            _apply_correlated_pauli2(frame, ca, ta, eff_p, rng)
            _apply_correlated_pauli2(frame, cb, tb, eff_p, rng)

        # --- crosstalk leakage ---
        if crosstalk_leakage_rate > 0:
            for q in (ca, ta, cb, tb):
                _apply_crosstalk_leakage(frame, q, crosstalk_leakage_rate, rng)


def _apply_correlated_pauli2(
    frame: PauliFrame, q0: int, q1: int, p: float, rng: np.random.Generator
):
    """
    ZZ-dominant 2-qubit Pauli 채널 (q0, q1 쌍에 적용).
    leaked 큐빗은 건드리지 않음.
    """
    n = frame.n_shots
    trigger = rng.random(n) < p

    if not trigger.any():
        return

    # ZZ-dominant 분포에서 Pauli 선택
    u = rng.random(trigger.sum())
    idx = np.searchsorted(_CUMWEIGHTS, u)
    idx = np.clip(idx, 0, len(_PAULIS_2Q) - 1)
    p0_arr = np.array([_PAULIS_2Q[i][0] for i in idx], dtype=np.int8)
    p1_arr = np.array([_PAULIS_2Q[i][1] for i in idx], dtype=np.int8)

    f0 = frame.frame[trigger, q0]
    f1 = frame.frame[trigger, q1]

    # leaked 큐빗은 스킵 (mask out)
    normal0 = f0 < 4
    normal1 = f1 < 4

    new_f0 = f0.copy()
    new_f1 = f1.copy()
    new_f0[normal0] = PAULI_MUL[f0[normal0], p0_arr[normal0]]
    new_f1[normal1] = PAULI_MUL[f1[normal1], p1_arr[normal1]]

    frame.frame[trigger, q0] = new_f0
    frame.frame[trigger, q1] = new_f1


def _apply_crosstalk_leakage(
    frame: PauliFrame, qubit: int, p: float, rng: np.random.Generator
):
    """crosstalk 유래 leakage: computational subspace → L2."""
    f = frame.frame[:, qubit]
    normal = f < 4
    leaks = normal & (rng.random(frame.n_shots) < p)
    frame.frame[leaks, qubit] = L2
