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
import numba
from .frame import PauliFrame, PAULI_MUL, I, X, Y, Z, L2
from .leakage import _apply_heating_nb

# ---------------------------------------------------------------------------
# ZZ-dominant 2-qubit Pauli 분포
# ---------------------------------------------------------------------------

# 15개 non-identity 2-qubit Pauli 목록 (각각 (p0, p1) 쌍)
_PAULIS_2Q = [(a, b) for a in range(4) for b in range(4) if not (a == 0 and b == 0)]
_PAULIS_2Q_ARR = np.array(_PAULIS_2Q, dtype=np.int8)

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
# 동시 실행 CZ 쌍 식별 (layout 초기화 시 사용)
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
            if len({ca, ta, cb, tb}) < 4:
                continue
            pairs.append((ca, ta, cb, tb))
    return pairs


def build_crosstalk_arrays(cx_layer: list[tuple[int, int]]):
    """sub-layer의 sim_pairs를 평탄화해 batch numba 함수용 배열 반환.

    Returns
    -------
    pair_q0, pair_q1 : (n_flat,) int32 — 노이즈 적용 대상 (q0, q1) 쌍 (각 sim_pair → 2개)
    leak_qubits     : (n_flat*2,) int32 — leakage 적용 대상 큐빗 (각 sim_pair → 4개)
    """
    sim_pairs = get_simultaneous_pairs(cx_layer)
    flat_q0, flat_q1, flat_leak = [], [], []
    for (ca, ta, cb, tb) in sim_pairs:
        flat_q0.extend([ca, cb])
        flat_q1.extend([ta, tb])
        flat_leak.extend([ca, ta, cb, tb])
    return (
        np.array(flat_q0, dtype=np.int32),
        np.array(flat_q1, dtype=np.int32),
        np.array(flat_leak, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Numba batch crosstalk channel
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _apply_correlated_pauli2_sparse_nb(frame, q0s, q1s,
                                        trig_i, trig_pi, pauli_idx,
                                        pauli_pairs, pm):
    """
    Sparse 형식으로 trigger된 (shot, pair) 위치에만 Pauli 적용.
    crosstalk처럼 trigger 비율이 매우 낮은 채널에 적합 — 전체 (shots, n_pairs)
    cell 순회를 회피해 빈 cell 메모리 접근 비용 제거.

    trig_i, trig_pi : (n_trig,) int — np.where(trigger_mask)로 얻은 인덱스
    pauli_idx       : (n_trig,)   int32 — 가중치 분포에서 샘플된 _PAULIS_2Q 인덱스
    """
    n_trig = len(trig_i)
    for k in range(n_trig):
        i = trig_i[k]
        pi = trig_pi[k]
        q0 = q0s[pi]
        q1 = q1s[pi]
        idx = pauli_idx[k]
        p0 = pauli_pairs[idx, 0]
        p1 = pauli_pairs[idx, 1]
        f0 = frame[i, q0]
        f1 = frame[i, q1]
        if f0 < 4:
            frame[i, q0] = pm[f0, p0]
        if f1 < 4:
            frame[i, q1] = pm[f1, p1]


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def apply_crosstalk(
    frame: PauliFrame,
    pair_q0: np.ndarray,
    pair_q1: np.ndarray,
    leak_qubits: np.ndarray,
    p_crosstalk: float,
    alpha: float = 0.25,
    crosstalk_leakage_rate: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """한 CX sub-layer의 crosstalk 노이즈 적용.

    Parameters
    ----------
    frame                  : PauliFrame
    pair_q0, pair_q1       : (n_pairs,) int32 — layout.crosstalk_pair_arrays[li]
    leak_qubits            : (n_leak,) int32 — layout.crosstalk_leakage_qubit_arrays[li]
    p_crosstalk            : 쌍당 총 crosstalk 에러율 (기본 9.5e-4)
    alpha                  : 스케일 팩터 (Bausch: 0.25)
    crosstalk_leakage_rate : 한 게이트 당 crosstalk 유래 leakage 확률
    """
    if rng is None:
        rng = np.random.default_rng()

    eff_p = p_crosstalk * alpha
    n_pairs = len(pair_q0)

    # --- correlated Pauli (sparse) ---
    # crosstalk eff_p ~ 1e-4이라 trigger되는 cell 비율이 0.01% 수준.
    # np.where로 trigger 위치만 추출해 sparse 적용.
    if eff_p > 0 and n_pairs > 0:
        trig_mask = rng.random((frame.n_shots, n_pairs)) < eff_p
        trig_i, trig_pi = np.where(trig_mask)
        n_trig = len(trig_i)
        if n_trig > 0:
            pauli_idx = rng.choice(len(_PAULIS_2Q), size=n_trig,
                                   p=_weights_arr).astype(np.int32)
            _apply_correlated_pauli2_sparse_nb(
                frame.frame, pair_q0, pair_q1,
                trig_i.astype(np.int64), trig_pi.astype(np.int64),
                pauli_idx, _PAULIS_2Q_ARR, PAULI_MUL,
            )

    # --- crosstalk leakage (heating과 동일 메커니즘 — L2로 전이) ---
    if crosstalk_leakage_rate > 0 and len(leak_qubits) > 0:
        _apply_heating_nb(frame.frame, leak_qubits, crosstalk_leakage_rate,
                          rng.random((frame.n_shots, len(leak_qubits))))
