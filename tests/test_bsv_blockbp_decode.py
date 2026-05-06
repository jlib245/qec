"""BSV BlockBP 전체 디코더 (C4) 테스트 — paper Algorithm 1 binary 버전.

검증:
- 결과 클래스가 0 또는 1
- delta_0 > delta_1이면 ValueError
- 작은 d 다중 syndrome에 대해 BlockBP 결과 == exact argmax (대부분 일치)
- fallback (수렴 못 했을 때) 동작
- syndrome=0이면 P(L=0|s) > P(L=1|s) → predicted=0
"""
import numpy as np
import pytest

from qec_sim.decoders._bsv_backend import FlatBackend
from qec_sim.decoders._bsv_blockbp import decode_blockbp
from qec_sim.decoders._bsv_peps import build_bsv_tn, contract_marginal


def _exact_argmax(d, p, syndrome) -> int:
    """열린 'L' TN을 exact contract → argmax over L."""
    tn = build_bsv_tn(d=d, p=p, syndrome_at_zstab=syndrome)
    marg = contract_marginal(tn)
    return int(np.argmax(marg))


# ──────────────────────────────────────────────
# 입력 검증
# ──────────────────────────────────────────────

def test_invalid_thresholds():
    with pytest.raises(ValueError):
        decode_blockbp(
            d=3, p=0.05, syndrome_at_zstab={}, k=3,
            max_iter=10, delta_0=0.1, delta_1=0.05, backend=FlatBackend(max_bond=4),
        )


def test_returns_valid_class():
    res = decode_blockbp(
        d=3, p=0.05, syndrome_at_zstab={(0, 1): 1}, k=3,
        max_iter=50, delta_0=1e-5, delta_1=1e-3, backend=FlatBackend(max_bond=8),
    )
    assert res.predicted_class in (0, 1)
    assert len(res.per_class) == 2


# ──────────────────────────────────────────────
# 기본 sanity
# ──────────────────────────────────────────────

def test_zero_syndrome_predicts_no_flip():
    """syndrome 모두 0 → 가장 가능성 높은 건 에러 없음 → class 0."""
    res = decode_blockbp(
        d=3, p=0.05, syndrome_at_zstab={}, k=3,
        max_iter=50, delta_0=1e-6, delta_1=1e-3, backend=FlatBackend(max_bond=16),
        damping=0.5,
    )
    assert res.predicted_class == 0


# ──────────────────────────────────────────────
# Exact argmax와 일치 — 다양한 syndrome
# ──────────────────────────────────────────────

@pytest.mark.parametrize("syndrome", [
    {(0, 1): 1},
    {(2, 1): 1},
    {(0, 1): 1, (2, 3): 1},
    {(0, 3): 1, (4, 1): 1},
    {(2, 1): 1, (2, 3): 1, (4, 1): 1},
])
def test_blockbp_argmax_matches_exact_d3(syndrome):
    """d=3 k=3 χ=16 충분한 max_iter에서 BlockBP 결정 == exact argmax."""
    exact = _exact_argmax(d=3, p=0.07, syndrome=syndrome)
    res = decode_blockbp(
        d=3, p=0.07, syndrome_at_zstab=syndrome, k=3,
        max_iter=200, delta_0=1e-6, delta_1=1e-2, backend=FlatBackend(max_bond=16),
        damping=0.5,
    )
    assert res.predicted_class == exact, (
        f"syndrome={syndrome}: BlockBP={res.predicted_class} ≠ exact={exact} "
        f"(fallback={res.fallback_used}, "
        f"per_class deltas={[(c.class_bit, c.bp.final_delta) for c in res.per_class]})"
    )


# ──────────────────────────────────────────────
# Fallback (수렴 강제 실패)
# ──────────────────────────────────────────────

def test_fallback_when_max_iter_too_small():
    """max_iter=1, delta_0/delta_1 매우 작게 → 둘 다 수렴 못 함 → fallback."""
    res = decode_blockbp(
        d=3, p=0.05, syndrome_at_zstab={(0, 1): 1}, k=3,
        max_iter=1, delta_0=1e-12, delta_1=1e-12, backend=FlatBackend(max_bond=4),
    )
    assert res.fallback_used
    assert res.predicted_class in (0, 1)
