"""CX Pauli-frame conjugation table 정확성.

commit df6771f에서 수정된 transpose 버그 회귀 방지용.
Heisenberg 규칙: X_c→X_cX_t, Z_c→Z_c, X_t→X_t, Z_t→Z_cZ_t.
"""
import numpy as np
from qec_sim.circuit.pauli_plus.frame import _CX_C, _CX_T, I, X, Y, Z, PAULI_MUL


# 정답: 손으로 계산한 CX 변환 테이블.
# (fc, ft) → (new_c, new_t)
_EXPECTED = {
    (I, I): (I, I), (I, X): (I, X), (I, Y): (Z, Y), (I, Z): (Z, Z),
    (X, I): (X, X), (X, X): (X, I), (X, Y): (Y, Z), (X, Z): (Y, Y),
    (Y, I): (Y, X), (Y, X): (Y, I), (Y, Y): (X, Z), (Y, Z): (X, Y),
    (Z, I): (Z, I), (Z, X): (Z, X), (Z, Y): (I, Y), (Z, Z): (I, Z),
}


def test_cx_table_all_pairs():
    """16개 (control, target) Pauli 조합 모두 검증."""
    for (fc, ft), (expected_c, expected_t) in _EXPECTED.items():
        actual_c = _CX_C[fc, ft]
        actual_t = _CX_T[fc, ft]
        assert actual_c == expected_c, (
            f"CX_C[{fc}, {ft}] = {actual_c}, expected {expected_c}"
        )
        assert actual_t == expected_t, (
            f"CX_T[{fc}, {ft}] = {actual_t}, expected {expected_t}"
        )


def test_cx_x_propagates_control_to_target():
    """X on control: X_c → X_c X_t (target에 X 복사, control X 유지)."""
    # 입력 (X, I) → (X, X)
    assert _CX_C[X, I] == X
    assert _CX_T[X, I] == X


def test_cx_z_propagates_target_to_control():
    """Z on target: Z_t → Z_c Z_t (control에 Z 복사, target Z 유지)."""
    # 입력 (I, Z) → (Z, Z)
    assert _CX_C[I, Z] == Z
    assert _CX_T[I, Z] == Z
